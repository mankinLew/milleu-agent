from agents import function_tool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
from openai import AsyncOpenAI
from types import SimpleNamespace
from guardrails.runtime import load_config_bundle, instantiate_guardrails, run_guardrails
from pydantic import BaseModel

# Tool definitions
@function_tool
def get_retention_offers(customer_id: str, account_type: str, current_plan: str, tenure_months: integer, recent_complaints: bool):
  pass

# Shared client for guardrails and file search
client = AsyncOpenAI()
ctx = SimpleNamespace(guardrail_llm=client)
# Guardrails definitions
jailbreak_guardrail_config = {
  "guardrails": [
    { "name": "Jailbreak", "config": { "model": "gpt-5-nano", "confidence_threshold": 0.7 } }
  ]
}
def guardrails_has_tripwire(results):
    return any((hasattr(r, "tripwire_triggered") and (r.tripwire_triggered is True)) for r in (results or []))

def get_guardrail_safe_text(results, fallback_text):
    for r in (results or []):
        info = (r.info if hasattr(r, "info") else None) or {}
        if isinstance(info, dict) and ("checked_text" in info):
            return info.get("checked_text") or fallback_text
    pii = next(((r.info if hasattr(r, "info") else {}) for r in (results or []) if isinstance((r.info if hasattr(r, "info") else None) or {}, dict) and ("anonymized_text" in ((r.info if hasattr(r, "info") else None) or {}))), None)
    if isinstance(pii, dict) and ("anonymized_text" in pii):
        return pii.get("anonymized_text") or fallback_text
    return fallback_text

async def scrub_conversation_history(history, config):
    try:
        guardrails = (config or {}).get("guardrails") or []
        pii = next((g for g in guardrails if (g or {}).get("name") == "Contains PII"), None)
        if not pii:
            return
        pii_only = {"guardrails": [pii]}
        for msg in (history or []):
            content = (msg or {}).get("content") or []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_text" and isinstance(part.get("text"), str):
                    res = await run_guardrails(ctx, part["text"], "text/plain", instantiate_guardrails(load_config_bundle(pii_only)), suppress_tripwire=True, raise_guardrail_errors=True)
                    part["text"] = get_guardrail_safe_text(res, part["text"])
    except Exception:
        pass

async def scrub_workflow_input(workflow, input_key, config):
    try:
        guardrails = (config or {}).get("guardrails") or []
        pii = next((g for g in guardrails if (g or {}).get("name") == "Contains PII"), None)
        if not pii:
            return
        if not isinstance(workflow, dict):
            return
        value = workflow.get(input_key)
        if not isinstance(value, str):
            return
        pii_only = {"guardrails": [pii]}
        res = await run_guardrails(ctx, value, "text/plain", instantiate_guardrails(load_config_bundle(pii_only)), suppress_tripwire=True, raise_guardrail_errors=True)
        workflow[input_key] = get_guardrail_safe_text(res, value)
    except Exception:
        pass

async def run_and_apply_guardrails(input_text, config, history, workflow):
    results = await run_guardrails(ctx, input_text, "text/plain", instantiate_guardrails(load_config_bundle(config)), suppress_tripwire=True, raise_guardrail_errors=True)
    guardrails = (config or {}).get("guardrails") or []
    mask_pii = next((g for g in guardrails if (g or {}).get("name") == "Contains PII" and ((g or {}).get("config") or {}).get("block") is False), None) is not None
    if mask_pii:
        await scrub_conversation_history(history, config)
        await scrub_workflow_input(workflow, "input_as_text", config)
        await scrub_workflow_input(workflow, "input_text", config)
    has_tripwire = guardrails_has_tripwire(results)
    safe_text = get_guardrail_safe_text(results, input_text)
    fail_output = build_guardrail_fail_output(results or [])
    pass_output = {"safe_text": (get_guardrail_safe_text(results, input_text) or input_text)}
    return {"results": results, "has_tripwire": has_tripwire, "safe_text": safe_text, "fail_output": fail_output, "pass_output": pass_output}

def build_guardrail_fail_output(results):
    def _get(name: str):
        for r in (results or []):
            info = (r.info if hasattr(r, "info") else None) or {}
            gname = (info.get("guardrail_name") if isinstance(info, dict) else None) or (info.get("guardrailName") if isinstance(info, dict) else None)
            if gname == name:
                return r
        return None
    pii, mod, jb, hal, nsfw, url, custom, pid = map(_get, ["Contains PII", "Moderation", "Jailbreak", "Hallucination Detection", "NSFW Text", "URL Filter", "Custom Prompt Check", "Prompt Injection Detection"])
    def _tripwire(r):
        return bool(r.tripwire_triggered)
    def _info(r):
        return r.info
    jb_info, hal_info, nsfw_info, url_info, custom_info, pid_info, mod_info, pii_info = map(_info, [jb, hal, nsfw, url, custom, pid, mod, pii])
    detected_entities = pii_info.get("detected_entities") if isinstance(pii_info, dict) else {}
    pii_counts = []
    if isinstance(detected_entities, dict):
        for k, v in detected_entities.items():
            if isinstance(v, list):
                pii_counts.append(f"{k}:{len(v)}")
    flagged_categories = (mod_info.get("flagged_categories") if isinstance(mod_info, dict) else None) or []
    
    return {
        "pii": { "failed": (len(pii_counts) > 0) or _tripwire(pii), "detected_counts": pii_counts },
        "moderation": { "failed": _tripwire(mod) or (len(flagged_categories) > 0), "flagged_categories": flagged_categories },
        "jailbreak": { "failed": _tripwire(jb) },
        "hallucination": { "failed": _tripwire(hal), "reasoning": (hal_info.get("reasoning") if isinstance(hal_info, dict) else None), "hallucination_type": (hal_info.get("hallucination_type") if isinstance(hal_info, dict) else None), "hallucinated_statements": (hal_info.get("hallucinated_statements") if isinstance(hal_info, dict) else None), "verified_statements": (hal_info.get("verified_statements") if isinstance(hal_info, dict) else None) },
        "nsfw": { "failed": _tripwire(nsfw) },
        "url_filter": { "failed": _tripwire(url) },
        "custom_prompt_check": { "failed": _tripwire(custom) },
        "prompt_injection": { "failed": _tripwire(pid) },
    }
class ClassificationAgentSchema(BaseModel):
  classification: str


classification_agent = Agent(
  name="Classification agent",
  instructions="""Classify the user’s intent into one of the following categories: \"return_item\" or \"get_information\". 

1. Any device-related return requests should route to return_item.
3. Any other requests should go to get_information.""",
  model="gpt-4.1-mini",
  output_type=ClassificationAgentSchema,
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


return_agent = Agent(
  name="Return agent",
  instructions="""Offer a replacement device with free shipping.
""",
  model="gpt-4.1-mini",
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


retention_agent = Agent(
  name="Retention Agent",
  instructions="You are a customer retention conversational agent whose goal is to prevent subscription cancellations. Ask for their current plan and reason for dissatisfaction. Use the get_retention_offers to identify return options. For now, just say there is a 20% offer available for 1 year.",
  model="gpt-4.1-mini",
  tools=[
    get_retention_offers
  ],
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    parallel_tool_calls=True,
    max_tokens=2048,
    store=True
  )
)


information_agent = Agent(
  name="Information agent",
  instructions="""You are an information agent for answering informational queries. Your aim is to provide clear, concise responses to user questions. Use the policy below to assemble your answer.

Company Name: Milieu Insights Region: South East Asia
Milieu Support Chatbot – Master Instruction Set
General Rules
Always answer using Milieu’s policies as defined below.
Keep answers clear, friendly, and concise, but always include the required steps and conditions.
When giving instructions that involve the Milieu app, reference paths like:
Profile → Account
Profile → More
Profile → Ledger
The agent must never invent policies. Only use rules listed in this document.
If a user asks something outside these FAQs, instruct them to contact Milieu Support.
1. ACCOUNT MANAGEMENT
1.1 Account Verification
Verification method depends on sign-up method:
Facebook/Apple ID: No extra email; verification happens through the platform.
Email signup: A verification email is sent to the user; they must open it and click the link.
Advise users to check spam/junk folders.
Support cannot manually activate accounts.
Verified email must remain active for receiving rewards.
1.2 Password Reset & Changes
If user signed up with email:
To reset password: use “I Forgot” on login page and follow the email link.
To change password while logged in: go to Profile → Account.
If user signed up with Facebook/Apple ID:
They do not have a Milieu password; password changes occur on the external platform.
1.3 Updating Personal Details
Editable: first name, last name, language, password (email login only).
Non-editable by user: birthdate and gender (critical for survey matching and verification).
If these are incorrect/missing, instruct user to contact support.
Inform user that birthdate/gender changes are normally allowed only once.
1.4 Changing the Email Address
Users cannot change emails themselves.
Instruct them to contact support with:
Current email
New email
Changes are subject to approval; policy allows one account per person/device.
1.5 Suspended Accounts
State possible reasons:
Multiple accounts
Identity misuse
Repeated attention-check failures
Low-quality responses
Terms of Use violations
Users may receive a suspension email.
If they dispute suspension, direct them to contact support.
1.6 Expired Accounts
Accounts expire after 12 months of inactivity.
Effects of inactivity:
Badge resets to Explorer
Boost resets
All points expire
Account may deactivate
To avoid expiration: regularly do Surveys/Hot Topics/Quizzes.
If already expired, user may appeal for reactivation.
1.7 Issues With Apple “Hide My Email”
Explain that Apple may generate a relay email.
Verification and reward emails are sent to that relay, then forwarded to their private inbox.
This is expected behavior.
1.8 Account Deletion
Path: Profile → Other → Delete my account.
Deletion is permanent.
All points, boosts, rewards, and badge levels are lost.
Encourage contacting support if they suspect an issue before deleting.
2. ACTIVITIES
2.1 Hot Topics & Quizzes
Hot Topics: opinion polls with instant results.
Quizzes: knowledge checks; users can view info/facts after completion.
Both award points.
2.2 Boost & Streak Rules
Badge Boost levels:
Explorer: 0%
Bronze: +3%
Silver: +5%
Gold: +10%
Platinum: +15%
Streak Boost:
Complete 2 activities in 7 days to activate.
Maintain by doing 2 activities every 7 days.
Boosts apply to survey/quiz/hot topic points only.
2.3 Why Two Different Point Figures
Dark figure = Lifetime points
Never decreases
Used for badge progression
Light figure = Available points
Spent on rewards/donations
Changes with redemptions
Includes special/campaign bonus points
2.4 Ledger & History
Path: Profile → Ledger
Show:
Activity name & date
Base points + boost
Reward claims & refunds
2.5 Attention Check Questions
Designed to test attentiveness.
Multiple recent failures can cause suspension.
Warn users that careful reading is required.
2.6 Activity Errors
If user reports errors (no options, broken media, scroll issue, etc.):
Request:
Phone model
OS version
App version (Profile → More)
Name of survey
Date
Screenshots/recordings
Advise that support can investigate.
2.7 Changing Submitted Responses
Users cannot edit survey answers after submission.
Encourage reading questions carefully.
Only attention checks have “correct” answers.
2.8 Not Receiving Surveys
Possible causes:
Missing or incorrect birthdate/gender
Natural drop after intro surveys
Limited survey quotas
Notifications clicked too late
Recommendations:
Log in regularly
Keep app updated
Check birthdate/gender accuracy
3. DONATIONS
3.1 How Donations Are Processed
All donations for a month are processed at the start of next month.
Donor receives a confirmation email.
Partners receive the contributor list.
For donation-specific issues, contact the charity partner.
3.2 Donation Points Not Deducted
Processing time: up to 1 working day.
If user redeems another reward before donation completes and balance becomes insufficient, donation fails.
Confirmation email indicates approval/rejection.
Users can check status in Ledger.
4. TECHNICAL TROUBLESHOOTING
4.1 App Crashing
Recommend uninstall → reinstall.
Points & status remain safe.
For Android:
Use latest version
Older OS versions may have issues
Clear cache/data before reinstall
Request device/app info if issue continues.
4.2 App Not Downloading
Device may not meet compatibility requirements.
Explain that Milieu is improving support for more devices over time.
5. REWARDS & REFERRALS
5.1 Reward Redemption
Steps:
Tap Ticket icon
Select a reward
Tap Claim
Fill accurate details (email + phone)
Check confirmation email (spam/junk included)
Rewards cannot be refunded or modified by partners easily.
5.2 Reward Processing Time
Normally 10 working days.
Reward partner sends email/SMS.
If nothing is received after 10 working days, user must provide:
Reward reference number
Reward type
Redemption date
Mobile number
Rewards cannot be cancelled or changed once processed.
5.3 Reward Activation Rules
For vouchers/e-wallet:
User receives partner email with link
Must activate via link
Merchant contact required if code fails
For prepaid credit:
SMS confirms credit added
Telecom operator handles missing balance issues
5.4 Refunded Reward Points
Occurs when partner cannot deliver due to invalid email/phone.
Points return to balance.
User may submit a new claim.
Persistent issues should go to support.
5.5 Changing Reward Details
Before submission: check correctness.
After submission:
User must contact support within 48 hours.
Changes depend on partner approval.
Once delivered, rewards cannot be reversed.
5.6 Expired Rewards
Must be activated before expiry date.
Once expired:
Reward becomes void
Points are not refunded
No replacement possible
6. REFER A FRIEND
6.1 Referral Rules
Referral code is case-sensitive.
Friend must enter code at registration (cannot be added later).
Friend must complete 7 surveys.
Both users receive 500 points automatically upon completion.
Misuse of referral system may lead to suspension.
Referral rewards stop once limit is reached.
6.2 Viewing Referral Details
Path: Profile → More
Shows:
Referral code
Copy/share options
Referral count
Referral cap
Successful when:
Friend signs up with code
Friend completes 7 surveys""",
  model="gpt-4.1-mini",
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


def approval_request(message: str):
  # TODO: Implement
  return True

class WorkflowInput(BaseModel):
  input_as_text: str


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
  with trace("Milieu Agent"):
    state = {

    }
    workflow = workflow_input.model_dump()
    conversation_history: list[TResponseInputItem] = [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": workflow["input_as_text"]
          }
        ]
      }
    ]
    guardrails_input_text = workflow["input_as_text"]
    guardrails_result = await run_and_apply_guardrails(guardrails_input_text, jailbreak_guardrail_config, conversation_history, workflow)
    guardrails_hastripwire = guardrails_result["has_tripwire"]
    guardrails_anonymizedtext = guardrails_result["safe_text"]
    guardrails_output = (guardrails_hastripwire and guardrails_result["fail_output"]) or guardrails_result["pass_output"]
    if guardrails_hastripwire:
      return guardrails_output
    else:
      classification_agent_result_temp = await Runner.run(
        classification_agent,
        input=[
          *conversation_history
        ],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_694a718c9964819089160a7912c26ee40d01ca396fad04f0"
        })
      )

      conversation_history.extend([item.to_input_item() for item in classification_agent_result_temp.new_items])

      classification_agent_result = {
        "output_text": classification_agent_result_temp.final_output.json(),
        "output_parsed": classification_agent_result_temp.final_output.model_dump()
      }
      if classification_agent_result["output_parsed"]["classification"] == "return_item":
        return_agent_result_temp = await Runner.run(
          return_agent,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_694a718c9964819089160a7912c26ee40d01ca396fad04f0"
          })
        )

        conversation_history.extend([item.to_input_item() for item in return_agent_result_temp.new_items])

        return_agent_result = {
          "output_text": return_agent_result_temp.final_output_as(str)
        }
        approval_message = "Does this work for you?"

        if approval_request(approval_message):
            end_result = {
              "message": "Your return is on the way."
            }
            return end_result
        else:
            end_result = {
              "message": "What else can I help you with?"
            }
            return end_result
      elif classification_agent_result["output_parsed"]["classification"] == "get_information":
        information_agent_result_temp = await Runner.run(
          information_agent,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_694a718c9964819089160a7912c26ee40d01ca396fad04f0"
          })
        )

        conversation_history.extend([item.to_input_item() for item in information_agent_result_temp.new_items])

        information_agent_result = {
          "output_text": information_agent_result_temp.final_output_as(str)
        }
      else:
        return classification_agent_result
