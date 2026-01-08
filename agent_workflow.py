import { tool, Agent, AgentInputItem, Runner, withTrace } from "@openai/agents";
import { z } from "zod";
import { OpenAI } from "openai";
import { runGuardrails } from "@openai/guardrails";


// Tool definitions
const getRetentionOffers = tool({
  name: "getRetentionOffers",
  description: "Retrieve possible retention offers for a customer",
  parameters: z.object({
    customer_id: z.string(),
    account_type: z.string(),
    current_plan: z.string(),
    tenure_months: z.integer(),
    recent_complaints: z.boolean()
  }),
  execute: async (input: {customer_id: string, account_type: string, current_plan: string, tenure_months: integer, recent_complaints: boolean}) => {
    // TODO: Unimplemented
  },
});

// Shared client for guardrails and file search
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Guardrails definitions
const jailbreakGuardrailConfig = {
  guardrails: [
    { name: "Jailbreak", config: { model: "gpt-5-nano", confidence_threshold: 0.7 } }
  ]
};
const context = { guardrailLlm: client };

function guardrailsHasTripwire(results: any[]): boolean {
    return (results ?? []).some((r) => r?.tripwireTriggered === true);
}

function getGuardrailSafeText(results: any[], fallbackText: string): string {
    for (const r of results ?? []) {
        if (r?.info && ("checked_text" in r.info)) {
            return r.info.checked_text ?? fallbackText;
        }
    }
    const pii = (results ?? []).find((r) => r?.info && "anonymized_text" in r.info);
    return pii?.info?.anonymized_text ?? fallbackText;
}

async function scrubConversationHistory(history: any[], piiOnly: any): Promise<void> {
    for (const msg of history ?? []) {
        const content = Array.isArray(msg?.content) ? msg.content : [];
        for (const part of content) {
            if (part && typeof part === "object" && part.type === "input_text" && typeof part.text === "string") {
                const res = await runGuardrails(part.text, piiOnly, context, true);
                part.text = getGuardrailSafeText(res, part.text);
            }
        }
    }
}

async function scrubWorkflowInput(workflow: any, inputKey: string, piiOnly: any): Promise<void> {
    if (!workflow || typeof workflow !== "object") return;
    const value = workflow?.[inputKey];
    if (typeof value !== "string") return;
    const res = await runGuardrails(value, piiOnly, context, true);
    workflow[inputKey] = getGuardrailSafeText(res, value);
}

async function runAndApplyGuardrails(inputText: string, config: any, history: any[], workflow: any) {
    const guardrails = Array.isArray(config?.guardrails) ? config.guardrails : [];
    const results = await runGuardrails(inputText, config, context, true);
    const shouldMaskPII = guardrails.find((g) => (g?.name === "Contains PII") && g?.config && g.config.block === false);
    if (shouldMaskPII) {
        const piiOnly = { guardrails: [shouldMaskPII] };
        await scrubConversationHistory(history, piiOnly);
        await scrubWorkflowInput(workflow, "input_as_text", piiOnly);
        await scrubWorkflowInput(workflow, "input_text", piiOnly);
    }
    const hasTripwire = guardrailsHasTripwire(results);
    const safeText = getGuardrailSafeText(results, inputText) ?? inputText;
    return { results, hasTripwire, safeText, failOutput: buildGuardrailFailOutput(results ?? []), passOutput: { safe_text: safeText } };
}

function buildGuardrailFailOutput(results: any[]) {
    const get = (name: string) => (results ?? []).find((r: any) => ((r?.info?.guardrail_name ?? r?.info?.guardrailName) === name));
    const pii = get("Contains PII"), mod = get("Moderation"), jb = get("Jailbreak"), hal = get("Hallucination Detection"), nsfw = get("NSFW Text"), url = get("URL Filter"), custom = get("Custom Prompt Check"), pid = get("Prompt Injection Detection"), piiCounts = Object.entries(pii?.info?.detected_entities ?? {}).filter(([, v]) => Array.isArray(v)).map(([k, v]) => k + ":" + v.length), conf = jb?.info?.confidence;
    return {
        pii: { failed: (piiCounts.length > 0) || pii?.tripwireTriggered === true, detected_counts: piiCounts },
        moderation: { failed: mod?.tripwireTriggered === true || ((mod?.info?.flagged_categories ?? []).length > 0), flagged_categories: mod?.info?.flagged_categories },
        jailbreak: { failed: jb?.tripwireTriggered === true },
        hallucination: { failed: hal?.tripwireTriggered === true, reasoning: hal?.info?.reasoning, hallucination_type: hal?.info?.hallucination_type, hallucinated_statements: hal?.info?.hallucinated_statements, verified_statements: hal?.info?.verified_statements },
        nsfw: { failed: nsfw?.tripwireTriggered === true },
        url_filter: { failed: url?.tripwireTriggered === true },
        custom_prompt_check: { failed: custom?.tripwireTriggered === true },
        prompt_injection: { failed: pid?.tripwireTriggered === true },
    };
}
const ClassificationAgentSchema = z.object({ classification: z.enum(["return_item", "cancel_subscription", "get_information"]) });
const classificationAgent = new Agent({
  name: "Classification agent",
  instructions: `Classify the user’s intent into one of the following categories: \"return_item\" or \"get_information\". 

1. Any device-related return requests should route to return_item.
3. Any other requests should go to get_information.`,
  model: "gpt-4.1-mini",
  outputType: ClassificationAgentSchema,
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const returnAgent = new Agent({
  name: "Return agent",
  instructions: `Offer a replacement device with free shipping.
`,
  model: "gpt-4.1-mini",
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const retentionAgent = new Agent({
  name: "Retention Agent",
  instructions: "You are a customer retention conversational agent whose goal is to prevent subscription cancellations. Ask for their current plan and reason for dissatisfaction. Use the get_retention_offers to identify return options. For now, just say there is a 20% offer available for 1 year.",
  model: "gpt-4.1-mini",
  tools: [
    getRetentionOffers
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    parallelToolCalls: true,
    maxTokens: 2048,
    store: true
  }
});

const informationAgent = new Agent({
  name: "Information agent",
  instructions: `You are an information agent for answering informational queries. Your aim is to provide clear, concise responses to user questions. Use the policy below to assemble your answer.

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
Friend completes 7 surveys`,
  model: "gpt-4.1-mini",
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const approvalRequest = (message: string) => {

  // TODO: Implement
  return true;
}

type WorkflowInput = { input_as_text: string };


// Main code entrypoint
export const runWorkflow = async (workflow: WorkflowInput) => {
  return await withTrace("Milieu Agent", async () => {
    const state = {

    };
    const conversationHistory: AgentInputItem[] = [
      { role: "user", content: [{ type: "input_text", text: workflow.input_as_text }] }
    ];
    const runner = new Runner({
      traceMetadata: {
        __trace_source__: "agent-builder",
        workflow_id: "wf_694a718c9964819089160a7912c26ee40d01ca396fad04f0"
      }
    });
    const guardrailsInputText = workflow.input_as_text;
    const { hasTripwire: guardrailsHasTripwire, safeText: guardrailsAnonymizedText, failOutput: guardrailsFailOutput, passOutput: guardrailsPassOutput } = await runAndApplyGuardrails(guardrailsInputText, jailbreakGuardrailConfig, conversationHistory, workflow);
    const guardrailsOutput = (guardrailsHasTripwire ? guardrailsFailOutput : guardrailsPassOutput);
    if (guardrailsHasTripwire) {
      return guardrailsOutput;
    } else {
      const classificationAgentResultTemp = await runner.run(
        classificationAgent,
        [
          ...conversationHistory
        ]
      );
      conversationHistory.push(...classificationAgentResultTemp.newItems.map((item) => item.rawItem));

      if (!classificationAgentResultTemp.finalOutput) {
          throw new Error("Agent result is undefined");
      }

      const classificationAgentResult = {
        output_text: JSON.stringify(classificationAgentResultTemp.finalOutput),
        output_parsed: classificationAgentResultTemp.finalOutput
      };
      if (classificationAgentResult.output_parsed.classification == "return_item") {
        const returnAgentResultTemp = await runner.run(
          returnAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...returnAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!returnAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const returnAgentResult = {
          output_text: returnAgentResultTemp.finalOutput ?? ""
        };
        const approvalMessage = "Does this work for you?";

        if (approvalRequest(approvalMessage)) {
            const endResult = {
              message: "Your return is on the way."
            };
            return endResult;
        } else {
            const endResult = {
              message: "What else can I help you with?"
            };
            return endResult;
        }
      } else if (classificationAgentResult.output_parsed.classification == "get_information") {
        const informationAgentResultTemp = await runner.run(
          informationAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...informationAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!informationAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const informationAgentResult = {
          output_text: informationAgentResultTemp.finalOutput ?? ""
        };
      } else {
        return classificationAgentResult;
      }
    }
  });
}
