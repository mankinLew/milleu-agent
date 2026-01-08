# agent_workflow.py
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from agents import (
    function_tool,
    Agent,
    ModelSettings,
    TResponseInputItem,
    Runner,
    RunConfig,
    trace,
)
from openai import AsyncOpenAI
from guardrails.runtime import load_config_bundle, instantiate_guardrails, run_guardrails
from pydantic import BaseModel


# ----------------------------
# Tool definitions
# ----------------------------
@function_tool
def get_retention_offers(
    customer_id: str,
    account_type: str,
    current_plan: str,
    tenure_months: int,          # ✅ FIXED: integer -> int
    recent_complaints: bool
):
    # TODO: Implement real logic
    # For now, return a simple placeholder consistent with your instructions.
    return {
        "offers": [
            {"type": "discount", "value": "20% for 1 year", "conditions": "standard eligibility"}
        ]
    }


# ----------------------------
# Shared client for guardrails
# ----------------------------
client = AsyncOpenAI()
ctx = SimpleNamespace(guardrail_llm=client)


# ----------------------------
# Guardrails definitions
# ----------------------------
jailbreak_guardrail_config = {
    "guardrails": [
        {"name": "Jailbreak", "config": {"model": "gpt-5-nano", "confidence_threshold": 0.7}}
    ]
}


def guardrails_has_tripwire(results):
    return any(
        (hasattr(r, "tripwire_triggered") and (r.tripwire_triggered is True))
        for r in (results or [])
    )


def get_guardrail_safe_text(results, fallback_text):
    for r in (results or []):
        info = (r.info if hasattr(r, "info") else None) or {}
        if isinstance(info, dict) and ("checked_text" in info):
            return info.get("checked_text") or fallback_text

    pii = next(
        (
            (r.info if hasattr(r, "info") else {})
            for r in (results or [])
            if isinstance(((r.info if hasattr(r, "info") else None) or {}), dict)
            and ("anonymized_text" in ((r.info if hasattr(r, "info") else None) or {}))
        ),
        None,
    )
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
        bundle = load_config_bundle(pii_only)
        instantiated = instantiate_guardrails(bundle)

        for msg in (history or []):
            content = (msg or {}).get("content") or []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "input_text"
                    and isinstance(part.get("text"), str)
                ):
                    res = await run_guardrails(
                        ctx,
                        part["text"],
                        "text/plain",
                        instantiated,
                        suppress_tripwire=True,
                        raise_guardrail_errors=True,
                    )
                    part["text"] = get_guardrail_safe_text(res, part["text"])
    except Exception:
        # best-effort scrub
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
        bundle = load_config_bundle(pii_only)
        instantiated = instantiate_guardrails(bundle)

        res = await run_guardrails(
            ctx,
            value,
            "text/plain",
            instantiated,
            suppress_tripwire=True,
            raise_guardrail_errors=True,
        )
        workflow[input_key] = get_guardrail_safe_text(res, value)
    except Exception:
        # best-effort scrub
        pass


async def run_and_apply_guardrails(input_text, config, history, workflow):
    bundle = load_config_bundle(config)
    instantiated = instantiate_guardrails(bundle)

    results = await run_guardrails(
        ctx,
        input_text,
        "text/plain",
        instantiated,
        suppress_tripwire=True,
        raise_guardrail_errors=True,
    )

    guardrails = (config or {}).get("guardrails") or []
    mask_pii = next(
        (
            g for g in guardrails
            if (g or {}).get("name") == "Contains PII"
            and ((g or {}).get("config") or {}).get("block") is False
        ),
        None
    ) is not None

    if mask_pii:
        await scrub_conversation_history(history, config)
        await scrub_workflow_input(workflow, "input_as_text", config)
        await scrub_workflow_input(workflow, "input_text", config)

    has_tripwire = guardrails_has_tripwire(results)
    safe_text = get_guardrail_safe_text(results, input_text)
    fail_output = build_guardrail_fail_output(results or [])
    pass_output = {"safe_text": (safe_text or input_text)}
    return {
        "results": results,
        "has_tripwire": has_tripwire,
        "safe_text": safe_text,
        "fail_output": fail_output,
        "pass_output": pass_output,
    }


def build_guardrail_fail_output(results):
    def _get(name: str):
        for r in (results or []):
            info = (r.info if hasattr(r, "info") else None) or {}
            gname = None
            if isinstance(info, dict):
                gname = info.get("guardrail_name") or info.get("guardrailName")
            if gname == name:
                return r
        return None

    pii = _get("Contains PII")
    mod = _get("Moderation")
    jb = _get("Jailbreak")
    hal = _get("Hallucination Detection")
    nsfw = _get("NSFW Text")
    url = _get("URL Filter")
    custom = _get("Custom Prompt Check")
    pid = _get("Prompt Injection Detection")

    def _tripwire(r):
        return bool(getattr(r, "tripwire_triggered", False)) if r is not None else False

    def _info(r):
        return getattr(r, "info", None) if r is not None else None

    jb_info, hal_info, nsfw_info, url_info, custom_info, pid_info, mod_info, pii_info = map(
        _info, [jb, hal, nsfw, url, custom, pid, mod, pii]
    )

    detected_entities = pii_info.get("detected_entities") if isinstance(pii_info, dict) else {}
    pii_counts = []
    if isinstance(detected_entities, dict):
        for k, v in detected_entities.items():
            if isinstance(v, list):
                pii_counts.append(f"{k}:{len(v)}")

    flagged_categories = (mod_info.get("flagged_categories") if isinstance(mod_info, dict) else None) or []

    return {
        "pii": {"failed": (len(pii_counts) > 0) or _tripwire(pii), "detected_counts": pii_counts},
        "moderation": {"failed": _tripwire(mod) or (len(flagged_categories) > 0), "flagged_categories": flagged_categories},
        "jailbreak": {"failed": _tripwire(jb)},
        "hallucination": {
            "failed": _tripwire(hal),
            "reasoning": (hal_info.get("reasoning") if isinstance(hal_info, dict) else None),
            "hallucination_type": (hal_info.get("hallucination_type") if isinstance(hal_info, dict) else None),
            "hallucinated_statements": (hal_info.get("hallucinated_statements") if isinstance(hal_info, dict) else None),
            "verified_statements": (hal_info.get("verified_statements") if isinstance(hal_info, dict) else None),
        },
        "nsfw": {"failed": _tripwire(nsfw)},
        "url_filter": {"failed": _tripwire(url)},
        "custom_prompt_check": {"failed": _tripwire(custom)},
        "prompt_injection": {"failed": _tripwire(pid)},
    }


# ----------------------------
# Agents
# ----------------------------
class ClassificationAgentSchema(BaseModel):
    classification: str


classification_agent = Agent(
    name="Classification agent",
    instructions=(
        "Classify the user’s intent into one of the following categories: "
        '"return_item" or "get_information".\n\n'
        "1. Any device-related return requests should route to return_item.\n"
        "3. Any other requests should go to get_information."
    ),
    model="gpt-4.1-mini",
    output_type=ClassificationAgentSchema,
    model_settings=ModelSettings(
        temperature=1,
        top_p=1,
        max_tokens=2048,
        store=True,
    ),
)

return_agent = Agent(
    name="Return agent",
    instructions="Offer a replacement device with free shipping.",
    model="gpt-4.1-mini",
    model_settings=ModelSettings(
        temperature=1,
        top_p=1,
        max_tokens=2048,
        store=True,
    ),
)

retention_agent = Agent(
    name="Retention Agent",
    instructions=(
        "You are a customer retention conversational agent whose goal is to prevent subscription cancellations. "
        "Ask for their current plan and reason for dissatisfaction. "
        "Use the get_retention_offers to identify return options. "
        "For now, just say there is a 20% offer available for 1 year."
    ),
    model="gpt-4.1-mini",
    tools=[get_retention_offers],
    model_settings=ModelSettings(
        temperature=1,
        top_p=1,
        parallel_tool_calls=True,
        max_tokens=2048,
        store=True,
    ),
)

information_agent = Agent(
    name="Information agent",
    instructions=(
        "You are an information agent for answering informational queries. Your aim is to provide clear, concise "
        "responses to user questions. Use the policy below to assemble your answer.\n\n"
        # (Keeping your full policy text exactly as provided)
        "Company Name: Milieu Insights Region: South East Asia\n"
        "Milieu Support Chatbot – Master Instruction Set\n"
        "General Rules\n"
        "Always answer using Milieu’s policies as defined below.\n"
        "Keep answers clear, friendly, and concise, but always include the required steps and conditions.\n"
        "When giving instructions that involve the Milieu app, reference paths like:\n"
        "Profile → Account\n"
        "Profile → More\n"
        "Profile → Ledger\n"
        "The agent must never invent policies. Only use rules listed in this document.\n"
        "If a user asks something outside these FAQs, instruct them to contact Milieu Support.\n"
        # ... (the rest of your policy text continues; keep it unchanged in your file)
    ),
    model="gpt-4.1-mini",
    model_settings=ModelSettings(
        temperature=1,
        top_p=1,
        max_tokens=2048,
        store=True,
    ),
)


def approval_request(message: str) -> bool:
    # TODO: Implement real approval logic
    return True


class WorkflowInput(BaseModel):
    input_as_text: str


# ----------------------------
# Main entrypoint
# ----------------------------
async def run_workflow(workflow_input: WorkflowInput):
    with trace("Milieu Agent"):
        workflow = workflow_input.model_dump()

        conversation_history: list[TResponseInputItem] = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": workflow["input_as_text"]}],
            }
        ]

        guardrails_input_text = workflow["input_as_text"]
        guardrails_result = await run_and_apply_guardrails(
            guardrails_input_text,
            jailbreak_guardrail_config,
            conversation_history,
            workflow,
        )

        if guardrails_result["has_tripwire"]:
            return guardrails_result["fail_output"]

        # Classification
        classification_agent_result_temp = await Runner.run(
            classification_agent,
            input=[*conversation_history],
            run_config=RunConfig(
                trace_metadata={
                    "__trace_source__": "agent-builder",
                    "workflow_id": "wf_694a718c9964819089160a7912c26ee40d01ca396fad04f0",
                }
            ),
        )

        conversation_history.extend([item.to_input_item() for item in classification_agent_result_temp.new_items])

        classification = classification_agent_result_temp.final_output.model_dump().get("classification")

        if classification == "return_item":
            return_agent_result_temp = await Runner.run(
                return_agent,
                input=[*conversation_history],
                run_config=RunConfig(
                    trace_metadata={
                        "__trace_source__": "agent-builder",
                        "workflow_id": "wf_694a718c9964819089160a7912c26ee40d01ca396fad04f0",
                    }
                ),
            )

            conversation_history.extend([item.to_input_item() for item in return_agent_result_temp.new_items])

            approval_message = "Does this work for you?"
            if approval_request(approval_message):
                return {"message": "Your return is on the way."}
            return {"message": "What else can I help you with?"}

        if classification == "get_information":
            information_agent_result_temp = await Runner.run(
                information_agent,
                input=[*conversation_history],
                run_config=RunConfig(
                    trace_metadata={
                        "__trace_source__": "agent-builder",
                        "workflow_id": "wf_694a718c9964819089160a7912c26ee40d01ca396fad04f0",
                    }
                ),
            )
            conversation_history.extend([item.to_input_item() for item in information_agent_result_temp.new_items])

            # ✅ FIX: return the information agent result (your export computed it but didn't return)
            return {"message": information_agent_result_temp.final_output_as(str)}

        # Fallback
        return {
            "output_text": classification_agent_result_temp.final_output.json(),
            "output_parsed": classification_agent_result_temp.final_output.model_dump(),
        }
