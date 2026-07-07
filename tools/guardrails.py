# tools/guardrails.py
from google.adk.models.llm_response import LlmResponse
from google.genai import types

BLOCKED_KEYWORDS = [
    "weapon", "drug", "scam", "fraud", "illegal", "hack", "exploit",
]

def input_guardrail(callback_context, llm_request):
    user_text = ""
    for content in llm_request.contents:
        if content.role == "user":
            for part in content.parts:
                if part.text:
                    user_text += part.text.lower()

    for word in BLOCKED_KEYWORDS:
        if word in user_text:
            # Mark this in session state so every later agent can check it
            callback_context.state["blocked"] = True
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=(
                        "I can't run this pitch through the war room -- "
                        "it appears to involve something outside legitimate "
                        "business territory. Try a different idea."
                    ))]
                )
            )
    return None


UNREALISTIC_PHRASES = [
    "guaranteed", "risk-free", "can't fail", "100% success",
    "no competition", "guaranteed return",
]

def output_guardrail(callback_context, llm_response):
    """after_model_callback attached to the synthesizer -- flags
    unrealistic claims instead of silently letting them through."""
    if not llm_response.content or not llm_response.content.parts:
        return None

    text = "".join(p.text or "" for p in llm_response.content.parts)
    lowered = text.lower()

    for phrase in UNREALISTIC_PHRASES:
        if phrase in lowered:
            warning = (
                "\n\n[GUARDRAIL NOTE: This pitch used the unrealistic claim "
                f"'{phrase}' -- no startup is risk-free or guaranteed. "
                "Treat this pitch's confidence level with skepticism.]"
            )
            llm_response.content.parts[0].text = text + warning
            return llm_response

    return None

# same file, tools/guardrails.py
from google.adk.events import EventActions
from google.adk.events.event import Event

def skip_if_blocked(callback_context):
    """before_agent_callback -- short-circuits this agent entirely if an
    earlier guardrail already blocked the pitch."""
    if callback_context.state.get("blocked"):
        return types.Content(
            role="model",
            parts=[types.Part(text="(skipped -- pitch was blocked earlier)")]
        )
    return None