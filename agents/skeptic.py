# agents/skeptic.py
from google.adk import Agent
from tools.loop_control import exit_loop
from tools.rate_limiter import rate_limit_callback
from tools.guardrails import skip_if_blocked

skeptic = Agent(
    name="skeptic",
    model="gemini-3.1-flash-lite",
    instruction=(
        "You are a blunt, experienced VC playing devil's advocate.\n\n"
        "Market research: {market_analysis}\n\n"
        "Current pitch (may have been revised already): {final_pitch?}\n\n"
        "If this is the first round (no revised pitch above, or it's "
        "unchanged from a prior critique), find the 1-2 strongest reasons "
        "this could fail — competition, timing, unit economics, execution "
        "risk. Be specific.\n\n"
        "If a revised pitch already addresses your prior critique well, "
        "call exit_loop instead of manufacturing a new objection — don't "
        "nitpick just to have something to say. Only keep critiquing if "
        "there's a genuine, serious gap left.\n\n"
        "Under 120 words if you do critique."
    ),
    tools=[exit_loop],
    output_key="critique",
    before_model_callback=rate_limit_callback,
    before_agent_callback=skip_if_blocked,
)