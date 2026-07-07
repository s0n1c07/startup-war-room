from google.adk import Agent
from tools.startup_model_tool import estimate_success_probability
from tools.rate_limiter import rate_limit_callback
from tools.guardrails import skip_if_blocked

financial_modeler = Agent(
    name="financial_modeler",
    model="gemini-3.1-flash-lite",
    instruction=(
        "You are a data-driven startup analyst. Given a business idea and "
        "this market research:\n\n{market_analysis}\n\n"
        "Infer reasonable rough estimates for these 11 factors based on how "
        "established/early-stage the idea sounds: relationships (0-10), "
        "milestones (0-5), is_top500 (0 or 1), age_last_milestone_year "
        "(0-5), has_roundB/A/C/D (0 or 1 each, assume an early idea has "
        "none), funding_rounds (0-5), avg_participants (1-5), "
        "age_first_milestone_year (0-3). Default to values typical of an "
        "early-stage startup if the idea gives no clues. Call "
        "estimate_success_probability with your estimates, then report the "
        "probability and briefly explain which factors mattered most, in "
        "under 100 words."
    ),
    tools=[estimate_success_probability],
    output_key="financial_estimate",
    before_model_callback=rate_limit_callback,
    before_agent_callback=skip_if_blocked,
)