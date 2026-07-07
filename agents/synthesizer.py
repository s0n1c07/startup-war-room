from google.adk import Agent
from tools.rate_limiter import rate_limit_callback
from tools.guardrails import output_guardrail, skip_if_blocked

synthesizer = Agent(
    name="synthesizer",
    model="gemini-3.1-flash-lite",
    instruction=(
        "You are a startup pitch coach. You've watched a debate unfold:\n\n"
        "Market research: {market_analysis}\n\n"
        "Skeptic's critique: {critique}\n\n"
        "Financial/success estimate: {financial_estimate}\n\n"
        "Your previous pitch draft (if any): {final_pitch?}\n\n"
        "Revise the pitch (under 200 words) to directly address the "
        "skeptic's latest critique, building on your previous draft rather "
        "than starting over. Incorporate the success-probability reality "
        "check, and end with one concrete next step the founder should take "
        "this week."
    ),
    output_key="final_pitch",
    before_model_callback=rate_limit_callback,
    after_model_callback=output_guardrail, 
    before_agent_callback=skip_if_blocked, # ensure the pitch is under 200 words
)