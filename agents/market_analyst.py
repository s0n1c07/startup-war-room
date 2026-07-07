from google.adk import Agent
from tools.search_tool import web_search
from tools.rate_limiter import rate_limit_callback
from tools.guardrails import input_guardrail

market_analyst = Agent(
    name="market_analyst",
    model="gemini-3.1-flash-lite",
    instruction=(
        "You are a sharp market research analyst. Given a business idea, "
        "use web_search to find: (1) the approximate market size, "
        "(2) 2-3 real or likely competitors, and (3) one recent relevant "
        "trend. Write a tight, factual summary in under 150 words. Do not "
        "give an opinion on whether the idea is good -- just report facts."
    ),
    tools=[web_search],
    output_key="market_analysis",
    before_model_callback=[rate_limit_callback, input_guardrail],
)