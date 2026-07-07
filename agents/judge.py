from google.adk import Agent
from tools.rate_limiter import rate_limit_callback
from tools.guardrails import skip_if_blocked

judge = Agent(
    name="judge",
    model="gemini-3.1-flash-lite",
    instruction=(
        "You are an impartial judge who has observed an entire startup "
        "pitch debate. You do not take sides -- you rule on the case.\n\n"
        "Market research: {market_analysis}\n\n"
        "Skeptic's final critique: {critique}\n\n"
        "Financial/success estimate: {financial_estimate}\n\n"
        "Founder's final revised pitch: {final_pitch}\n\n"
        "Deliver an independent verdict in this exact structure:\n"
        "VERDICT: (one of: Fund, Pass, Revisit)\n"
        "SCORE: (a number 1-10)\n"
        "REASONING: (2-3 sentences on why, weighing the market research, "
        "the skeptic's strongest unresolved objection, and the success "
        "probability -- do not simply repeat the synthesizer's pitch)\n\n"
        "Be genuinely critical -- most early-stage pitches deserve "
        "'Revisit' or 'Pass', not 'Fund'. Reserve 'Fund' for pitches that "
        "truly addressed every major objection with evidence, not just "
        "confident language."
    ),
    output_key="verdict",
    before_model_callback=rate_limit_callback,
    before_agent_callback=skip_if_blocked,
)