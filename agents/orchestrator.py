from google.adk.agents import SequentialAgent , LoopAgent

from agents.market_analyst import market_analyst
from agents.skeptic import skeptic
from agents.financial_modeler import financial_modeler
from agents.synthesizer import synthesizer
from agents.judge import judge

debate_loop = LoopAgent(
    name="debate_loop",
    sub_agents=[skeptic, synthesizer],
    max_iterations=6,  # hard ceiling even if skeptic never exits
)

root_agent = SequentialAgent(
    name="startup_war_room",
    sub_agents=[market_analyst, financial_modeler, debate_loop,judge],
)