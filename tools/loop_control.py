# tools/loop_control.py
from google.adk.tools import ToolContext

def exit_loop(tool_context: ToolContext) -> dict:
    """Call this when you have no further meaningful objections to the
    pitch — signals the debate is complete and ends the loop early.
    """
    tool_context.actions.escalate = True
    return {"status": "debate concluded"}