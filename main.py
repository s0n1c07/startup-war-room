import sys
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found — check your .env file")
os.environ["GOOGLE_API_KEY"] = api_key

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.orchestrator import root_agent

APP_NAME = "startup_war_room"
USER_ID = "founder_1"


async def run_pipeline(idea: str, retries: int = 3, delay: int = 5):
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=idea)])

    print(f"\n=== Pitching: {idea} ===\n")

    for attempt in range(retries):
        try:
            already_blocked = False

            async for event in runner.run_async(
                user_id=USER_ID, session_id=session.id, new_message=content
            ):
                if event.content and event.content.parts:
                    author = event.author or "agent"
                    text = "".join(p.text or "" for p in event.content.parts)
                    if "(skipped -- pitch was blocked earlier)" in text:
                        if not already_blocked:
                            print(f"--- {author} ---\n{text.strip()}\n")
                            already_blocked = True
                        continue  # suppress repeats, but keep draining the generator
                    if text.strip():
                        print(f"--- {author} ---\n{text.strip()}\n")
            return
        except Exception as e:
            if ("UNAVAILABLE" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < retries - 1:
                print(f"Rate limited or busy, retrying in {delay}s... (attempt {attempt+1}/{retries})")
                await asyncio.sleep(delay)
            else:
                raise


if __name__ == "__main__":
    idea = " ".join(sys.argv[1:]) or "A subscription box for artisanal hot sauce"
    asyncio.run(run_pipeline(idea))