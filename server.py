"""
Startup War Room -- web backend.

Run: uvicorn server:app --reload
Then open http://127.0.0.1:8000
"""

import asyncio
import json
import os
import warnings
import logging
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google.adk").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found -- check your .env file")
os.environ["GOOGLE_API_KEY"] = api_key

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.cloud import firestore
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from agents.orchestrator import root_agent

app = FastAPI()
APP_NAME = "startup_war_room"

FIRESTORE_TIMEOUT_SECONDS = 10
_db_executor = ThreadPoolExecutor(max_workers=4)

# On Render (or any non-local host), authenticate via a service account
# key stored in an env var, since there's no local gcloud login there.
# Locally, this falls back to `gcloud auth application-default login`.
from google.oauth2 import service_account

_sa_json = os.getenv("FIRESTORE_SERVICE_ACCOUNT_JSON")
if _sa_json:
    _creds = service_account.Credentials.from_service_account_info(
        json.loads(_sa_json)
    )
    db = firestore.Client(
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        database=os.getenv("FIRESTORE_DATABASE_ID", "warroomdb"),
        credentials=_creds,
    )
else:
    db = firestore.Client(
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        database=os.getenv("FIRESTORE_DATABASE_ID", "warroomdb"),
    )

PITCHES_COLLECTION = "pitches"
MESSAGES_SUBCOLLECTION = "messages"


def _save_pitch_blocking(idea: str) -> str:
    doc_ref = db.collection(PITCHES_COLLECTION).document()
    doc_ref.set(
        {"idea": idea, "created_at": datetime.now(timezone.utc).isoformat()}
    )
    return doc_ref.id


def save_pitch(idea: str) -> str | None:
    """Saves a pitch to Firestore with a hard timeout. Returns None (and
    logs a warning) if Firestore hangs or fails, instead of blocking the
    whole pipeline forever -- persistence failing shouldn't stop the demo
    from running."""
    future = _db_executor.submit(_save_pitch_blocking, idea)
    try:
        return future.result(timeout=FIRESTORE_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        print(f"[firestore] save_pitch TIMED OUT after {FIRESTORE_TIMEOUT_SECONDS}s", flush=True)
        return None
    except Exception as e:
        print(f"[firestore] save_pitch ERROR: {e}", flush=True)
        return None


def _save_message_blocking(pitch_id: str, author: str, event_type: str, text: str, seq: int):
    db.collection(PITCHES_COLLECTION).document(pitch_id).collection(
        MESSAGES_SUBCOLLECTION
    ).document().set(
        {
            "author": author,
            "event_type": event_type,
            "text": text,
            "seq": seq,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def save_message(pitch_id: str | None, author: str, event_type: str, text: str, seq: int):
    if pitch_id is None:
        return  # earlier save_pitch failed -- skip persistence, keep the demo running
    future = _db_executor.submit(
        _save_message_blocking, pitch_id, author, event_type, text, seq
    )
    try:
        future.result(timeout=FIRESTORE_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        print(f"[firestore] save_message TIMED OUT after {FIRESTORE_TIMEOUT_SECONDS}s", flush=True)
    except Exception as e:
        print(f"[firestore] save_message ERROR: {e}", flush=True)


@app.get("/api/pitches")
async def list_pitches():
    docs = (
        db.collection(PITCHES_COLLECTION)
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(50)
        .stream()
    )
    return [{"id": d.id, **d.to_dict()} for d in docs]


@app.get("/api/pitches/{pitch_id}")
async def get_pitch(pitch_id: str):
    doc_ref = db.collection(PITCHES_COLLECTION).document(pitch_id)
    doc = doc_ref.get()
    if not doc.exists:
        return {"error": "not found"}

    messages = (
        doc_ref.collection(MESSAGES_SUBCOLLECTION).order_by("seq").stream()
    )
    return {
        "pitch": {"id": doc.id, **doc.to_dict()},
        "messages": [m.to_dict() for m in messages],
    }


@app.delete("/api/pitches/{pitch_id}")
async def delete_pitch(pitch_id: str):
    doc_ref = db.collection(PITCHES_COLLECTION).document(pitch_id)
    # Firestore doesn't cascade-delete subcollections -- remove messages first
    for m in doc_ref.collection(MESSAGES_SUBCOLLECTION).stream():
        m.reference.delete()
    doc_ref.delete()
    return {"deleted": pitch_id}


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.websocket("/ws/pitch")
async def pitch_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            idea = data.get("idea", "").strip()
            if not idea:
                continue

            print(f"[ws] received pitch text: {idea[:60]}", flush=True)

            session_service = InMemorySessionService()
            session = await session_service.create_session(
                app_name=APP_NAME, user_id="web_user"
            )
            runner = Runner(
                agent=root_agent, app_name=APP_NAME, session_service=session_service
            )
            content = types.Content(role="user", parts=[types.Part(text=idea)])

            print(f"[ws] about to save_pitch...", flush=True)
            pitch_id = save_pitch(idea)
            print(f"[pitch {pitch_id}] started: {idea[:60]}", flush=True)
            await websocket.send_json({"type": "start", "idea": idea, "pitch_id": pitch_id})

            already_blocked = False
            seq = 0
            retries = 3
            for attempt in range(retries):
                try:
                    async for event in runner.run_async(
                        user_id="web_user", session_id=session.id, new_message=content
                    ):
                        author = event.author or "agent"
                        if event.content and event.content.parts:
                            for part in event.content.parts:
                                fn_call = getattr(part, "function_call", None)
                                if fn_call:
                                    print(f"[pitch {pitch_id}] {author} -> tool_call: {fn_call.name}", flush=True)
                                    save_message(pitch_id, author, "tool_call", fn_call.name, seq)
                                    seq += 1
                                    await websocket.send_json(
                                        {
                                            "type": "tool_call",
                                            "author": author,
                                            "tool": fn_call.name,
                                        }
                                    )
                                if part.text:
                                    text = part.text.strip()
                                    if not text:
                                        continue
                                    if "(skipped -- pitch was blocked earlier)" in text:
                                        if already_blocked:
                                            continue
                                        already_blocked = True
                                    print(f"[pitch {pitch_id}] {author} -> speech ({len(text)} chars)", flush=True)
                                    save_message(pitch_id, author, "speech", text, seq)
                                    seq += 1
                                    await websocket.send_json(
                                        {
                                            "type": "speech",
                                            "author": author,
                                            "text": text,
                                        }
                                    )
                    break
                except Exception as e:
                    print(f"[pitch {pitch_id}] EXCEPTION: {e}", flush=True)
                    if (
                        "UNAVAILABLE" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                    ) and attempt < retries - 1:
                        await websocket.send_json(
                            {"type": "info", "text": "Model busy, retrying..."}
                        )
                        await asyncio.sleep(5)
                    else:
                        await websocket.send_json({"type": "error", "text": str(e)})
                        break

            print(f"[pitch {pitch_id}] done", flush=True)
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass