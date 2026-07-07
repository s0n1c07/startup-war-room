# tools/rate_limiter.py
import asyncio
import time

_last_call_time = 0.0
_lock = asyncio.Lock()
MIN_INTERVAL_SECONDS = 5  # 60s / 15 requests = 4s, +1s safety margin

async def rate_limit_callback(callback_context, llm_request):
    global _last_call_time
    async with _lock:
        now = time.monotonic()
        wait = MIN_INTERVAL_SECONDS - (now - _last_call_time)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_time = time.monotonic()
    return None  # returning None lets the LLM call proceed normally