import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from ddgs import DDGS

SEARCH_TIMEOUT_SECONDS = 15
_executor = ThreadPoolExecutor(max_workers=4)


def _do_search(query: str):
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=4))


def web_search(query: str) -> str:
    """Searches the web and returns a few relevant results as text.

    Args:
        query: A single search query string. If you need to research
            multiple things, call this tool multiple times, once per
            individual query -- do not pass a list.

    Returns:
        A short text summary of the top search results, including titles and snippets.
    """
    if isinstance(query, list):
        query = " ".join(str(q) for q in query)

    print(f"[web_search] searching: {query}", flush=True)

    future = _executor.submit(_do_search, query)
    try:
        results = future.result(timeout=SEARCH_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        print(f"[web_search] TIMED OUT after {SEARCH_TIMEOUT_SECONDS}s: {query}", flush=True)
        return (
            "Search timed out -- no results available for this query. "
            "Proceed using general knowledge instead."
        )
    except Exception as e:
        print(f"[web_search] ERROR: {e}", flush=True)
        return f"Search failed ({e}) -- proceed using general knowledge instead."

    if not results:
        print(f"[web_search] no results for: {query}", flush=True)
        return "No results found."

    print(f"[web_search] got {len(results)} results for: {query}", flush=True)
    return "\n\n".join(f"{r['title']}: {r['body']}" for r in results)