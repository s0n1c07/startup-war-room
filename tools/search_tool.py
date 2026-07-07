from ddgs import DDGS

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

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=4))
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}: {r['body']}" for r in results)