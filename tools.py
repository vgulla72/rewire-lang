from langchain_core.tools import Tool
import requests
import os

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def serper_search_fn(query: str, num_results: int = 10) -> str:
    """Search for LinkedIn-style career transitions using Serper.dev."""
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"q": query, "num": num_results}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"
    results = response.json().get("organic", [])
    return "\n\n".join(
        [f"Title: {r['title']}\nSnippet: {r['description']}\nURL: {r['link']}" for r in results]
    )

# Define the tool as a Tool instance
serper_search = Tool.from_function(
    name="serper_search",
    func=serper_search_fn,
    description="Use Serper.dev to search for real-world LinkedIn profiles by query string.",
)
