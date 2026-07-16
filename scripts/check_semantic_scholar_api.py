import os
import asyncio
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()

SNIPPET_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"
PAPER_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


async def test_snippet_search():
    print("\n=== Test 1: Semantic Scholar Snippet Search ===")

    headers = {
        "User-Agent": "RAGSearchBot/1.0"
    }

    if API_KEY:
        headers["x-api-key"] = API_KEY

    params = {
        "query": "Monte Carlo tree search",
        "limit": 5,
        "fields": "snippet.text,snippet.snippetKind,snippet.section",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(SNIPPET_URL, params=params, headers=headers)

    print("Status:", resp.status_code)
    print("Headers:", {
        "retry-after": resp.headers.get("retry-after"),
        "x-ratelimit-limit": resp.headers.get("x-ratelimit-limit"),
        "x-ratelimit-remaining": resp.headers.get("x-ratelimit-remaining"),
    })

    try:
        data = resp.json()
        print("Response JSON:")
        print(data)
    except Exception:
        print("Raw response:")
        print(resp.text)


async def test_paper_search():
    print("\n=== Test 2: Semantic Scholar Paper Search ===")

    headers = {
        "User-Agent": "RAGSearchBot/1.0"
    }

    if API_KEY:
        headers["x-api-key"] = API_KEY

    params = {
        "query": "Monte Carlo tree search",
        "limit": 5,
        "fields": "title,year,authors,venue,citationCount,abstract",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(PAPER_SEARCH_URL, params=params, headers=headers)

    print("Status:", resp.status_code)
    print("Headers:", {
        "retry-after": resp.headers.get("retry-after"),
        "x-ratelimit-limit": resp.headers.get("x-ratelimit-limit"),
        "x-ratelimit-remaining": resp.headers.get("x-ratelimit-remaining"),
    })

    try:
        data = resp.json()
        print("Response JSON:")
        print(data)
    except Exception:
        print("Raw response:")
        print(resp.text)


async def main():
    print("API key loaded:", bool(API_KEY))
    if API_KEY:
        print("API key prefix:", API_KEY[:6] + "..." + API_KEY[-4:])

    await test_snippet_search()
    await test_paper_search()


if __name__ == "__main__":
    asyncio.run(main())
