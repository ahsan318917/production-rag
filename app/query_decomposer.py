import json
import re

from ollama import chat

from config import LLM_MODEL


def _parse_json_safe(text):
    """Safely extract a JSON object from model output."""

    if not text:
        return None

    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def _looks_multitask(query: str) -> bool:
    """Heuristically detect obvious multi-intent queries before calling the LLM."""
    if not query:
        return False

    lowered = query.lower()

    markers = [
        " and ",
        " also ",
        " as well as ",
        " along with ",
        " besides ",
        " both ",
        " all ",
        " compare ",
        " difference between ",
    ]

    if re.search(r"\?\s*\?", query):
        return True

    if any(marker in lowered for marker in markers):
        return True

    if "," in query and query.count(",") >= 1:
        return True

    return False


def decompose_query(query: str):
    """
    Only split a query when it clearly contains multiple independent information needs.
    If decomposition is unnecessary, or parsing fails, return the original query as a
    single-item list.
    """

    if not query or not query.strip():
        return [""]

    cleaned_query = query.strip()

    if not _looks_multitask(cleaned_query):
        return [cleaned_query]

    prompt = f"""
You are NOT answering the question.
You are NOT improving the question.
You are NOT expanding the question.
You are ONLY deciding whether the query contains multiple independent questions.
If there is only one information need, return it unchanged.
If decomposition is not required, do not change wording except for trivial whitespace cleanup.
Do not invent calculations, assumptions, missing entities, or follow-up questions.
Return only valid JSON.

User query:
{cleaned_query}

Expected JSON format:
{{
  "sub_queries": [
    "...",
    "..."
  ]
}}
"""

    try:
        response = chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You decide whether a user query contains multiple independent information needs. "
                        "Return only valid JSON with a 'sub_queries' array. "
                        "If there is only one information need, return the original query unchanged."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0,
            },
        )

        payload = _parse_json_safe(response.message.content)

        if isinstance(payload, dict):
            sub_queries = payload.get("sub_queries")

            if isinstance(sub_queries, list):
                cleaned_sub_queries = [
                    str(item).strip()
                    for item in sub_queries
                    if str(item).strip()
                ]

                if cleaned_sub_queries:
                    if len(cleaned_sub_queries) == 1 and cleaned_sub_queries[0] == cleaned_query:
                        return cleaned_sub_queries

                    if len(cleaned_sub_queries) >= 2:
                        return cleaned_sub_queries

    except Exception:
        pass

    return [cleaned_query]
