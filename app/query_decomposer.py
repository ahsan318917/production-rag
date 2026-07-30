import json

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


def decompose_query(query: str):
    """
    Split a query into sub-queries when it contains multiple independent
    information needs. If decomposition is unnecessary or fails, return the
    original query as a single-item list.
    """

    if not query or not query.strip():
        return [""]

    cleaned_query = query.strip()

    prompt = f"""
You are a query decomposition assistant.

Your job is only to split the user's query into sub-queries when it contains multiple independent information needs.

Rules:
- Never answer the question.
- Never retrieve information.
- Only split the query if there are multiple independent information needs.
- Preserve the user's wording as much as possible.
- Do not rewrite unless necessary.
- Return only valid JSON.

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
                        "You split complex user questions into simpler sub-queries. "
                        "Return only valid JSON with a 'sub_queries' array."
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
                    return cleaned_sub_queries

    except Exception:
        pass

    return [cleaned_query]
