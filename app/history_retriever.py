from ollama import chat

from config import LLM_MODEL


def format_history(memory):
    """
    Convert conversation history into a plain-text format.
    """

    if memory is None:
        return "No previous conversation."

    history = memory.get_history()

    if not history:
        return "No previous conversation."

    conversation = []

    for message in history:

        role = message.get("role", "user").capitalize()
        content = message.get("content", "")

        conversation.append(f"{role}: {content}")

    return "\n".join(conversation)


def rewrite_query(query, memory):
    """
    Rewrite a follow-up question into a standalone question using
    conversation history, without answering the question.
    """

    if not query:
        return ""

    cleaned_query = query.strip()

    if not cleaned_query:
        return ""

    if memory is None:
        return cleaned_query

    history = memory.get_history()

    if not history:
        return cleaned_query

    history_text = format_history(memory)

    prompt = f"""
You are a query rewriting assistant.

Rewrite the current user question into a complete standalone question.
Use conversation history only to resolve references such as:
- it
- they
- that
- this
- those

Rules:
- Preserve the user's intent.
- Do NOT answer the question.
- Do NOT add information.
- If the question is already standalone, return it unchanged.
- Output ONLY the rewritten question.

--------------------------------------------------
CONVERSATION HISTORY
--------------------------------------------------
{history_text}

--------------------------------------------------
CURRENT QUESTION
--------------------------------------------------
{cleaned_query}

--------------------------------------------------
REWRITTEN QUESTION
--------------------------------------------------
"""

    try:
        response = chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite follow-up questions into standalone questions. "
                        "Do not answer the question. "
                        "Return only the rewritten question."
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

        rewritten_question = response.message.content.strip()

        if rewritten_question:
            return rewritten_question

    except Exception:
        pass

    return cleaned_query