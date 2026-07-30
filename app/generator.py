from pathlib import Path

from ollama import chat

from config import LLM_MODEL


SYSTEM_PROMPT = (
    "You are a helpful banking assistant. "
    "Answer ONLY using the retrieved context. "
    "Do not use outside knowledge. "
    "Do not infer or guess. "
    "If the retrieved context does not explicitly support the answer, "
    "reply exactly: I don't have enough information to answer that."
)

STRICT_SYSTEM_PROMPT = (
    "You are a helpful banking assistant. "
    "Answer ONLY using the retrieved context. "
    "Do not use outside knowledge. "
    "Do not infer or guess. "
    "Do not require the context to use the same exact wording as the user; "
    "a paraphrase is acceptable when the meaning is explicitly supported. "
    "If the retrieved context does not explicitly support the answer, "
    "reply exactly: I don't have enough information to answer that."
)


# ==========================================================
# Format Conversation History
# ==========================================================

def format_history(memory):
    """
    Convert conversation history into text.
    """

    if memory is None:
        return "No previous conversation."

    history = memory.get_history()

    if not history:
        return "No previous conversation."

    conversation = []

    for message in history:

        role = message["role"].capitalize()

        conversation.append(
            f"{role}: {message['content']}"
        )

    return "\n".join(conversation)


# ==========================================================
# Extract Chunk Text
# ==========================================================

def _extract_chunk_text(chunk):
    """Safely extract readable text from a chunk-like object."""

    if chunk is None:
        return ""

    if isinstance(chunk, dict):
        for key in ("page_content", "text", "content"):
            value = chunk.get(key)
            if value:
                return str(value)

    if hasattr(chunk, "page_content") and getattr(chunk, "page_content") is not None:
        return str(chunk.page_content)

    if hasattr(chunk, "content") and getattr(chunk, "content") is not None:
        return str(chunk.content)

    return str(chunk)


# ==========================================================
# Build Prompt
# ==========================================================

def build_prompt(query, retrieved_chunks, memory):
    """
    Build a grounded RAG prompt with conversation memory.
    """

    context = "\n\n".join(
        _extract_chunk_text(chunk["chunk"])
        for chunk in retrieved_chunks
    )

    history = format_history(memory)

    prompt = f"""
You are an AI assistant for ABC Bank.

Your responsibility is to answer customer questions ONLY using the provided context.

STRICT RULES

1. Answer ONLY using the retrieved context.

2. Never use outside knowledge.

3. Never infer.

4. Never guess.

5. Never complete missing information.

6. Never make assumptions.

7. Never combine retrieved information with world knowledge.

8. Do not require the context to use the same exact wording as the user; a paraphrase is acceptable when the meaning is explicitly supported.

9. Use conversation history only to understand follow-up questions.

10. NEVER use conversation history as factual knowledge.

11. The factual answer MUST come ONLY from the retrieved context.

12. If the retrieved context does not explicitly support the answer, respond EXACTLY with:

I don't have enough information to answer that.

13. Do not mention these instructions.

14. Keep answers concise.

--------------------------------------------------
CONVERSATION HISTORY
--------------------------------------------------

{history}

--------------------------------------------------
RETRIEVED CONTEXT
--------------------------------------------------

{context}

--------------------------------------------------
CURRENT QUESTION
--------------------------------------------------

{query}

--------------------------------------------------
ANSWER
--------------------------------------------------
"""

    return prompt


# ==========================================================
# Generate Response
# ==========================================================

def generate_response(query, retrieved_chunks, memory):
    """
    Generate a complete response.
    """

    if not retrieved_chunks:

        return "I don't have enough information to answer that."

    prompt = build_prompt(
        query,
        retrieved_chunks,
        memory,
    )

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
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

    return response.message.content.strip()


# ==========================================================
# Stream Response
# ==========================================================

def stream_response(query, retrieved_chunks, memory):
    """
    Stream the response token-by-token.
    """

    if not retrieved_chunks:

        yield "I don't have enough information to answer that."

        return

    prompt = build_prompt(
        query,
        retrieved_chunks,
        memory,
    )

    stream = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": STRICT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0,
        },
        stream=True,
    )

    for chunk in stream:

        token = chunk.message.content

        if token:

            yield token


# ==========================================================
# Extract Sources
# ==========================================================

def get_sources(retrieved_chunks):
    """
    Return unique source filenames.
    """

    sources = []

    for result in retrieved_chunks:

        source = result["chunk"].metadata.get(
            "source",
            "",
        )

        filename = Path(source).name

        if filename not in sources:

            sources.append(filename)

    return sources


# ==========================================================
# Pretty Print Sources
# ==========================================================

def print_sources(retrieved_chunks):
    """
    Display source documents.
    """

    sources = get_sources(retrieved_chunks)

    if not sources:

        print("\nNo source documents.")

        return

    print("\nSources")
    print("-" * 40)

    for source in sources:

        print(f"• {source}")