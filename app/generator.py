from ollama import chat

from config import LLM_MODEL


# -------------------------------------------------
# Build Prompt
# -------------------------------------------------

def build_prompt(query, retrieved_chunks):
    """
    Build the prompt using retrieved context.
    """

    context = "\n\n".join(
        chunk["chunk"].page_content
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are an intelligent AI assistant for ABC Bank.

Your job is to answer ONLY using the information provided in the context.

Rules:

1. Never make up information.

2. If the answer is not present in the context, reply exactly:

"I don't have enough information to answer that."

3. Be concise.

4. Use complete sentences.

Context:
-------------------------
{context}
-------------------------

Question:
{query}

Answer:
"""

    return prompt


# -------------------------------------------------
# Generate Response
# -------------------------------------------------

def generate_response(query, retrieved_chunks):
    """
    Generate an answer using the retrieved chunks.
    """

    prompt = build_prompt(
        query,
        retrieved_chunks
    )

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content


# -------------------------------------------------
# Stream Response
# -------------------------------------------------

def stream_response(query, retrieved_chunks):
    """
    Stream the answer token-by-token.
    """

    prompt = build_prompt(
        query,
        retrieved_chunks
    )

    stream = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=True
    )

    for chunk in stream:

        token = chunk.message.content

        if token:
            yield token

# -------------------------------------------------
# Extract Source Documents
# -------------------------------------------------

from pathlib import Path


def get_sources(retrieved_chunks):
    """
    Extract unique source document names.
    """

    sources = []

    for item in retrieved_chunks:

        source = item["chunk"].metadata.get("source", "")

        filename = Path(source).name

        if filename not in sources:
            sources.append(filename)

    return sources