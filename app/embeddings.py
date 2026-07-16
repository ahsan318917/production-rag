from ollama import embed

from config import EMBED_MODEL


def generate_embedding(text: str):
    """
    Generate embedding for a single text.
    """

    response = embed(
        model=EMBED_MODEL,
        input=text
    )

    return response["embeddings"][0]


def generate_embeddings(texts):
    """
    Generate embeddings for multiple texts in one Ollama request.
    """

    print(f"Generating embeddings for {len(texts)} chunks...")

    response = embed(
        model=EMBED_MODEL,
        input=texts
    )

    embeddings = response["embeddings"]

    print("Embeddings generated successfully.")

    return embeddings