import ollama

from config import EMBED_MODEL


def generate_embeddings(texts):
    """
    Generate embeddings using Ollama.

    Accepts:
        - A single string
        - A list of strings

    Returns:
        - A single embedding (list[float]) if input is a string
        - A list of embeddings if input is a list
    """

    # -------------------------------------
    # Single Query
    # -------------------------------------

    if isinstance(texts, str):

        response = ollama.embed(
            model=EMBED_MODEL,
            input=texts
        )

        return response["embeddings"][0]

    # -------------------------------------
    # Multiple Chunks
    # -------------------------------------

    elif isinstance(texts, list):

        print(f"Generating embeddings for {len(texts)} chunks...")

        response = ollama.embed(
            model=EMBED_MODEL,
            input=texts
        )

        print("Embeddings generated successfully.")

        return response["embeddings"]

    else:
        raise TypeError(
            "Input must be a string or a list of strings."
        )