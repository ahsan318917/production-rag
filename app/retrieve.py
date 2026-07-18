import faiss
import numpy as np

from config import (
    TOP_K,
    MIN_SIMILARITY,
)
from embeddings import generate_embeddings
from faiss_store import (
    load_index,
    load_chunks,
)

# -------------------------------------------------
# Retrieval Settings
# -------------------------------------------------

MIN_SIMILARITY = 0.60

# -------------------------------------------------
# Search FAISS Index
# -------------------------------------------------

def search_index(query_embedding, top_k=TOP_K):
    """
    Search the FAISS index using cosine similarity.

    Returns:
        scores : Cosine similarity scores
        indices: Matching chunk indices
    """

    index = load_index()

    query_vector = np.array(
        [query_embedding],
        dtype=np.float32
    )

    # Normalize query vector
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(
        query_vector,
        top_k
    )

    return scores[0], indices[0]


# -------------------------------------------------
# Retrieve Relevant Chunks
# -------------------------------------------------

def retrieve(query, top_k=TOP_K):
    """
    Retrieve the most relevant chunks.

    Returns:
        [
            {
                "chunk": chunk,
                "score": similarity_score
            }
        ]
    """

    print("=" * 60)
    print("Retrieving Documents")
    print("=" * 60)

    print("\nGenerating query embedding...")

    query_embedding = generate_embeddings(query)

    print("Embedding generated.")

    print("\nSearching FAISS index...")

    scores, indices = search_index(
        query_embedding,
        top_k
    )

    chunks = load_chunks()

    retrieved_chunks = []

    for score, idx in zip(scores, indices):

        if idx == -1:
            continue

        # Filter weak matches
        if score < MIN_SIMILARITY:
            continue

        retrieved_chunks.append(
            {
                "chunk": chunks[idx],
                "score": float(score)
            }
        )

    print(f"Retrieved {len(retrieved_chunks)} relevant chunk(s).\n")

    return retrieved_chunks


# -------------------------------------------------
# Display Results
# -------------------------------------------------

def display_results(results):

    print("=" * 60)
    print("Retrieved Chunks")
    print("=" * 60)

    if not results:
        print("No relevant chunks found.")
        return

    for i, result in enumerate(results, start=1):

        chunk = result["chunk"]

        print("-" * 60)
        print(f"Rank {i}")
        print("-" * 60)

        print(f"Similarity : {result['score']:.4f}")

        print("\nContent:\n")

        print(chunk.page_content)

        print("\nMetadata:")

        print(chunk.metadata)

        print()


# -------------------------------------------------
# Main (Testing)
# -------------------------------------------------

def main():

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        results = retrieve(query)

        display_results(results)


# -------------------------------------------------
# Entry Point
# -------------------------------------------------

if __name__ == "__main__":
    main()