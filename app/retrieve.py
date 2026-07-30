import numpy as np

from config import (
    TOP_K,
    MIN_SIMILARITY,
)

from embeddings import generate_embeddings

from faiss_store import (
    load_index,
    load_chunks,
    search,
)


# =====================================================
# Retrieve Relevant Chunks
# =====================================================

def retrieve(query: str, top_k: int = TOP_K):
    """
    Retrieve the most relevant chunks from FAISS.

    Returns:
    [
        {
            "vector_id": int,
            "score": float,
            "chunk": Document
        }
    ]
    """

    print("=" * 60)
    print("Retrieving Documents")
    print("=" * 60)

    print("\nGenerating query embedding...")

    query_embedding = generate_embeddings(query)

    print("Embedding generated.")

    print("\nLoading FAISS index...")

    index = load_index()

    print("Searching index...")

    scores, vector_ids = search(
        index=index,
        query_embedding=query_embedding,
        top_k=top_k,
    )

    chunks = load_chunks()

    retrieved = []

    for score, vector_id in zip(scores, vector_ids):

        # Invalid FAISS result
        if vector_id == -1:
            continue

        # Ignore weak matches
        if score < MIN_SIMILARITY:
            continue

        # Vector removed from storage
        if vector_id not in chunks:
            continue

        retrieved.append(
            {
                "vector_id": int(vector_id),
                "score": float(score),
                "chunk": chunks[vector_id],
            }
        )

    # Highest similarity first
    retrieved.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    print(f"\nRetrieved {len(retrieved)} relevant chunk(s).")

    return retrieved


# =====================================================
# Display Results
# =====================================================

def display_results(results):

    print("\n" + "=" * 60)
    print("Retrieved Chunks")
    print("=" * 60)

    if not results:
        print("No relevant chunks found.")
        return

    for rank, result in enumerate(results, start=1):

        chunk = result["chunk"]
        metadata = chunk.metadata

        print("\n" + "-" * 60)
        print(f"Rank      : {rank}")
        print(f"Vector ID : {result['vector_id']}")
        print(f"Similarity: {result['score']:.4f}")

        print(
            f"Source    : {metadata.get('source', 'Unknown')}"
        )

        print(
            f"Chunk No. : {metadata.get('chunk', metadata.get('chunk_id', 'N/A'))}"
        )

        print("-" * 60)

        print(chunk.page_content)

    print("\n" + "=" * 60)


# =====================================================
# Interactive CLI
# =====================================================

def main():

    print("=" * 60)
    print("Production RAG Retriever")
    print("=" * 60)

    while True:

        query = input(
            "\nAsk a question (type 'exit' to quit): "
        ).strip()

        if not query:
            continue

        if query.lower() == "exit":
            break

        results = retrieve(query)

        display_results(results)


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":
    main()