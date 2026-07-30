from sentence_transformers import CrossEncoder

from config import (
    MIN_RERANK_SCORE,
    RERANK_TOP_K,
)


MODEL_CANDIDATES = [
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]

_reranker = None
_reranker_model_name = None


def _load_reranker():
    global _reranker, _reranker_model_name

    if _reranker is not None:
        return _reranker

    last_error = None

    for model_name in MODEL_CANDIDATES:
        try:
            _reranker = CrossEncoder(model_name)
            _reranker_model_name = model_name
            break
        except Exception as exc:
            last_error = exc
            continue

    if _reranker is None:
        raise RuntimeError(
            "Failed to load a local reranker model. "
            "Install sentence-transformers and ensure at least one of the following models is available: "
            f"{MODEL_CANDIDATES}."
        ) from last_error

    return _reranker


def rerank(query, retrieved_chunks, top_n: int = RERANK_TOP_K):
    """Rerank FAISS-retrieved chunks using a local cross-encoder model."""

    if not query or not retrieved_chunks:
        return []

    if top_n is None or top_n <= 0:
        top_n = RERANK_TOP_K

    reranker = _load_reranker()

    pairs = [
        (query, chunk["chunk"].page_content)
        for chunk in retrieved_chunks
    ]

    scores = reranker.predict(pairs)

    results = []

    for item, score in zip(retrieved_chunks, scores):
        numeric_score = float(score)

        if numeric_score < MIN_RERANK_SCORE:
            continue

        results.append(
            {
                "chunk": item["chunk"],
                "vector_id": item["vector_id"],
                "score": numeric_score,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_n]
