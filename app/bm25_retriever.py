import pickle

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - exercised when rank_bm25 is unavailable
    BM25Okapi = None

from config import BM25_TOP_K, CHUNKS_FILE


def _load_chunks():
    """Load chunk objects from the persisted chunk store."""

    if not CHUNKS_FILE.exists():
        return []

    with open(CHUNKS_FILE, "rb") as handle:
        return pickle.load(handle)


def _extract_text(chunk):
    """Extract searchable text from a chunk-like object."""

    if chunk is None:
        return ""

    if hasattr(chunk, "page_content") and getattr(chunk, "page_content") is not None:
        return str(chunk.page_content)

    if isinstance(chunk, dict):
        for key in ("page_content", "text", "content"):
            value = chunk.get(key)
            if value:
                return str(value)

    if hasattr(chunk, "content") and getattr(chunk, "content") is not None:
        return str(chunk.content)

    return str(chunk)


def _tokenize(text):
    """Tokenize text using simple lowercase whitespace splitting."""

    if not text:
        return []

    return str(text).lower().split()


def retrieve(query, top_k: int = BM25_TOP_K):
    """
    Retrieve the top-k relevant chunks using BM25 lexical matching.

    Returns the same structure as the FAISS retriever so the rest of the
    pipeline remains compatible.
    """

    chunks = _load_chunks()

    if not chunks:
        return []

    if top_k is None or top_k <= 0:
        top_k = BM25_TOP_K

    tokenized_documents = [
        _tokenize(_extract_text(chunk))
        for chunk in chunks
    ]

    query_tokens = _tokenize(query)

    if not query_tokens:
        return []

    if BM25Okapi is not None:
        bm25 = BM25Okapi(tokenized_documents)
        scores = bm25.get_scores(query_tokens)
    else:
        scores = []

        for document_tokens in tokenized_documents:
            overlap = sum(
                1 for token in query_tokens if token in document_tokens
            )
            scores.append(float(overlap))

    ranked_results = []

    for vector_id, score in enumerate(scores):
        if score <= 0:
            continue

        ranked_results.append(
            {
                "vector_id": int(vector_id),
                "score": float(score),
                "chunk": chunks[vector_id],
            }
        )

    ranked_results.sort(key=lambda item: item["score"], reverse=True)

    return ranked_results[:top_k]
