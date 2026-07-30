from bm25_retriever import retrieve as retrieve_bm25
from config import BM25_TOP_K, FAISS_TOP_K
from metadata import infer_category


def _deduplication_key(result):
    """Create a stable key for duplicate detection."""

    chunk = result.get("chunk")

    if chunk is None:
        return None

    metadata = getattr(chunk, "metadata", {}) or {}

    chunk_id = metadata.get("chunk_id") or metadata.get("chunk")

    if chunk_id is not None:
        return f"chunk_id:{chunk_id}"

    return chunk.page_content.strip()


def _detect_category(query):
    """Infer a simple category from the user's query using keyword matching."""

    if not query:
        return None

    text = query.lower()

    if any(word in text for word in ("loan", "mortgage", "financing", "repayment")):
        return "loan"
    if any(word in text for word in ("atm", "withdrawal")):
        return "atm"
    if any(word in text for word in ("card", "debit", "credit")):
        return "card"
    if any(word in text for word in ("account", "balance", "statement")):
        return "account"

    return None


def _filter_by_category(results, category):
    """Restrict results to the requested category when possible."""

    if not category:
        return results

    filtered = []

    for result in results:
        chunk = result.get("chunk")
        if chunk is None:
            continue

        metadata = getattr(chunk, "metadata", {}) or {}
        chunk_category = metadata.get("category") or infer_category(getattr(chunk, "metadata", {}).get("source", ""))

        if chunk_category == category:
            filtered.append(result)

    return filtered


def hybrid_retrieve(query, faiss_top_k: int = FAISS_TOP_K, bm25_top_k: int = BM25_TOP_K):
    """
    Combine FAISS dense retrieval and BM25 sparse retrieval.

    The returned object format stays compatible with the existing pipeline so
    reranking and generation can remain unchanged.
    """

    if not query or not query.strip():
        return []

    detected_category = _detect_category(query)

    try:
        from retrieve import retrieve as retrieve_faiss
    except Exception:
        retrieve_faiss = None

    if retrieve_faiss is not None:
        faiss_results = retrieve_faiss(query, top_k=faiss_top_k)
    else:
        faiss_results = []

    bm25_results = retrieve_bm25(query, top_k=bm25_top_k)

    if detected_category:
        print(f"\nDetected Category: {detected_category}")
        faiss_results = _filter_by_category(faiss_results, detected_category)
        bm25_results = _filter_by_category(bm25_results, detected_category)
        print(f"Filtered Chunks: {len(faiss_results) + len(bm25_results)}")
    else:
        print("\nDetected Category: None")
        print("Searching entire knowledge base.")

    merged_results = []
    seen_keys = set()

    for result in faiss_results + bm25_results:
        key = _deduplication_key(result)

        if key is None or key in seen_keys:
            continue

        seen_keys.add(key)
        merged_results.append(result)

    return merged_results
