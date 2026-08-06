import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.config import ANSWER_SIMILARITY_THRESHOLD, CONTEXT_SIMILARITY_THRESHOLD
from app.context_compressor import compress_context
from app.generator import get_sources, stream_response
from app.history_retriever import rewrite_query
from app.hybrid_retriever import hybrid_retrieve
from app.memory import ConversationMemory
from app.query_decomposer import decompose_query
from app.reranker import rerank, _load_reranker
from evaluation.semantic_utils import (
    compute_answer_relevancy,
    compute_bleu,
    compute_context_precision,
    compute_context_recall,
    compute_entity_match,
    compute_faithfulness,
    compute_numeric_match,
    compute_rouge_l,
    compute_semantic_similarity,
    compute_token_metrics,
    get_semantic_model,
    is_unsupported_refusal,
    normalize_text,
)

DATASET_PATH = ROOT_DIR / "evaluation" / "evaluation_dataset.json"
RESULTS_PATH = ROOT_DIR / "evaluation" / "evaluation_results.json"
SUMMARY_PATH = ROOT_DIR / "evaluation" / "evaluation_summary.json"
DASHBOARD_PATH = ROOT_DIR / "evaluation" / "evaluation_dashboard.json"
UNSUPPORTED_ANSWER = "i don't have enough information to answer that."


def _is_unsupported_answer(text: str) -> bool:
    """Return True when answer is an unsupported fallback or refusal."""
    return is_unsupported_refusal(text) or normalize_text(text) == UNSUPPORTED_ANSWER


def _extract_chunk_text(chunk: Any) -> str:
    """Safely extract readable text from a chunk-like object."""
    if chunk is None:
        return ""

    if isinstance(chunk, dict):
        for key in ("page_content", "text", "content"):
            if key in chunk and chunk.get(key):
                return str(chunk[key])

    if hasattr(chunk, "page_content") and getattr(chunk, "page_content") is not None:
        return str(chunk.page_content)

    if hasattr(chunk, "content") and getattr(chunk, "content") is not None:
        return str(chunk.content)

    return str(chunk)


def infer_question_category(question: str, dataset_category: Optional[str] = None, expected_answer: str = "") -> str:
    """Classify a question into standard functional categories."""
    if _is_unsupported_answer(expected_answer):
        return "Unsupported"

    q_lower = question.lower()

    if any(k in q_lower for k in ["age", "income", "cnic", "eligible", "requirement", "document"]):
        return "Eligibility"
    if any(k in q_lower for k in ["fee", "charge", "cost", "annual", "late"]):
        return "Fees"
    if any(k in q_lower for k in ["rate", "profit", "interest", "percent", "%"]):
        return "Rates"
    if any(k in q_lower for k in ["limit", "maximum", "minimum", "amount"]):
        return "Limits"
    if any(k in q_lower for k in ["how", "apply", "process", "steps", "transfer", "withdraw", "cancel"]):
        return "Procedures"
    if any(k in q_lower for k in ["password", "pin", "otp", "biometric", "block", "lost", "secure"]):
        return "Security"
    if any(k in q_lower for k in ["expire", "valid", "tenure", "days", "months", "years"]):
        return "Dates"
    if any(k in q_lower for k in ["what is", "definition", "meaning"]):
        return "Definitions"

    if dataset_category and dataset_category.strip():
        cat = dataset_category.strip().replace("_", " ").title()
        return cat

    return "General"


def prewarm_models():
    """Warm up evaluation and pipeline models once before running the dataset loop."""
    print("Pre-warming evaluation & pipeline models...")
    try:
        get_semantic_model()
    except Exception:
        pass

    try:
        _load_reranker()
    except Exception:
        pass


def _classify_failure(
    answer_correct: bool,
    source_correct: bool,
    retrieval_success: bool,
    hallucinated: bool,
    faithfulness: float,
    all_retrieved_sources: List[str],
    reranked_sources: List[str],
    final_sources: List[str],
    expected_answer: str,
    generated_answer: str,
    expected_source: Optional[str],
) -> Optional[str]:
    """
    Classify failure reason into 9 non-overlapping categories:
    1. Unsupported Success (None)
    2. Retrieval Failure
    3. Reranker Failure
    4. Context Compression Failure
    5. Wrong Source
    6. Hallucination
    7. Generator Failure
    8. Evaluation False Negative
    9. Success (None)
    """
    is_exp_unsupported = _is_unsupported_answer(expected_answer)
    is_gen_unsupported = _is_unsupported_answer(generated_answer)

    # 1. Unsupported question correctly refused -> Success
    if is_exp_unsupported and is_gen_unsupported:
        return None

    # 2. Retrieval Failure (expected source missing from initial retrieval)
    if expected_source is not None:
        exp_norm = Path(str(expected_source)).name.lower()
        all_norm = [Path(str(s)).name.lower() for s in all_retrieved_sources]
        rerank_norm = [Path(str(s)).name.lower() for s in reranked_sources]
        final_norm = [Path(str(s)).name.lower() for s in final_sources]

        if not all_norm or not any(exp_norm in s or s in exp_norm for s in all_norm):
            return "Retrieval Failure"

        # 3. Reranker Failure
        if not any(exp_norm in s or s in exp_norm for s in rerank_norm):
            return "Reranker Failure"

        # 4. Context Compression Failure
        if not any(exp_norm in s or s in exp_norm for s in final_norm):
            return "Context Compression Failure"

        # 5. Wrong Source
        if not source_correct:
            return "Wrong Source"

    elif not retrieval_success:
        return "Retrieval Failure"

    # 6. Hallucination
    if hallucinated or faithfulness < 0.70:
        return "Hallucination"

    # 7. Generator Failure
    if not answer_correct:
        return "Generator Failure"

    # 8. All criteria passed -> Success
    if answer_correct and (expected_source is None or source_correct):
        return None

    return "Unknown"


def _to_json_safe(value: Any) -> Any:
    """Convert evaluation payloads into JSON-serializable structures."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]

    if hasattr(value, "to_dict"):
        try:
            return _to_json_safe(value.to_dict())
        except Exception:
            pass

    if hasattr(value, "page_content") or hasattr(value, "content") or hasattr(value, "metadata"):
        payload: Dict[str, Any] = {}

        if hasattr(value, "page_content"):
            payload["page_content"] = _to_json_safe(getattr(value, "page_content"))

        if hasattr(value, "content"):
            payload["content"] = _to_json_safe(getattr(value, "content"))

        if hasattr(value, "metadata"):
            payload["metadata"] = _to_json_safe(getattr(value, "metadata"))

        return payload

    if hasattr(value, "__dict__"):
        return {
            key: _to_json_safe(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return str(value)


def load_dataset() -> List[Dict[str, Any]]:
    """Load the evaluation dataset from disk."""
    if not DATASET_PATH.exists():
        print(f"Dataset file not found: {DATASET_PATH}")
        return []

    try:
        with DATASET_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in dataset file: {exc}")
        return []
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"Failed to read dataset file: {exc}")
        return []

    if not isinstance(payload, list):
        print("Dataset must be a JSON list of question objects.")
        return []

    if not payload:
        print("Evaluation dataset is empty.")
        return []

    return payload


def _collect_stream_response(query: str, retrieved_chunks: List[Any], memory: ConversationMemory) -> str:
    """Collect streamed generator output into a single string."""
    response_tokens = []

    for token in stream_response(
        query=query,
        retrieved_chunks=retrieved_chunks,
        memory=memory,
    ):
        response_tokens.append(token)

    return "".join(response_tokens).strip()


def _run_rag_pipeline(question: str, memory: ConversationMemory) -> Dict[str, Any]:
    """
    Run the existing RAG pipeline for a single question and capture per-stage latencies.
    Does not modify any underlying pipeline modules in app/.
    """
    pipeline_start = time.perf_counter()

    try:
        rewrite_start = time.perf_counter()
        standalone_query = rewrite_query(question, memory)
        rewrite_latency = round(time.perf_counter() - rewrite_start, 3)

        decomp_start = time.perf_counter()
        sub_queries = decompose_query(standalone_query)
        decomposition_latency = round(time.perf_counter() - decomp_start, 3)

        all_retrieved_chunks = []
        seen_chunk_keys = set()
        retrieval_start = time.perf_counter()

        for sub_query in sub_queries:
            retrieved_chunks = hybrid_retrieve(sub_query)

            for chunk_result in retrieved_chunks:
                chunk = chunk_result.get("chunk")
                if chunk is None:
                    continue

                metadata = getattr(chunk, "metadata", {}) or {}
                chunk_id = metadata.get("chunk_id") or metadata.get("chunk")

                if chunk_id is not None:
                    key = f"chunk_id:{chunk_id}"
                else:
                    key = getattr(chunk, "page_content", str(chunk)).strip()

                if key in seen_chunk_keys:
                    continue

                seen_chunk_keys.add(key)
                all_retrieved_chunks.append(chunk_result)

        retrieval_latency = round(time.perf_counter() - retrieval_start, 3)

        if not all_retrieved_chunks:
            generated_answer = "I don't have enough information to answer that."
            retrieved_sources = []
            retrieval_success = False
            reranked_chunks = []
            final_chunks = []
            reranker_top_score = None
            reranker_scores = []
            reranker_latency = 0.0
            compression_latency = 0.0
            generation_latency = 0.0
        else:
            rerank_start = time.perf_counter()
            reranked_chunks = rerank(query=question, retrieved_chunks=all_retrieved_chunks)
            reranker_latency = round(time.perf_counter() - rerank_start, 3)

            if not reranked_chunks:
                generated_answer = "I don't have enough information to answer that."
                retrieved_sources = []
                retrieval_success = False
                final_chunks = []
                reranker_top_score = None
                reranker_scores = []
                compression_latency = 0.0
                generation_latency = 0.0
            else:
                compression_start = time.perf_counter()
                compressed_chunks = compress_context(query=question, retrieved_chunks=reranked_chunks)
                compression_latency = round(time.perf_counter() - compression_start, 3)

                final_chunks = compressed_chunks if compressed_chunks else reranked_chunks
                retrieved_sources = get_sources(final_chunks)
                retrieval_success = True

                reranker_scores = [round(float(item["score"]), 4) for item in reranked_chunks if "score" in item]
                reranker_top_score = reranker_scores[0] if reranker_scores else None

                generation_start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        _collect_stream_response,
                        question,
                        final_chunks,
                        memory,
                    )

                    try:
                        generated_answer = future.result(timeout=25)
                    except TimeoutError:
                        generated_answer = "I don't have enough information to answer that."
                    except Exception:
                        generated_answer = "I don't have enough information to answer that."

                generation_latency = round(time.perf_counter() - generation_start, 3)

                if not generated_answer:
                    generated_answer = "I don't have enough information to answer that."

        total_latency = round(time.perf_counter() - pipeline_start, 3)

        memory.add_user_message(question)
        memory.add_assistant_message(generated_answer)

        return {
            "generated_answer": generated_answer,
            "retrieved_sources": retrieved_sources,
            "retrieval_success": retrieval_success,
            "all_retrieved_chunks": all_retrieved_chunks,
            "reranked_chunks": reranked_chunks,
            "final_chunks": final_chunks,
            "reranker_top_score": reranker_top_score,
            "reranker_scores": reranker_scores,
            "rewrite_latency": rewrite_latency,
            "decomposition_latency": decomposition_latency,
            "retrieval_latency": retrieval_latency,
            "reranker_latency": reranker_latency,
            "compression_latency": compression_latency,
            "generation_latency": generation_latency,
            "total_latency": total_latency,
        }

    except Exception as exc:  # pragma: no cover - defensive guard
        total_latency = round(time.perf_counter() - pipeline_start, 3)
        return {
            "generated_answer": "I don't have enough information to answer that.",
            "retrieved_sources": [],
            "retrieval_success": False,
            "all_retrieved_chunks": [],
            "reranked_chunks": [],
            "final_chunks": [],
            "reranker_top_score": None,
            "reranker_scores": [],
            "rewrite_latency": 0.0,
            "decomposition_latency": 0.0,
            "retrieval_latency": 0.0,
            "reranker_latency": 0.0,
            "compression_latency": 0.0,
            "generation_latency": 0.0,
            "total_latency": total_latency,
            "error": str(exc),
        }


def evaluate_question(question_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one question against the production RAG evaluation suite."""
    question = question_entry.get("question", "")
    expected_answer = question_entry.get("expected_answer", "")
    expected_source = question_entry.get("expected_source")
    dataset_category = question_entry.get("category")

    memory = ConversationMemory()
    pipeline_result = _run_rag_pipeline(question, memory)

    generated_answer = pipeline_result.get("generated_answer", "")
    retrieved_sources = pipeline_result.get("retrieved_sources", [])
    all_chunks = pipeline_result.get("all_retrieved_chunks", [])
    reranked_chunks = pipeline_result.get("reranked_chunks", [])
    final_chunks = pipeline_result.get("final_chunks", []) or reranked_chunks or all_chunks

    category = infer_question_category(question, dataset_category, expected_answer)

    # 1. Multi-Stage Hybrid Answer Correctness
    semantic_similarity = compute_semantic_similarity(generated_answer, expected_answer)
    numeric_match = compute_numeric_match(generated_answer, expected_answer)
    entity_match = compute_entity_match(generated_answer, expected_answer)
    bleu = compute_bleu(generated_answer, expected_answer)
    rouge_l = compute_rouge_l(generated_answer, expected_answer)
    token_metrics = compute_token_metrics(generated_answer, expected_answer)

    is_exp_unsupported = _is_unsupported_answer(expected_answer)
    is_gen_unsupported = _is_unsupported_answer(generated_answer)

    if is_exp_unsupported and is_gen_unsupported:
        answer_correct = True
    elif is_exp_unsupported != is_gen_unsupported:
        answer_correct = False
    elif numeric_match and entity_match:
        answer_correct = True
    elif semantic_similarity >= ANSWER_SIMILARITY_THRESHOLD:
        answer_correct = True
    elif token_metrics["token_f1"] >= 0.65 or rouge_l >= 0.70:
        answer_correct = True
    else:
        answer_correct = False

    # 2. Retrieval, Hit@k, MRR & Source Ranks
    all_retrieved_sources = get_sources(all_chunks)
    reranked_sources = get_sources(reranked_chunks)

    if expected_source is None:
        source_correct = not retrieved_sources
        hit_at_1 = None
        hit_at_3 = None
        hit_at_5 = None
        mrr = None
        correct_source_rank = None
        retrieval_precision = 1.0 if not retrieved_sources else 0.0
        retrieval_recall = 1.0
        retrieval_f1 = 1.0 if not retrieved_sources else 0.0
    else:
        exp_norm = Path(str(expected_source)).name.lower()
        norm_final = [Path(str(s)).name.lower() for s in retrieved_sources]

        source_correct = any(exp_norm in s or s in exp_norm for s in norm_final)

        # Hit@k inspected over reranked & retrieved chunks in order
        candidate_sources = reranked_sources if reranked_sources else all_retrieved_sources
        norm_candidates = [Path(str(s)).name.lower() for s in candidate_sources]

        correct_source_rank = None
        for idx, s in enumerate(norm_candidates, start=1):
            if exp_norm in s or s in exp_norm:
                correct_source_rank = idx
                break

        hit_at_1 = 1 if (correct_source_rank is not None and correct_source_rank <= 1) else 0
        hit_at_3 = 1 if (correct_source_rank is not None and correct_source_rank <= 3) else 0
        hit_at_5 = 1 if (correct_source_rank is not None and correct_source_rank <= 5) else 0
        mrr = round(1.0 / correct_source_rank, 4) if correct_source_rank is not None else 0.0

        # Retrieval P/R/F1
        retrieval_precision = 1.0 if source_correct else 0.0
        retrieval_recall = 1.0 if source_correct else 0.0
        retrieval_f1 = 1.0 if source_correct else 0.0

    # Extract metadata arrays
    retrieved_categories = []
    retrieved_documents = []
    retrieved_sections = []

    for item in all_chunks:
        chunk = item.get("chunk")
        if chunk is not None:
            meta = getattr(chunk, "metadata", {}) or {}
            cat = meta.get("category")
            doc = meta.get("document")
            sec = meta.get("section")

            if cat and cat not in retrieved_categories:
                retrieved_categories.append(cat)
            if doc and doc not in retrieved_documents:
                retrieved_documents.append(doc)
            if sec and sec not in retrieved_sections:
                retrieved_sections.append(sec)

    # 3. Context Recall, Context Precision, Faithfulness & Answer Relevancy
    retrieved_context = "\n\n".join(_extract_chunk_text(item.get("chunk")) for item in final_chunks if item.get("chunk"))
    context_recall = compute_context_recall(expected_answer, generated_answer, retrieved_context)
    context_contains_answer = context_recall >= CONTEXT_SIMILARITY_THRESHOLD

    faithfulness = compute_faithfulness(generated_answer, retrieved_context)
    context_precision = compute_context_precision(question, final_chunks)
    answer_relevancy = compute_answer_relevancy(question, generated_answer)

    # 4. Hallucination Detection
    retrieval_success = pipeline_result.get("retrieval_success", False)
    hallucinated = bool(
        retrieval_success
        and not is_gen_unsupported
        and (faithfulness < 0.70 or (not answer_correct and context_recall < CONTEXT_SIMILARITY_THRESHOLD))
    )

    # 5. Pipeline Confidence Score (0 - 100%)
    reranker_top = pipeline_result.get("reranker_top_score")
    rerank_val = reranker_top if reranker_top is not None else (1.0 if retrieval_success else 0.0)
    pipeline_confidence = round(100.0 * (0.3 * rerank_val + 0.3 * (1.0 if source_correct else 0.0) + 0.2 * context_recall + 0.2 * semantic_similarity), 1)

    # 6. Failure Classification (9 categories)
    failure_reason = _classify_failure(
        answer_correct=answer_correct,
        source_correct=source_correct,
        retrieval_success=retrieval_success,
        hallucinated=hallucinated,
        faithfulness=faithfulness,
        all_retrieved_sources=all_retrieved_sources,
        reranked_sources=reranked_sources,
        final_sources=retrieved_sources,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
        expected_source=expected_source,
    )

    return _to_json_safe({
        "question": question,
        "category": category,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "expected_source": expected_source,
        "source_correct": source_correct,
        "semantic_similarity": semantic_similarity,
        "numeric_match": numeric_match,
        "entity_match": entity_match,
        "token_f1": token_metrics["token_f1"],
        "bleu": bleu,
        "rouge_l": rouge_l,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "pipeline_confidence": pipeline_confidence,
        "answer_correct": answer_correct,
        "exact_match": token_metrics["exact_match"],
        "token_precision": token_metrics["token_precision"],
        "token_recall": token_metrics["token_recall"],
        "retrieval_success": retrieval_success,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "correct_source_rank": correct_source_rank,
        "mrr": mrr,
        "retrieval_precision": retrieval_precision,
        "retrieval_recall": retrieval_recall,
        "retrieval_f1": retrieval_f1,
        "retrieved_chunk_count": len(all_chunks),
        "retrieved_source_count": len(retrieved_sources),
        "retrieved_sources": retrieved_sources,
        "retrieved_categories": retrieved_categories,
        "retrieved_documents": retrieved_documents,
        "retrieved_sections": retrieved_sections,
        "retrieved_context": retrieved_context,
        "context_contains_answer": context_contains_answer,
        "reranker_top_score": reranker_top,
        "reranker_scores": pipeline_result.get("reranker_scores", []),
        "rewrite_latency": pipeline_result.get("rewrite_latency", 0.0),
        "decomposition_latency": pipeline_result.get("decomposition_latency", 0.0),
        "retrieval_latency": pipeline_result.get("retrieval_latency", 0.0),
        "reranker_latency": pipeline_result.get("reranker_latency", 0.0),
        "compression_latency": pipeline_result.get("compression_latency", 0.0),
        "generation_latency": pipeline_result.get("generation_latency", 0.0),
        "total_latency": pipeline_result.get("total_latency", 0.0),
        "hallucinated": hallucinated,
        "failure_reason": failure_reason,
    })


def run_evaluation() -> List[Dict[str, Any]]:
    """Run through the evaluation dataset and print a production summary report."""
    dataset = load_dataset()

    if not dataset:
        print("No questions were evaluated.")
        return []

    prewarm_models()

    results: List[Dict[str, Any]] = []

    for index, question_entry in enumerate(dataset, start=1):
        print("=" * 58)
        print(f"Question {index} / {len(dataset)}")
        print("=" * 58)
        print("Question:")
        print(question_entry.get("question", ""))
        print("\nExpected Answer:")
        print(question_entry.get("expected_answer", ""))
        print("\nRunning RAG pipeline...")

        result = evaluate_question(question_entry)
        results.append(result)

        print("\nGenerated Answer:")
        print(result["generated_answer"])
        print("\nRetrieved Sources:")
        print(result["retrieved_sources"])
        print()

    total_questions = len(results)

    # Core metrics calculation
    pipeline_successes = sum(
        1 for item in results
        if item["answer_correct"] and item["source_correct"] and item["retrieval_success"]
    )
    correct_answers = sum(1 for item in results if item["answer_correct"])
    exact_matches = sum(1 for item in results if item["exact_match"])
    correct_sources = sum(1 for item in results if item["source_correct"])
    retrieval_successes = sum(1 for item in results if item["retrieval_success"])
    context_recall_matches = sum(1 for item in results if item["context_contains_answer"])
    hallucination_count = sum(1 for item in results if item["hallucinated"])

    unsupported_results = [
        item for item in results if _is_unsupported_answer(item.get("expected_answer", ""))
    ]
    unsupported_correct = sum(1 for item in unsupported_results if item["answer_correct"])

    # Supported source metrics
    supported_source_items = [r for r in results if r.get("expected_source") is not None]
    tot_supported = len(supported_source_items)

    hit1_acc = round((sum(r["hit_at_1"] for r in supported_source_items if r["hit_at_1"] == 1) / tot_supported) * 100, 1) if tot_supported else 0.0
    hit3_acc = round((sum(r["hit_at_3"] for r in supported_source_items if r["hit_at_3"] == 1) / tot_supported) * 100, 1) if tot_supported else 0.0
    avg_mrr = round(sum(r["mrr"] for r in supported_source_items if r["mrr"] is not None) / tot_supported, 4) if tot_supported else 0.0

    pipeline_success_rate = round((pipeline_successes / total_questions) * 100, 1) if total_questions else 0.0
    semantic_answer_accuracy = round((correct_answers / total_questions) * 100, 1) if total_questions else 0.0
    exact_match_accuracy = round((exact_matches / total_questions) * 100, 1) if total_questions else 0.0
    source_accuracy = round((correct_sources / total_questions) * 100, 1) if total_questions else 0.0
    retrieval_accuracy = round((retrieval_successes / total_questions) * 100, 1) if total_questions else 0.0
    avg_faithfulness = round(sum(r["faithfulness"] for r in results) / total_questions, 4) if total_questions else 0.0
    avg_answer_relevancy = round(sum(r["answer_relevancy"] for r in results) / total_questions, 4) if total_questions else 0.0
    avg_context_precision = round(sum(r["context_precision"] for r in results) / total_questions, 4) if total_questions else 0.0
    context_recall = round((context_recall_matches / total_questions) * 100, 1) if total_questions else 0.0
    hallucination_rate = round((hallucination_count / total_questions) * 100, 1) if total_questions else 0.0
    unsupported_accuracy = round((unsupported_correct / len(unsupported_results)) * 100, 1) if unsupported_results else 0.0
    avg_confidence = round(sum(r["pipeline_confidence"] for r in results) / total_questions, 1) if total_questions else 0.0

    avg_retrieval_lat = round(sum(r["retrieval_latency"] for r in results) / total_questions, 2) if total_questions else 0.0
    avg_reranker_lat = round(sum(r["reranker_latency"] for r in results) / total_questions, 2) if total_questions else 0.0
    avg_generation_lat = round(sum(r["generation_latency"] for r in results) / total_questions, 2) if total_questions else 0.0
    avg_total_lat = round(sum(r["total_latency"] for r in results) / total_questions, 2) if total_questions else 0.0

    # Category Breakdown
    category_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        cat = item["category"]
        category_groups.setdefault(cat, []).append(item)

    category_accuracy = {}
    for cat, items in category_groups.items():
        cat_total = len(items)
        cat_correct = sum(1 for i in items if i["answer_correct"] and i["source_correct"])
        category_accuracy[cat] = {
            "total": cat_total,
            "correct": cat_correct,
            "accuracy": round((cat_correct / cat_total) * 100, 1) if cat_total else 0.0,
        }

    failure_counts = Counter(
        item["failure_reason"] for item in results if item.get("failure_reason") is not None
    )

    missed_documents = Counter(
        str(item["expected_source"]) for item in results if item.get("expected_source") and not item["source_correct"]
    )

    # Output JSON Files
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(_to_json_safe(results), handle, indent=2, ensure_ascii=False)

    summary = {
        "total_questions": total_questions,
        "pipeline_success_rate": pipeline_success_rate,
        "semantic_answer_accuracy": semantic_answer_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
        "source_accuracy": source_accuracy,
        "retrieval_accuracy": retrieval_accuracy,
        "hit_at_1": hit1_acc,
        "hit_at_3": hit3_acc,
        "mrr": avg_mrr,
        "faithfulness": avg_faithfulness,
        "answer_relevancy": avg_answer_relevancy,
        "context_precision": avg_context_precision,
        "context_recall": context_recall,
        "hallucination_rate": hallucination_rate,
        "unsupported_accuracy": unsupported_accuracy,
        "avg_pipeline_confidence": avg_confidence,
        "avg_retrieval_latency": avg_retrieval_lat,
        "avg_reranker_latency": avg_reranker_lat,
        "avg_generation_latency": avg_generation_lat,
        "avg_total_latency": avg_total_lat,
        "category_accuracy": category_accuracy,
        "top_failure_reasons": dict(failure_counts.most_common()),
        "missed_documents": dict(missed_documents.most_common()),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    # Failure Dashboard JSON
    valid_rerank_scores = [r["reranker_top_score"] for r in results if r.get("reranker_top_score") is not None]
    avg_reranker_score = round(sum(valid_rerank_scores) / len(valid_rerank_scores), 4) if valid_rerank_scores else 0.0

    histogram = {"0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
    for r in results:
        conf = r.get("pipeline_confidence", 0.0)
        if conf < 20:
            histogram["0-20"] += 1
        elif conf < 40:
            histogram["20-40"] += 1
        elif conf < 60:
            histogram["40-60"] += 1
        elif conf < 80:
            histogram["60-80"] += 1
        else:
            histogram["80-100"] += 1

    dashboard = {
        "top_failed_documents": dict(missed_documents.most_common()),
        "top_failed_categories": dict(
            sorted(category_accuracy.items(), key=lambda x: x[1]["accuracy"])
        ),
        "most_hallucinated_questions": [
            r["question"] for r in results if r.get("hallucinated")
        ],
        "average_reranker_score": avg_reranker_score,
        "average_retrieval_score": round(sum(1.0 for r in results if r["retrieval_success"]) / total_questions, 4),
        "most_common_failure_reason": failure_counts.most_common(1)[0][0] if failure_counts else None,
        "latency_distributions": {
            "retrieval": avg_retrieval_lat,
            "reranker": avg_reranker_lat,
            "generation": avg_generation_lat,
            "total": avg_total_lat,
        },
        "question_difficulty": [
            {"question": r["question"], "confidence": r["pipeline_confidence"], "correct": r["answer_correct"]}
            for r in sorted(results, key=lambda x: x["pipeline_confidence"])[:5]
        ],
        "confidence_histogram": histogram,
    }

    with DASHBOARD_PATH.open("w", encoding="utf-8") as handle:
        json.dump(dashboard, handle, indent=2, ensure_ascii=False)

    # Print Final Evaluation Report
    print("=" * 58)
    print("Production RAG Evaluation Report")
    print("=" * 58)
    print(f"Questions Evaluated       : {total_questions}")
    print(f"Pipeline Success Rate     : {pipeline_success_rate}%")
    print(f"Semantic Answer Accuracy  : {semantic_answer_accuracy}%")
    print(f"Exact Match Accuracy      : {exact_match_accuracy}%")
    print(f"Source Accuracy           : {source_accuracy}%")
    print(f"Retrieval Accuracy        : {retrieval_accuracy}%")
    print(f"Hit@1                     : {hit1_acc}%")
    print(f"Hit@3                     : {hit3_acc}%")
    print(f"MRR                       : {avg_mrr}")
    print(f"Faithfulness              : {avg_faithfulness}")
    print(f"Answer Relevancy          : {avg_answer_relevancy}")
    print(f"Context Precision         : {avg_context_precision}")
    print(f"Context Recall            : {context_recall}%")
    print(f"Hallucination Rate        : {hallucination_rate}%")
    print(f"Unsupported Accuracy      : {unsupported_accuracy}%")
    print(f"Average Pipeline Conf.   : {avg_confidence}%")
    print(f"Average Retrieval Latency : {avg_retrieval_lat:.2f} sec")
    print(f"Average Reranker Latency  : {avg_reranker_lat:.2f} sec")
    print(f"Average Generation Latency: {avg_generation_lat:.2f} sec")
    print(f"Average Total Latency     : {avg_total_lat:.2f} sec")
    print()

    print("=" * 58)
    print("Category-wise Accuracy")
    print("=" * 58)
    for cat, info in sorted(category_accuracy.items()):
        print(f"{cat:<24}: {info['accuracy']:>5.1f}% ({info['correct']}/{info['total']})")
    print()

    print("=" * 58)
    print("Top Failure Reasons")
    print("=" * 58)
    if failure_counts:
        for reason, count in failure_counts.most_common():
            print(f"{reason:<28}: {count}")
    else:
        print("No failures recorded!")
    print()

    print("=" * 58)
    print("Most Frequently Missed Documents")
    print("=" * 58)
    if missed_documents:
        for doc, count in missed_documents.most_common():
            print(f"{doc:<28}: {count}")
    else:
        print("None! All expected sources were successfully retrieved.")
    print()

    # Failed Question Analysis
    failed_questions = [item for item in results if item.get("failure_reason") is not None]

    if failed_questions:
        print("=" * 58)
        print("Failed Question Analysis")
        print("=" * 58)

        for item in failed_questions:
            print(f"Question            : {item['question']}")
            print(f"Category            : {item['category']}")
            print(f"Expected Answer     : {item['expected_answer']}")
            print(f"Generated Answer    : {item['generated_answer']}")
            print(f"Semantic Similarity : {item['semantic_similarity']}")
            print(f"Faithfulness        : {item['faithfulness']}")
            print(f"Answer Relevancy    : {item['answer_relevancy']}")
            print(f"Context Recall      : {item['context_recall']}")
            print(f"Expected Source     : {item['expected_source']}")
            print(f"Retrieved Sources   : {item['retrieved_sources']}")
            print(f"Reranker Top Score  : {item['reranker_top_score']}")
            print(f"Pipeline Confidence : {item['pipeline_confidence']}%")
            print(f"Failure Reason      : {item['failure_reason']}")
            print("-" * 58)

    return results


def main() -> None:
    """Entry point for the evaluation runner."""
    run_evaluation()


if __name__ == "__main__":
    main()
