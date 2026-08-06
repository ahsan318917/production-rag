import re
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - runtime fallback
    SentenceTransformer = None

SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
_semantic_model: Optional[Any] = None

REFUSAL_PHRASES = [
    "i don't have enough information",
    "i don t have enough information",
    "i cannot answer",
    "not available in the provided documents",
    "information not found",
    "i don't have enough information to answer that",
]


def normalize_text(text: str) -> str:
    """Normalize text for basic string comparisons and tokenization."""
    if not text:
        return ""

    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9.%]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_unsupported_refusal(text: str) -> bool:
    """Check whether a text represents a standard acceptable refusal response."""
    if not text:
        return False

    norm = normalize_text(text)
    if not norm:
        return False

    for phrase in REFUSAL_PHRASES:
        if phrase in norm or norm in phrase:
            return True

    return False


def get_semantic_model():
    """Lazy-load the evaluation embedding model once globally."""
    global _semantic_model

    if _semantic_model is not None:
        return _semantic_model

    if SentenceTransformer is None:
        return None

    try:
        _semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    except Exception:
        _semantic_model = None

    return _semantic_model


@lru_cache(maxsize=1024)
def _get_embedding(text: str) -> Optional[Tuple[float, ...]]:
    """Cached single string embedding generation."""
    model = get_semantic_model()
    if model is None or not text:
        return None

    try:
        emb = model.encode(text, convert_to_tensor=False, normalize_embeddings=True)
        return tuple(float(x) for x in emb)
    except Exception:
        return None


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two text snippets using SentenceTransformers.
    Uses LRU caching and falls back to SequenceMatcher ratio.
    """
    if not text1 or not text2:
        return 0.0

    t1_norm = normalize_text(text1)
    t2_norm = normalize_text(text2)

    if not t1_norm or not t2_norm:
        return 0.0

    if t1_norm == t2_norm:
        return 1.0

    emb1 = _get_embedding(text1)
    emb2 = _get_embedding(text2)

    if emb1 is not None and emb2 is not None:
        try:
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = sum(a * a for a in emb1) ** 0.5
            norm2 = sum(b * b for b in emb2) ** 0.5

            if norm1 > 0 and norm2 > 0:
                sim = dot_product / (norm1 * norm2)
                return round(max(0.0, min(1.0, float(sim))), 4)
        except Exception:
            pass

    return round(SequenceMatcher(None, t1_norm, t2_norm).ratio(), 4)


def extract_numeric_facts(text: str) -> Set[str]:
    """Extract numeric values, percentages, amounts, and durations from text."""
    if not text:
        return set()

    normalized = text.lower().replace(",", "")
    facts = set()

    # Percentages
    for match in re.findall(r"\b\d+(?:\.\d+)?\s*%\b|\b\d+(?:\.\d+)?\s*percent\b", normalized):
        num = re.search(r"\d+(?:\.\d+)?", match)
        if num:
            facts.add(f"{float(num.group()):g}%")

    # Currencies / Amounts
    for match in re.findall(r"\b\d+(?:\.\d+)?\s*pkr\b|\b\d+(?:\.\d+)?\s*rupees\b", normalized):
        num = re.search(r"\d+(?:\.\d+)?", match)
        if num:
            facts.add(f"{float(num.group()):g}pkr")

    # Durations / Ages
    for match in re.findall(r"\b\d+\s*(?:years?|months?|days?|hours?)\b", normalized):
        num = re.search(r"\d+", match)
        unit = re.search(r"years?|months?|days?|hours?", match)
        if num and unit:
            u = unit.group()[0]  # y, m, d, h
            facts.add(f"{num.group()}{u}")

    # Plain standalone numbers
    for match in re.findall(r"\b\d+(?:\.\d+)?\b", normalized):
        val = float(match)
        facts.add(f"{val:g}")

    return facts


def compute_numeric_match(generated_answer: str, expected_answer: str) -> bool:
    """Return True if all numeric facts in expected_answer are present in generated_answer."""
    exp_facts = extract_numeric_facts(expected_answer)
    if not exp_facts:
        return False

    gen_facts = extract_numeric_facts(generated_answer)

    # Check if all expected facts exist in generated facts
    for fact in exp_facts:
        if fact not in gen_facts:
            # Flexible check for percentage / number
            val = fact.rstrip("%").rstrip("pkr").rstrip("y").rstrip("m").rstrip("d").rstrip("h")
            if not any(val in gf for gf in gen_facts):
                return False

    return True


def extract_key_entities(text: str) -> Set[str]:
    """Extract key domain entities from text."""
    if not text:
        return set()

    text_lower = text.lower()
    entities = set()

    keywords = [
        "cnic", "otp", "classic", "gold", "platinum", "beneficiary",
        "biometric", "helpline", "email", "24/7", "early withdrawal",
        "credit card", "fixed deposit", "internet banking", "loan"
    ]

    for kw in keywords:
        if kw in text_lower:
            entities.add(kw)

    return entities


def compute_entity_match(generated_answer: str, expected_answer: str) -> bool:
    """Return True if key entities in expected_answer match generated_answer."""
    exp_entities = extract_key_entities(expected_answer)
    if not exp_entities:
        return True

    gen_entities = extract_key_entities(generated_answer)
    return exp_entities.issubset(gen_entities) or len(exp_entities & gen_entities) > 0


def compute_bleu(generated_answer: str, expected_answer: str) -> float:
    """Compute 1-gram & 2-gram sentence BLEU score with brevity penalty."""
    gen_tokens = normalize_text(generated_answer).split()
    exp_tokens = normalize_text(expected_answer).split()

    if not gen_tokens or not exp_tokens:
        return 0.0

    # 1-gram precision
    gen_1grams = gen_tokens
    exp_1grams = exp_tokens
    common_1grams = sum((Counter(gen_1grams) & Counter(exp_1grams)).values())
    p1 = common_1grams / len(gen_1grams) if gen_1grams else 0.0

    # 2-gram precision
    gen_2grams = [f"{gen_tokens[i]} {gen_tokens[i+1]}" for i in range(len(gen_tokens)-1)]
    exp_2grams = [f"{exp_tokens[i]} {exp_tokens[i+1]}" for i in range(len(exp_tokens)-1)]
    if gen_2grams and exp_2grams:
        common_2grams = sum((Counter(gen_2grams) & Counter(exp_2grams)).values())
        p2 = common_2grams / len(gen_2grams)
    else:
        p2 = p1

    if p1 == 0.0:
        return 0.0

    precision = (p1 * (p2 if p2 > 0 else p1)) ** 0.5

    # Brevity Penalty
    r = len(exp_tokens)
    c = len(gen_tokens)
    bp = 1.0 if c > r else (exp(1 - r / c) if c > 0 else 0.0)

    return round(max(0.0, min(1.0, float(bp * precision))), 4)


def exp(x: float) -> float:
    import math
    return math.exp(x)


def compute_rouge_l(generated_answer: str, expected_answer: str) -> float:
    """Compute ROUGE-L (Longest Common Subsequence ratio)."""
    gen_tokens = normalize_text(generated_answer).split()
    exp_tokens = normalize_text(expected_answer).split()

    if not gen_tokens or not exp_tokens:
        return 0.0

    m, n = len(gen_tokens), len(exp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if gen_tokens[i] == exp_tokens[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])

    lcs = dp[m][n]
    return round(float(lcs / n), 4)


def compute_token_metrics(generated_answer: str, expected_answer: str) -> Dict[str, Any]:
    """Compute Exact Match, Token Precision, Token Recall, and Token F1."""
    gen_norm = normalize_text(generated_answer)
    exp_norm = normalize_text(expected_answer)

    exact_match = (gen_norm == exp_norm) and bool(gen_norm)

    gen_tokens = gen_norm.split() if gen_norm else []
    exp_tokens = exp_norm.split() if exp_norm else []

    if not gen_tokens or not exp_tokens:
        return {
            "exact_match": exact_match,
            "token_precision": 1.0 if gen_tokens == exp_tokens else 0.0,
            "token_recall": 1.0 if gen_tokens == exp_tokens else 0.0,
            "token_f1": 1.0 if gen_tokens == exp_tokens else 0.0,
        }

    common_tokens = Counter(gen_tokens) & Counter(exp_tokens)
    num_common = sum(common_tokens.values())

    precision = num_common / len(gen_tokens) if gen_tokens else 0.0
    recall = num_common / len(exp_tokens) if exp_tokens else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        "exact_match": exact_match,
        "token_precision": round(precision, 4),
        "token_recall": round(recall, 4),
        "token_f1": round(f1, 4),
    }


def compute_context_recall(expected_answer: str, generated_answer: str, context_text: str) -> float:
    """
    Compute sentence-level Context Recall by taking max similarity across
    both expected_answer and generated_answer against context sentences.
    """
    if not context_text:
        return 0.0

    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", context_text)
    sentences = [s.strip() for s in raw_sentences if s and len(normalize_text(s)) > 3]

    if not sentences:
        return 0.0

    max_sim = 0.0

    for sentence in sentences:
        if expected_answer:
            sim_exp = compute_semantic_similarity(expected_answer, sentence)
            if sim_exp > max_sim:
                max_sim = sim_exp

        if generated_answer and not is_unsupported_refusal(generated_answer):
            sim_gen = compute_semantic_similarity(generated_answer, sentence)
            if sim_gen > max_sim:
                max_sim = sim_gen

    return round(max_sim, 4)


def compute_faithfulness(generated_answer: str, context_text: str) -> float:
    """
    Compute Faithfulness (DeepEval/RAGAS style).
    Verifies that claims in generated_answer are supported by retrieved context.
    """
    if not generated_answer or is_unsupported_refusal(generated_answer):
        return 1.0

    if not context_text:
        return 0.0

    raw_claims = re.split(r"(?<=[.!?])\s+|\n+", generated_answer)
    claims = [c.strip() for c in raw_claims if c and len(normalize_text(c)) > 3]

    if not claims:
        return 1.0

    supported = 0

    for claim in claims:
        sim = compute_context_recall(claim, "", context_text)
        num_facts = extract_numeric_facts(claim)
        ctx_facts = extract_numeric_facts(context_text)

        # Claim is supported if semantic similarity >= 0.50 and all numeric facts exist in context
        facts_ok = num_facts.issubset(ctx_facts) if num_facts else True

        if sim >= 0.50 and facts_ok:
            supported += 1

    return round(float(supported / len(claims)), 4)


def compute_context_precision(question: str, final_chunks: List[Any]) -> float:
    """
    Compute Context Precision: rank-weighted proportion of useful retrieved chunks.
    """
    if not question or not final_chunks:
        return 0.0

    relevant_count = 0
    weighted_precision = 0.0

    for k, item in enumerate(final_chunks, start=1):
        chunk = item.get("chunk") if isinstance(item, dict) else item
        text = getattr(chunk, "page_content", str(chunk)) or ""

        sim = compute_semantic_similarity(question, text)
        is_relevant = sim >= 0.40

        if is_relevant:
            relevant_count += 1
            precision_at_k = relevant_count / k
            weighted_precision += precision_at_k

    if relevant_count == 0:
        return 0.0

    return round(float(weighted_precision / relevant_count), 4)


def compute_answer_relevancy(question: str, generated_answer: str) -> float:
    """Compute Answer Relevancy between question and generated_answer."""
    if not question or not generated_answer:
        return 0.0

    if is_unsupported_refusal(generated_answer):
        return 1.0

    return compute_semantic_similarity(question, generated_answer)
