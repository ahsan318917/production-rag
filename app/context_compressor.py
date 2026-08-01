import re

from typing import List, Dict, Any

from config import (
    CONTEXT_COMPRESSION_ENABLED,
    COMPRESSION_MIN_SCORE,
    COMPRESSION_TOP_SENTENCES,
    CONTEXT_WINDOW_SIZE,
    MIN_UNITS_FOR_COMPRESSION,
)

from reranker import score_pairs

try:
    from langchain_core.documents import Document
except ImportError:
    Document = None


PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
STRUCTURED_BLOCK_THRESHOLD = 0.6
SEMANTIC_BLOCK_HINTS = (
    "profit",
    "rate",
    "interest",
    "eligibility",
    "deposit",
    "amount",
    "tenure",
    "minimum",
    "maximum",
    "fee",
    "charge",
    "premature",
    "encash",
    "nomination",
    "salary",
    "income",
)


def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not normalized:
        return []

    fragments = SENTENCE_SPLIT_PATTERN.split(normalized)

    sentences = [fragment.strip() for fragment in fragments if fragment and fragment.strip()]

    return sentences


def _is_structured_block(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]

    if len(lines) <= 1:
        return False

    structured_lines = [
        line for line in lines
        if len(line) < 120 and not re.search(r"[.!?]$", line)
    ]

    return len(structured_lines) / len(lines) >= STRUCTURED_BLOCK_THRESHOLD


def _is_semantically_related(unit: str, query: str) -> bool:
    if not unit or not query:
        return True

    normalized_query = query.lower()
    normalized_unit = unit.lower()

    if any(hint in normalized_query for hint in ("profit", "rate", "interest")):
        return any(hint in normalized_unit for hint in ("profit", "rate", "interest", "percent", "%", "deposit", "tenure"))

    return True


def split_into_units(text: str) -> List[str]:
    if not text:
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not normalized:
        return []

    paragraphs = [paragraph.strip() for paragraph in PARAGRAPH_SPLIT_PATTERN.split(normalized) if paragraph.strip()]

    units: List[str] = []

    for paragraph in paragraphs:
        if _is_structured_block(paragraph):
            units.append(paragraph)
            continue

        sentences = split_into_sentences(paragraph)

        if len(sentences) <= 1:
            units.append(paragraph)
        else:
            units.extend(sentences)

    return units


def _merge_windows(selected_indices: List[int], window_size: int, unit_count: int) -> List[tuple]:
    if not selected_indices:
        return []

    intervals = []

    for idx in sorted(selected_indices):
        start = max(0, idx - window_size)
        end = min(unit_count - 1, idx + window_size)
        intervals.append((start, end))

    merged = []

    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    return merged


def _extract_chunk_text(chunk: Any) -> str:
    if chunk is None:
        return ""

    if isinstance(chunk, dict):
        for key in ("page_content", "text", "content"):
            value = chunk.get(key)
            if value:
                return str(value)

    if hasattr(chunk, "page_content") and getattr(chunk, "page_content") is not None:
        return str(chunk.page_content)

    if hasattr(chunk, "content") and getattr(chunk, "content") is not None:
        return str(chunk.content)

    try:
        return str(chunk)
    except Exception:
        return ""


def score_sentences(query: str, sentences: List[str]) -> List[float]:
    if not query or not sentences:
        return []

    return score_pairs(query, sentences)


def _build_compressed_chunk(original_chunk: Any, compressed_text: str):
    if not compressed_text:
        return original_chunk

    metadata = {}

    if hasattr(original_chunk, "metadata"):
        metadata = getattr(original_chunk, "metadata") or {}
    elif isinstance(original_chunk, dict):
        metadata = original_chunk.get("metadata", {}) or {}

    if Document is not None:
        try:
            return Document(
                page_content=compressed_text,
                metadata=metadata,
            )
        except Exception:
            pass

    return {
        "page_content": compressed_text,
        "metadata": metadata,
    }


def compress_chunk(query: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
    if not CONTEXT_COMPRESSION_ENABLED:
        return chunk

    if not chunk or "chunk" not in chunk:
        return chunk

    original_chunk = chunk["chunk"]
    text = _extract_chunk_text(original_chunk)
    units = split_into_units(text)

    if len(units) <= 1:
        return chunk

    if len(units) < MIN_UNITS_FOR_COMPRESSION:
        print("Compression skipped: chunk is already small.")
        return chunk

    try:
        scores = score_sentences(query, units)

        indexed_scores = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        selected_indices = [index for index, score in indexed_scores if score >= COMPRESSION_MIN_SCORE]

        if not selected_indices and indexed_scores:
            selected_indices = [indexed_scores[0][0]]

        selected_indices = selected_indices[:COMPRESSION_TOP_SENTENCES]
        selected_indices = sorted(dict.fromkeys(selected_indices))

        filtered_indices = [
            index for index in selected_indices
            if _is_semantically_related(units[index], query)
        ]

        if not filtered_indices:
            return chunk

        windows = _merge_windows(filtered_indices, CONTEXT_WINDOW_SIZE, len(units))

        kept_units = []
        seen_units = set()
        for start, end in windows:
            for offset in range(start, end + 1):
                unit = units[offset]
                if unit in seen_units:
                    continue
                seen_units.add(unit)
                kept_units.append(unit)

        compressed_text = "\n\n".join(kept_units).strip()

        if not compressed_text or len(kept_units) == 0:
            return chunk

        compressed_chunk = _build_compressed_chunk(original_chunk, compressed_text)

        original_count = len(units)
        kept_count = len(kept_units)
        ratio = 100.0 - (kept_count / original_count * 100.0) if original_count else 0.0

        print("----------------------------------------")
        print("Compressing Context")
        print(f"Chunk {chunk.get('vector_id', 'unknown')}")
        print("Compression skipped: False")

        for index in filtered_indices:
            print(f"Selected Block: {units[index]}")
            print(f"Relevance Score: {scores[index]:.2f}")
            print("Reason: Highest semantic similarity")

        print("\nNeighbor Blocks Included:")
        if windows:
            for start, end in windows:
                window_text = "\n\n".join(units[start:end + 1]).strip()
                print(window_text)
                if end != windows[-1][1]:
                    print("\n...")
        else:
            print("None")

        print(f"Original Units    : {original_count}")
        print(f"Kept Units        : {kept_count}")
        print(f"Compression Ratio : {ratio:.0f}%")
        print("\nFinal Context:")
        print(compressed_text)
        print("----------------------------------------")

        compressed_result = dict(chunk)
        compressed_result["chunk"] = compressed_chunk

        return compressed_result

    except Exception as exc:
        print("Context compression failed. Using original chunk.")
        print(exc)
        return chunk


def compress_context(query: str, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not CONTEXT_COMPRESSION_ENABLED:
        return retrieved_chunks

    if not retrieved_chunks:
        return retrieved_chunks

    print("\nCompressing retrieved context...\n")

    compressed = []
    total_original = 0
    total_kept = 0

    for chunk_result in retrieved_chunks:
        compressed_result = compress_chunk(query, chunk_result)

        original_text = _extract_chunk_text(chunk_result.get("chunk"))
        compressed_text = _extract_chunk_text(compressed_result.get("chunk"))

        original_sentences = len(split_into_sentences(original_text))
        kept_sentences = len(split_into_sentences(compressed_text))

        total_original += original_sentences
        total_kept += kept_sentences

        compressed.append(compressed_result)

    print("========================================")
    print(f"Total Original Sentences : {total_original}")
    print(f"Total Kept Sentences     : {total_kept}")
    print("========================================")

    return compressed
