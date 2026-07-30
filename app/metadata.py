import re
from pathlib import Path


def infer_category(filename):
    """Infer a simple category from the filename."""

    if not filename:
        return "unknown"

    name = Path(str(filename)).name.lower()

    if "loan" in name:
        return "loan"
    if "atm" in name:
        return "atm"
    if "card" in name:
        return "card"
    if "account" in name:
        return "account"
    if "policy" in name:
        return "policy"

    return "unknown"


def infer_document_name(filename):
    """Infer a readable document label from the filename."""

    if not filename:
        return "Unknown Document"

    name = Path(str(filename)).stem
    name = re.sub(r"[_-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    if not name:
        return "Unknown Document"

    return name.title()


def infer_section(chunk):
    """Infer a heading/section from chunk text using a simple heuristic."""

    if chunk is None:
        return "Unknown"

    text = getattr(chunk, "page_content", "") or ""

    if not text:
        return "Unknown"

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        if len(line) <= 80 and not any(char in line for char in ".:!?"):
            if line.isupper() or line.istitle() or line.startswith(("Eligibility", "Early", "Requirements", "Documents", "Interest")):
                return line

    return "Unknown"


def apply_metadata(chunk, filename):
    """Attach metadata to a chunk without overwriting existing values."""

    if chunk is None:
        return chunk

    metadata = getattr(chunk, "metadata", None)

    if metadata is None:
        metadata = {}
        chunk.metadata = metadata

    metadata.setdefault("source", str(filename))
    metadata.setdefault("document", infer_document_name(filename))
    metadata.setdefault("category", infer_category(filename))
    metadata.setdefault("section", infer_section(chunk))

    return chunk
