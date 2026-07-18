from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

VECTOR_STORE_DIR = BASE_DIR / "vector_store"

MODELS_DIR = BASE_DIR / "models"

# Create required directories if they don't exist
VECTOR_STORE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# -----------------------------
# FAISS Files
# -----------------------------

FAISS_INDEX = VECTOR_STORE_DIR / "faiss.index"

CHUNKS_FILE = VECTOR_STORE_DIR / "chunks.pkl"

METADATA_FILE = VECTOR_STORE_DIR / "metadata.pkl"

HASHES_FILE = VECTOR_STORE_DIR / "hashes.pkl"

# -----------------------------
# Models
# -----------------------------

LLM_MODEL = "qwen2.5:1.5b"

EMBED_MODEL = "nomic-embed-text"

# -----------------------------
# Chunking
# -----------------------------

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100

# -----------------------------
# Retrieval
# -----------------------------

TOP_K = 5

MIN_SIMILARITY = 0.60