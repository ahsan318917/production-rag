import os
import pickle

import faiss
import numpy as np

from config import (
    FAISS_INDEX,
    CHUNKS_FILE,
    METADATA_FILE,
    HASHES_FILE,
)


# -------------------------------------------------
# Create FAISS Index
# -------------------------------------------------

def create_index(embeddings):
    """
    Create a FAISS index from embeddings.
    """

    vectors = np.array(
    embeddings,
    dtype=np.float32
)

    # Normalize to unit length
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    return index


# -------------------------------------------------
# Save / Load FAISS Index
# -------------------------------------------------

def save_index(index):
    """
    Save FAISS index to disk.
    """

    faiss.write_index(
        index,
        str(FAISS_INDEX)
    )


def load_index():
    """
    Load FAISS index from disk.
    """

    if not FAISS_INDEX.exists():
        raise FileNotFoundError("FAISS index not found.")

    return faiss.read_index(str(FAISS_INDEX))


# -------------------------------------------------
# Save / Load Chunks
# -------------------------------------------------

def save_chunks(chunks):
    """
    Save LangChain document chunks.
    """

    with open(CHUNKS_FILE, "wb") as file:
        pickle.dump(chunks, file)


def load_chunks():
    """
    Load LangChain document chunks.
    """

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("Chunks file not found.")

    with open(CHUNKS_FILE, "rb") as file:
        return pickle.load(file)


# -------------------------------------------------
# Save / Load Metadata
# -------------------------------------------------

def save_metadata(metadata):
    """
    Save vector store metadata.
    """

    with open(METADATA_FILE, "wb") as file:
        pickle.dump(metadata, file)


def load_metadata():
    """
    Load vector store metadata.
    """

    if not METADATA_FILE.exists():
        raise FileNotFoundError("Metadata file not found.")

    with open(METADATA_FILE, "rb") as file:
        return pickle.load(file)


# -------------------------------------------------
# Save / Load Document Hashes
# -------------------------------------------------

def save_hashes(hashes):
    """
    Save document hashes.
    """

    with open(HASHES_FILE, "wb") as file:
        pickle.dump(hashes, file)


def load_hashes():
    """
    Load document hashes.
    """

    if not HASHES_FILE.exists():
        raise FileNotFoundError("Hashes file not found.")

    with open(HASHES_FILE, "rb") as file:
        return pickle.load(file)


# -------------------------------------------------
# Utility Functions
# -------------------------------------------------

def index_exists():
    """
    Check whether a complete vector store exists.
    """

    return (
        FAISS_INDEX.exists()
        and CHUNKS_FILE.exists()
        and METADATA_FILE.exists()
    )


def delete_index():
    """
    Delete the entire vector store.
    """

    files = [
        FAISS_INDEX,
        CHUNKS_FILE,
        METADATA_FILE,
        HASHES_FILE,
    ]

    for file in files:

        if file.exists():
            file.unlink()

    print("Vector store deleted successfully.")