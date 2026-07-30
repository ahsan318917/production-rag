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
# Create Empty Index
# -------------------------------------------------

def create_index(dimension: int):
    """
    Create an empty FAISS IndexIDMap2 using
    cosine similarity (Inner Product).
    """

    base_index = faiss.IndexFlatIP(dimension)

    index = faiss.IndexIDMap2(base_index)

    return index


# -------------------------------------------------
# Add Vectors
# -------------------------------------------------

def add_vectors(index, embeddings, vector_ids):
    """
    Add vectors with explicit IDs.
    """

    vectors = np.asarray(
        embeddings,
        dtype=np.float32
    )

    faiss.normalize_L2(vectors)

    ids = np.asarray(
        vector_ids,
        dtype=np.int64
    )

    index.add_with_ids(
        vectors,
        ids
    )

    return index


# -------------------------------------------------
# Remove Vectors
# -------------------------------------------------

def remove_vectors(index, vector_ids):
    """
    Remove vectors by ID.
    """

    if len(vector_ids) == 0:
        return index

    ids = np.asarray(
        vector_ids,
        dtype=np.int64
    )

    index.remove_ids(ids)

    return index


# -------------------------------------------------
# Search Index
# -------------------------------------------------

def search(index, query_embedding, top_k):
    """
    Search using cosine similarity.
    Returns:
        scores,
        vector_ids
    """

    query = np.asarray(
        [query_embedding],
        dtype=np.float32
    )

    faiss.normalize_L2(query)

    scores, ids = index.search(
        query,
        top_k
    )

    return scores[0], ids[0]


# -------------------------------------------------
# Save / Load Index
# -------------------------------------------------

def save_index(index):

    faiss.write_index(
        index,
        str(FAISS_INDEX)
    )


def load_index():

    if not FAISS_INDEX.exists():

        raise FileNotFoundError(
            "FAISS index not found."
        )

    return faiss.read_index(
        str(FAISS_INDEX)
    )


# -------------------------------------------------
# Vector Utilities
# -------------------------------------------------

def vector_count(index):
    """
    Number of vectors stored.
    """

    return index.ntotal


def allocate_vector_ids(metadata, count):
    """
    Allocate unique vector IDs.

    Updates metadata["next_vector_id"].
    """

    start = metadata["next_vector_id"]

    ids = list(
        range(
            start,
            start + count
        )
    )

    metadata["next_vector_id"] += count

    return ids


# -------------------------------------------------
# Save / Load Chunks
# -------------------------------------------------

def save_chunks(chunks):

    with open(CHUNKS_FILE, "wb") as f:

        pickle.dump(chunks, f)


def load_chunks():

    if not CHUNKS_FILE.exists():

        return []

    with open(CHUNKS_FILE, "rb") as f:

        return pickle.load(f)


# -------------------------------------------------
# Save / Load Metadata
# -------------------------------------------------

def save_metadata(metadata):

    with open(METADATA_FILE, "wb") as f:

        pickle.dump(metadata, f)


def load_metadata():

    if not METADATA_FILE.exists():

        return {

            "embedding_model": None,
            "dimension": None,
            "num_vectors": 0,
            "next_vector_id": 0,
            "created_at": None,
            "updated_at": None,

        }

    with open(METADATA_FILE, "rb") as f:

        return pickle.load(f)


# -------------------------------------------------
# Save / Load Hashes
# -------------------------------------------------

def save_hashes(hashes):

    with open(HASHES_FILE, "wb") as f:

        pickle.dump(hashes, f)


def load_hashes():

    if not HASHES_FILE.exists():

        return {}

    with open(HASHES_FILE, "rb") as f:

        return pickle.load(f)


# -------------------------------------------------
# Utility Functions
# -------------------------------------------------

def index_exists():
    """
    Check whether the FAISS index exists.
    """

    return FAISS_INDEX.exists()


def delete_index():
    """
    Delete the entire vector store.
    """

    for file in (
        FAISS_INDEX,
        CHUNKS_FILE,
        METADATA_FILE,
        HASHES_FILE,
    ):

        if file.exists():

            file.unlink()

    print("Vector store deleted successfully.")