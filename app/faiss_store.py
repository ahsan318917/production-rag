import pickle

import faiss
import numpy as np

from config import VECTOR_STORE_DIR


INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.pkl"


def create_index(embeddings):
    """
    Create a FAISS index from embeddings.
    """

    vectors = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    return index


def save_index(index):
    """
    Save FAISS index to disk.
    """

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )


def load_index():
    """
    Load FAISS index from disk.
    """

    return faiss.read_index(str(INDEX_PATH))


def save_chunks(chunks):
    """
    Save LangChain chunks.
    """

    with open(CHUNKS_PATH, "wb") as file:

        pickle.dump(chunks, file)


def load_chunks():
    """
    Load LangChain chunks.
    """

    with open(CHUNKS_PATH, "rb") as file:

        return pickle.load(file)