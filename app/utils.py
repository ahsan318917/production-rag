import os
import pickle
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from config import CHUNK_SIZE, CHUNK_OVERLAP


from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents):
    """
    Split LangChain Document objects into overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    return chunks


def compute_file_hash(file_path):
    """
    Compute SHA256 hash of a file.
    Used to detect whether a document has changed.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(4096)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def save_pickle(data, file_path):
    """
    Save Python object to pickle.
    """

    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(file_path):
    """
    Load pickle file.
    Returns None if file doesn't exist.
    """

    if not os.path.exists(file_path):
        return None

    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_faiss(index, index_path):
    """
    Save FAISS index.
    """

    faiss.write_index(index, str(index_path))


def load_faiss(index_path):
    """
    Load FAISS index.
    Returns None if file doesn't exist.
    """

    if not os.path.exists(index_path):
        return None

    return faiss.read_index(str(index_path))