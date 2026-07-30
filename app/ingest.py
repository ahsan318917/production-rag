from pathlib import Path
from datetime import datetime
import traceback

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from config import (
    DATA_DIR,
    EMBED_MODEL,
)

from utils import chunk_documents
from embeddings import generate_embeddings

from faiss_store import (
    create_index,
    load_index,
    save_index,
    add_vectors,
    remove_vectors,
    allocate_vector_ids,
    index_exists,
    load_chunks,
    save_chunks,
    load_metadata,
    save_metadata,
    load_hashes,
    save_hashes,
)

from hash import calculate_file_hash
from metadata import apply_metadata

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
}
def find_documents():
    """
    Find every supported document inside DATA_DIR.
    """

    documents = []

    for file in DATA_DIR.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            documents.append(file)

    return sorted(documents)
def load_document(file_path: Path):
    """
    Load a document using the appropriate LangChain loader.
    """

    suffix = file_path.suffix.lower()

    if suffix == ".txt":

        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

    elif suffix == ".pdf":

        loader = PyPDFLoader(
            str(file_path)
        )

    elif suffix == ".docx":

        loader = Docx2txtLoader(
            str(file_path)
        )

    else:

        raise ValueError(
            f"Unsupported file type: {suffix}"
        )

    return loader.load()

def load_or_create_vector_store():
    """
    Load an existing vector store.
    If it doesn't exist, create a new one.
    """

    if index_exists():

        print("Loading existing vector store...")

        index = load_index()

        chunks = load_chunks()

        if isinstance(chunks, (list, tuple)):

            chunks = {

                i: chunk

                for i, chunk in enumerate(chunks)

            }

        metadata = load_metadata()

        hashes = load_hashes()

        hashes = {

            str(path): normalize_hash_entry(entry)

            for path, entry in hashes.items()

        }

        metadata = normalize_metadata(metadata, chunks, hashes)

        if index.__class__.__name__ != "IndexIDMap2":

            print("Existing FAISS index is not IndexIDMap2; rebuilding index...")

            if chunks:

                chunk_list = [chunks[i] for i in sorted(chunks.keys())]

                embeddings = embed_chunks(chunk_list)

                index = create_index(

                    len(embeddings[0])

                )

                vector_ids = sorted(chunks.keys())

                add_vectors(

                    index,

                    embeddings,

                    vector_ids,

                )

            else:

                index = None

    else:

        print("Creating new vector store...")

        metadata = {

            "embedding_model": EMBED_MODEL,

            "dimension": None,

            "num_vectors": 0,

            "next_vector_id": 0,

            "created_at": datetime.now().isoformat(),

            "updated_at": datetime.now().isoformat()

        }

        index = None

        chunks = {}

        hashes = {}

    return (
        index,
        chunks,
        metadata,
        hashes
    )
def build_chunks(file_path):
    """
    Load a document and split it into chunks.
    """

    docs = load_document(file_path)

    chunks = chunk_documents(docs)

    return chunks
def embed_chunks(chunks):
    """
    Generate embeddings for a list of chunks.
    """

    texts = [

        chunk.page_content

        for chunk in chunks

    ]

    embeddings = generate_embeddings(texts)

    return embeddings

def ensure_index(index, metadata, embeddings):
    """
    Create the FAISS index if this is the first ingestion.
    """

    if index is not None:

        return index

    dimension = len(embeddings[0])

    metadata["dimension"] = dimension

    index = create_index(dimension)

    return index


def normalize_hash_entry(hash_entry):
    """
    Normalize legacy hash entries stored as plain strings.
    """

    if isinstance(hash_entry, dict):

        return hash_entry

    return {

        "hash": hash_entry,

        "vector_ids": [],

        "chunk_count": 0,

        "updated_at": None,

    }


def normalize_metadata(metadata, chunks, hashes=None):
    """
    Ensure required metadata keys exist and derive missing bookkeeping values.
    """

    defaults = {

        "embedding_model": EMBED_MODEL,

        "dimension": None,

        "num_vectors": 0,

        "next_vector_id": 0,

        "created_at": datetime.now().isoformat(),

        "updated_at": datetime.now().isoformat(),

    }

    for key, value in defaults.items():

        metadata.setdefault(key, value)

    if isinstance(chunks, dict):

        metadata["num_vectors"] = len(chunks)

        if chunks:

            max_id = max(chunks.keys())

            metadata["next_vector_id"] = max(

                metadata.get("next_vector_id", 0),

                max_id + 1,

            )

        else:

            metadata.setdefault("next_vector_id", 0)

    elif isinstance(chunks, (list, tuple)):

        metadata["num_vectors"] = len(chunks)

        metadata["next_vector_id"] = max(

            metadata.get("next_vector_id", 0),

            len(chunks),

        )

    else:

        metadata.setdefault("num_vectors", 0)

        metadata.setdefault("next_vector_id", 0)

    if hashes is not None:

        max_vector_id = 0

        for entry in hashes.values():

            if isinstance(entry, dict):

                vector_ids = entry.get("vector_ids", [])

                if vector_ids:

                    max_vector_id = max(

                        max_vector_id,

                        max(vector_ids) + 1,

                    )

        if max_vector_id:

            metadata["next_vector_id"] = max(

                metadata.get("next_vector_id", 0),

                max_vector_id,

            )

    if metadata.get("next_vector_id") is None:

        metadata["next_vector_id"] = 0

    return metadata

def save_vector_store(
    index,
    chunks,
    metadata,
    hashes
):
    """
    Save every vector store component.
    """

    metadata["num_vectors"] = len(chunks)

    metadata["updated_at"] = datetime.now().isoformat()

    save_index(index)

    save_chunks(chunks)

    save_metadata(metadata)

    save_hashes(hashes)


def process_new_document(
    file_path,
    index,
    chunks,
    metadata,
    hashes,
    current_hash,
):
    """
    Ingest a brand-new document.
    """

    print(f"\nNew document detected: {file_path.name}")

    # ---------------------------------
    # Chunk document
    # ---------------------------------

    document_chunks = build_chunks(file_path)

    if not document_chunks:

        return index

    # ---------------------------------
    # Generate embeddings
    # ---------------------------------

    embeddings = embed_chunks(document_chunks)

    if not embeddings:

        return index


    # ---------------------------------
    # Create index if this is first run
    # ---------------------------------

    index = ensure_index(
        index,
        metadata,
        embeddings
    )

    # ---------------------------------
    # Allocate vector IDs
    # ---------------------------------

    vector_ids = allocate_vector_ids(
        metadata,
        len(document_chunks)
    )

    # ---------------------------------
    # Add vectors
    # ---------------------------------

    add_vectors(
        index,
        embeddings,
        vector_ids
    )

    # ---------------------------------
    # Store chunks
    # ---------------------------------

    for vector_id, chunk in zip(
        vector_ids,
        document_chunks
    ):

        apply_metadata(chunk, file_path)
        chunk.metadata["vector_id"] = vector_id

        chunks[vector_id] = chunk

    # ---------------------------------
    # Store hash information
    # ---------------------------------

    hashes[str(file_path)] = {

        "path": str(file_path),

        "hash": current_hash,

        "vector_ids": vector_ids,

        "chunk_count": len(document_chunks),

        "updated_at": datetime.now().isoformat()

    }

    return index

def process_modified_document(
    file_path,
    index,
    chunks,
    metadata,
    hashes,
    current_hash,
):
    """
    Update an existing document.
    """

    print(f"\nModified document: {file_path.name}")

    # ---------------------------------
    # Remove old vectors
    # ---------------------------------

    old_hash_entry = hashes.get(
        str(file_path),
        {}
    )

    old_vector_ids = []

    if isinstance(old_hash_entry, dict):

        old_vector_ids = old_hash_entry.get(
            "vector_ids",
            []
        )

    remove_vectors(
        index,
        old_vector_ids
    )
    # ---------------------------------
    # Remove old chunks
    # ---------------------------------

    for vector_id in old_vector_ids:

        chunks.pop(
            vector_id,
            None
        )

    # ---------------------------------
    # Rebuild document
    # ---------------------------------

    document_chunks = build_chunks(file_path)

    if not document_chunks:

        return index

    embeddings = embed_chunks(
        document_chunks
    )

    if not embeddings:

        return index

    index = ensure_index(
        index,
        metadata,
        embeddings
    )

    # ---------------------------------
    # Allocate new IDs
    # ---------------------------------

    new_vector_ids = allocate_vector_ids(
        metadata,
        len(document_chunks)
    )

    # ---------------------------------
    # Add vectors
    # ---------------------------------

    add_vectors(
        index,
        embeddings,
        new_vector_ids
    )

    # ---------------------------------
    # Store chunks
    # ---------------------------------

    for vector_id, chunk in zip(
        new_vector_ids,
        document_chunks
    ):

        apply_metadata(chunk, file_path)
        chunk.metadata["vector_id"] = vector_id

        chunks[vector_id] = chunk

    # ---------------------------------
    # Update hashes
    # ---------------------------------

    hashes[str(file_path)] = {

        "path": str(file_path),

        "hash": current_hash,

        "vector_ids": new_vector_ids,

        "chunk_count": len(document_chunks),

        "updated_at": datetime.now().isoformat()

    }

    return index

def process_deleted_documents(
    existing_files,
    index,
    chunks,
    metadata,
    hashes,
):
    """
    Remove documents that were deleted
    from the data folder.
    """

    current_files = {

        str(path)

        for path in existing_files

    }

    stored_files = list(
        hashes.keys()
    )

    deleted_count = 0

    for file_path in stored_files:

        if file_path in current_files:

            continue

        print(f"\nDeleted document: {Path(file_path).name}")

        stored_entry = hashes[file_path]

        vector_ids = []

        if isinstance(stored_entry, dict):

            vector_ids = stored_entry.get(
                "vector_ids",
                []
            )

        # -----------------------------
        # Remove vectors
        # -----------------------------

        remove_vectors(
            index,
            vector_ids
        )

        # -----------------------------
        # Remove chunks
        # -----------------------------

        for vector_id in vector_ids:

            chunks.pop(
                vector_id,
                None
            )

        # -----------------------------
        # Remove hash entry
        # -----------------------------

        del hashes[file_path]

        deleted_count += 1

    return index, deleted_count


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("=" * 60)
    print("Production RAG Incremental Ingestion")
    print("=" * 60)

    # ---------------------------------
    # Load existing vector store
    # ---------------------------------

    (
        index,
        chunks,
        metadata,
        hashes,
    ) = load_or_create_vector_store()

    # ---------------------------------
    # Find documents
    # ---------------------------------

    documents = find_documents()

    print(f"\nFound {len(documents)} document(s).")

    changed = False
    new_documents = 0
    modified_documents = 0
    deleted_documents = 0
    skipped_documents = 0

    # ---------------------------------
    # Remove deleted documents
    # ---------------------------------

    if index is not None:

        index, deleted_count = process_deleted_documents(
            documents,
            index,
            chunks,
            metadata,
            hashes,
        )

        if deleted_count:
            changed = True
            deleted_documents += deleted_count

    # ---------------------------------
    # Process every document
    # ---------------------------------

    for file_path in documents:

        current_hash = calculate_file_hash(file_path)

        file_key = str(file_path)

        # -----------------------------
        # New document
        # -----------------------------

        if file_key not in hashes:

            try:
                index = process_new_document(
                    file_path,
                    index,
                    chunks,
                    metadata,
                    hashes,
                    current_hash,
                )
                changed = True
                new_documents += 1
            except Exception:
                print(f"Failed to process {file_path.name}")
                traceback.print_exc()
                continue

        # -----------------------------
        # Modified document
        # -----------------------------

        elif hashes[file_key]["hash"] != current_hash:

            try:
                index = process_modified_document(
                    file_path,
                    index,
                    chunks,
                    metadata,
                    hashes,
                    current_hash,
                )
                changed = True
                modified_documents += 1
            except Exception:
                print(f"Failed to process {file_path.name}")
                traceback.print_exc()
                continue

        # -----------------------------
        # Unchanged
        # -----------------------------

        else:

            skipped_documents += 1
            print(
                f"Skipping {file_path.name} (unchanged)"
            )

    # ---------------------------------
    # Save
    # ---------------------------------

    if changed:

        if index is not None:

            save_vector_store(
                index,
                chunks,
                metadata,
                hashes,
            )

    else:

        print("\nNo changes detected. Vector store already up-to-date.")

    metadata["num_vectors"] = len(chunks)

    print("\n")
    print("=" * 60)
    print("Vector Store Summary")
    print("=" * 60)
    print(f"New Documents      : {new_documents}")
    print(f"Modified Documents : {modified_documents}")
    print(f"Deleted Documents  : {deleted_documents}")
    print(f"Skipped Documents  : {skipped_documents}")
    print("\n")
    print(f"Vectors            : {metadata['num_vectors']}")
    print(f"Documents          : {len(hashes)}")
    print(f"Next Vector ID     : {metadata['next_vector_id']}")
    print("=" * 60)

# -------------------------------------------------
# Entry Point
# -------------------------------------------------

if __name__ == "__main__":
    main()