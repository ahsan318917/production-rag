from pathlib import Path
from datetime import datetime

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
    save_index,
    save_chunks,
    save_metadata,
)


# -------------------------------------------------
# Supported File Types
# -------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx"
}


# -------------------------------------------------
# Find All Documents
# -------------------------------------------------

def find_documents():

    documents = []

    for file in DATA_DIR.rglob("*"):

        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(file)

    return documents


# -------------------------------------------------
# Load Document
# -------------------------------------------------

def load_document(file_path: Path):

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


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("=" * 60)
    print("Production RAG Ingestion")
    print("=" * 60)

    document_paths = find_documents()

    print(f"\nFound {len(document_paths)} document(s).\n")

    all_chunks = []

    # -------------------------------------
    # Process Documents
    # -------------------------------------

    for document_path in document_paths:

        print("=" * 60)
        print(f"Loading: {document_path.name}")
        print("=" * 60)

        try:

            docs = load_document(document_path)

            print(f"Loaded {len(docs)} document object(s).")

            chunks = chunk_documents(docs)

            print(f"Created {len(chunks)} chunk(s).")

            all_chunks.extend(chunks)

        except Exception as e:

            print(f"\nFailed to process {document_path.name}")
            print(type(e).__name__)
            print(e)

    # -------------------------------------
    # Chunk Summary
    # -------------------------------------

    print("\n" + "=" * 60)
    print("Chunk Summary")
    print("=" * 60)

    print(f"Total Chunks: {len(all_chunks)}")

    # -------------------------------------
    # Display Chunks
    # -------------------------------------

    for index, chunk in enumerate(all_chunks, start=1):

        print("-" * 60)
        print(f"Chunk {index}")
        print("-" * 60)

        print(chunk.page_content[:250])

        print("\nMetadata:")
        print(chunk.metadata)

        print()

    # -------------------------------------
    # Prepare Text
    # -------------------------------------

    texts = [
        chunk.page_content
        for chunk in all_chunks
    ]

    # -------------------------------------
    # Generate Embeddings
    # -------------------------------------

    print("=" * 60)
    print("Generating Embeddings")
    print("=" * 60)

    embeddings = generate_embeddings(texts)

    print(f"\nGenerated {len(embeddings)} embeddings.")

    print("\nEmbedding Dimension:", len(embeddings[0]))

    print("\nFirst 10 Values:")

    print(embeddings[0][:10])

    # -------------------------------------
    # Create Metadata
    # -------------------------------------

    metadata = {

        "embedding_model": EMBED_MODEL,

        "dimension": len(embeddings[0]),

        "num_vectors": len(embeddings),

        "created_at": datetime.now().isoformat()

    }

    # -------------------------------------
    # Create FAISS Index
    # -------------------------------------

    print("\nCreating FAISS index...")

    index = create_index(embeddings)

    print("FAISS index created.")

    # -------------------------------------
    # Save Vector Store
    # -------------------------------------

    print("\nSaving FAISS index...")
    save_index(index)

    print("Saving chunks...")
    save_chunks(all_chunks)

    print("Saving metadata...")
    save_metadata(metadata)

    print("\nVector Store Saved Successfully!")

    print("=" * 60)


# -------------------------------------------------
# Entry Point
# -------------------------------------------------

if __name__ == "__main__":
    main()