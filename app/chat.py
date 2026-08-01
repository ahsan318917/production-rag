from hybrid_retriever import hybrid_retrieve

from generator import (
    stream_response,
    get_sources,
)

from history_retriever import rewrite_query
from query_decomposer import decompose_query
from reranker import rerank
from context_compressor import compress_context

from memory import (
    ConversationMemory,
)

# -------------------------------------------------
# Banner
# -------------------------------------------------

def print_banner():

    print("=" * 60)
    print("Production RAG Chatbot")
    print("=" * 60)
    print("Type 'exit' to quit.")
    print()


# -------------------------------------------------
# Chat Loop
# -------------------------------------------------

def main():

    print_banner()

    # -----------------------------------------
    # Conversation Memory
    # -----------------------------------------

    memory = ConversationMemory()

    while True:

        query = input("You: ").strip()

        if not query:
            continue

        if query.lower() in ["exit", "quit"]:

            print("\nGoodbye!")

            break

        # -----------------------------------------
        # Rewrite Follow-up Query for Retrieval
        # -----------------------------------------

        standalone_query = rewrite_query(query, memory)

        # -----------------------------------------
        # Decompose Query for Retrieval
        # -----------------------------------------

        sub_queries = decompose_query(standalone_query)

        print("\nSub-queries:")
        for item in sub_queries:
            print(f"- {item}")

        # -----------------------------------------
        # Retrieve Documents
        # -----------------------------------------

        all_retrieved_chunks = []
        seen_chunk_keys = set()

        for sub_query in sub_queries:
            retrieved_chunks = hybrid_retrieve(sub_query)

            for chunk_result in retrieved_chunks:
                chunk = chunk_result.get("chunk")

                if chunk is None:
                    continue

                metadata = getattr(chunk, "metadata", {}) or {}
                chunk_id = metadata.get("chunk_id") or metadata.get("chunk")

                if chunk_id is not None:
                    key = f"chunk_id:{chunk_id}"
                else:
                    key = chunk.page_content.strip()

                if key in seen_chunk_keys:
                    continue

                seen_chunk_keys.add(key)
                all_retrieved_chunks.append(chunk_result)

        retrieved_chunks = all_retrieved_chunks

        if not retrieved_chunks:

            print("\nAI: I don't have enough information to answer that.\n")

            memory.add_user_message(query)

            memory.add_assistant_message(
                "I don't have enough information to answer that."
            )

            continue

        # -----------------------------------------
        # Rerank Retrieved Chunks
        # -----------------------------------------

        print("\nReranking...")

        reranked_chunks = rerank(
            query=query,
            retrieved_chunks=retrieved_chunks,
        )

        for item in reranked_chunks:
            print(f"Chunk {item['vector_id']} : {item['score']:.2f}")

        if not reranked_chunks:
            print("\nAI: I don't have enough information to answer that.\n")

            memory.add_user_message(query)

            memory.add_assistant_message(
                "I don't have enough information to answer that."
            )

            continue

        print(f"\nKeeping top {len(reranked_chunks)} chunks.")

        compressed_chunks = compress_context(
            query=query,
            retrieved_chunks=reranked_chunks,
        )

        retrieved_chunks = compressed_chunks

        # -----------------------------------------
        # Generate Response
        # -----------------------------------------

        print("\nAI: ", end="", flush=True)

        full_response = ""

        try:

            for token in stream_response(
                query=query,
                retrieved_chunks=retrieved_chunks,
                memory=memory,
            ):

                print(token, end="", flush=True)

                full_response += token

        except Exception as e:

            print("\n")

            print("Error while generating response:")

            print(e)

            continue

        print("\n")

        # -----------------------------------------
        # Update Memory
        # -----------------------------------------

        memory.add_user_message(query)

        memory.add_assistant_message(full_response)

        # -----------------------------------------
        # Display Sources
        # -----------------------------------------

        sources = get_sources(retrieved_chunks)

        if sources:

            print("-" * 60)

            print("Sources")

            print()

            for i, source in enumerate(sources, start=1):

                print(f"{i}. {source}")

            print("-" * 60)

        print()


# -------------------------------------------------
# Entry Point
# -------------------------------------------------

if __name__ == "__main__":

    main()