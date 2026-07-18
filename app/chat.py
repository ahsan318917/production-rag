from retrieve import retrieve
from generator import (
    stream_response,
    get_sources,
)

# -------------------------------------------------
# Chat Application
# -------------------------------------------------

def main():

    print("=" * 60)
    print("Production RAG Chatbot")
    print("=" * 60)

    print("Type 'exit' to quit.\n")

    while True:

        query = input("You: ")

        if query.lower() == "exit":
            print("\nGoodbye!")
            break

        # ---------------------------------
        # Retrieve Relevant Chunks
        # ---------------------------------

        retrieved_chunks = retrieve(query)

        if len(retrieved_chunks) == 0:

            print("\nAI: I couldn't find any relevant information.\n")
            continue

        # ---------------------------------
        # Generate Response
        # ---------------------------------

        print("\nAI: ", end="", flush=True)

        for token in stream_response(
            query,
            retrieved_chunks
        ):
            print(token, end="", flush=True)

        print("\n")

        # ---------------------------------
        # Display Sources
        # ---------------------------------

        sources = get_sources(retrieved_chunks)

        print("Sources:")
        print("-" * 60)

        for index, source in enumerate(sources, start=1):
            print(f"{index}. {source}")

        print("\n")


# -------------------------------------------------
# Entry Point
# -------------------------------------------------

if __name__ == "__main__":
    main()