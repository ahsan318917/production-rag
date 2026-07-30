class ConversationMemory:
    """
    Simple conversation memory for a RAG chat session.
    Stores user and assistant messages in order.
    """

    def __init__(self):

        self.messages = []

    def add_user_message(self, content):

        self.messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

    def add_assistant_message(self, content):

        self.messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )

    def get_history(self):

        return list(self.messages)

    def clear(self):

        self.messages.clear()

    def __len__(self):

        return len(self.messages)
