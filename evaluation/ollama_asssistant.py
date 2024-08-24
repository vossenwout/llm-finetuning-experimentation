import ollama


class OllamaAssistant:
    def __init__(
        self, system_prompt, model_name, is_chat_mode=True, print_messages=False
    ):
        self.model_name = model_name
        self.message_history = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        self.is_chat_mode = is_chat_mode
        self.print_messages = print_messages

    def manage_message_history(self):
        if len(self.message_history) > 8:
            self.message_history.pop(0)
            print("Message history is too long. Removed the oldest message.")
        return self.message_history

    def reset_message_history(self):
        self.message_history = [
            {
                "role": "system",
                "content": "Resetting the message history.",
            }
        ]

    def generate_answer(self, question):
        question = {
            "role": "user",
            "content": question,
        }
        self.message_history.append(question)
        stream = ollama.chat(
            model=self.model_name,
            messages=self.message_history,
            stream=True,
        )

        answer = ""
        for chunk in stream:
            answer += chunk["message"]["content"]
            if self.print_messages:
                print(chunk["message"]["content"], end="", flush=True)

        self.message_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )
        self.manage_message_history()

        if not self.is_chat_mode:
            self.reset_message_history()

        return answer

    def print_message_history(self):
        print("\n")
        print(self.message_history)
