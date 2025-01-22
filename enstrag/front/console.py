from .base_front import Front

class ConsoleFront(Front):
    def launch(self):
        while True:
            query = input("Enter the question (Type exit to close)\n>>>")
            if query == "exit":
                break
            result, retrieved_context = self.agent.answer_question(query, verbose=True)
            