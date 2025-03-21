from .base_front import Front

class ConsoleFront(Front):
    def launch(self, share: bool = False) -> None:
        while True:
            query = input("Enter the question (Type exit to close)\n>>>")
            if query == "exit":
                break
            result, retrieved_context, *_ = self.agent.answer_question(query, verbose=True)

class XAIConsoleFront(Front):
    def __init__(self, agent, pipeline_xrag):
        super().__init__(agent)
        self.pipeline_xrag = pipeline_xrag

    def launch(self, share: bool = False) -> None:
        while True:
            query = input("Enter the question (Type exit to close)\n>>>")
            if query == "exit":
                break
            result, retrieved_context, *_ = self.agent.answer_question(query, verbose=True)

            # Explainable part
            k = int(input("How many top explicative tokens do you want?\n>>>"))
            prompt = {"context": retrieved_context, "question": query}
            top_tokens = self.pipeline_xrag.top_k_tokens(prompt, k)

            print(f"\nThe top {k} tokens are", top_tokens)
            