import gradio as gr
from .base_front import Front

class GradioFront(Front):
    def ask(self, query, history):
        result, retrieved_context = self.agent.answer_question(query, verbose=True)
        return result

    def launch(self):
        demo = gr.ChatInterface(fn=self.ask, type="messages", title="Enstrag Bot")
        demo.launch(share=True)