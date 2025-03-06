import os
import requests
import pwd

import gradio as gr
from gradio_pdf import PDF

from .base_front import Front
from .utils import highlight_text_in_pdf


class GradioFront(Front):
    def ask(self, query):
        result, retrieved_context, sources, (pdf_path, pdf_name, context_to_highlight) = self.agent.answer_question(query, verbose=True)

        if os.environ.get("PERSIST_PATH") is None:
            url = pdf_path
            TMP_FOLDER = "/tmp/enstrag_"+str(pwd.getpwuid(os.getuid())[0])
            pdf_path = os.path.join(TMP_FOLDER, "tmp.pdf")
            os.makedirs(TMP_FOLDER, exist_ok=True)
            with open(pdf_path, 'wb') as f:
                try:
                    response = requests.get(url)
                    f.write(response.content)
                except Exception:
                    print(f"Failed to download {url}. Ignoring...")
                    return ""

        context_to_highlight = context_to_highlight[:25]
        pdf, page_number = highlight_text_in_pdf(pdf_path, context_to_highlight)
        print(pdf)
        return result, sources + f" - Page {page_number}", PDF(pdf, label=pdf_name, starting_page=3, interactive=True)

    def launch(self, share: bool = False) -> None:

        with gr.Blocks() as self.demo:
            title = gr.HTML(f"<center><h1>Enstrag Bot</h1> <h3>{', '.join(self.agent.db.themes)}</h3></center>")
            with gr.Row():
                with gr.Column(scale=2):
                    input = gr.Textbox(label="Question", autofocus=True, interactive=True)
                    btn = gr.Button("Ask", variant="primary")
                    output = gr.Markdown(
                        label="Anwser",
                        latex_delimiters=[
                            { "left": "$$", "right": "$$", "display": True },
                            { "left": "$", "right": "$", "display": False }
                        ]
                    )
                with gr.Column(scale=2):
                    srcs = gr.Textbox(label="Sources", interactive=False)
                    pdf = PDF(label="Document")
                
            btn.click(fn=self.ask, inputs=input, outputs=[output, srcs, pdf])

        self.demo.launch(
            share=share,
            allowed_paths=[os.environ.get("PERSIST_PATH")],
        )