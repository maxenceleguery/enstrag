import os
import requests
import pwd
import time
import re
from hashlib import sha256

import gradio as gr
from gradio_pdf import PDF

from .base_front import Front
from .utils import highlight_text_in_pdf
from ..data.parser import FileDocument, store_filedoc

HASH_PASSWORD = "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"

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
    
    def add_doc(self, url, name, label):
        filedoc = FileDocument(url, name, label)
        if re.match(r'^https?://', url) is None:
            return "Invalid URL"
        
        store_filedoc(filedoc)
        return f"Received: {url}, {name}, {label}"

    def launch(self, share: bool = False) -> None:

        with gr.Blocks() as rag:
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

        def check_password(password):
            if sha256(password.encode('utf-8')).hexdigest() == HASH_PASSWORD:
                admin_controls.visible = True
                login.visible = False
                return "Login successful", gr.update(visible=True), gr.update(visible=False)
            else:
                admin_controls.visible = False
                login.visible = True
                time.sleep(1)
                return "Incorrect password", gr.update(visible=False), gr.update(visible=True)

        with gr.Blocks() as admin_panel:
            with gr.Column(visible=True, scale=0) as login:
                password_input = gr.Textbox(label="Password", type="password", interactive=True)
                login_button = gr.Button("Login", variant="primary")
                login_output = gr.Markdown(label="")

            with gr.Column(visible=False, scale=0) as admin_controls:
                url = gr.Textbox(label="PDF URL", interactive=True)
                name = gr.Textbox(label="Name", interactive=True)
                label = gr.Textbox(label="Label", interactive=True)
                submit_button = gr.Button("Add document", variant="primary")
                submit_output = gr.Markdown(label="")

            login_button.click(fn=check_password, inputs=password_input, outputs=[login_output, admin_controls, login])
            submit_button.click(fn=self.add_doc, inputs=[url, name, label], outputs=submit_output)
        

        demo = gr.TabbedInterface([rag, admin_panel], tab_names=["Enstrag Bot", "Admin panel"])

        demo.launch(
            share=share,
            allowed_paths=[os.environ.get("PERSIST_PATH")],
        )