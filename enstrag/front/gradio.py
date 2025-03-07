import os
import requests
import pwd
import time
import shutil
import re
from hashlib import sha256

import gradio as gr
from gradio_pdf import PDF

from .base_front import Front
from .utils import highlight_text_in_pdf
from ..data.parser import FileDocument, store_filedoc, load_filedocs, Parser

HASH_PASSWORD = "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"

def toggle_visibility(choice):
    if choice == "URL":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

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
    
    def add_doc(self, url, path, name, label):
        if url is not None and re.match(r'^https?://', url) is None:
            return f"Invalid URL. Got {url}"
        
        if path is None:
            path = Parser.download_pdf(url, name)

        filedoc = FileDocument(url, path, name, label)
        store_filedoc(filedoc)
        self.agent.db.add_document(Parser.get_document_from_filedoc(filedoc))
        print(f"Received: {url}, {path}, {name}, {label}")
        return f"Document {name} added successfully"

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
                admin_actions.visible = True
                login.visible = False
                return "Login successful", gr.update(visible=True), gr.update(visible=False)
            else:
                admin_actions.visible = False
                login.visible = True
                time.sleep(1)
                return "Incorrect password", gr.update(visible=False), gr.update(visible=True)

        with gr.Blocks() as admin_panel:
            with gr.Row(visible=True) as login:
                with gr.Column():
                    pass
                with gr.Column():
                    password_input = gr.Textbox(label="Password", type="password", interactive=True)
                    login_button = gr.Button("Login", variant="primary")
                    login_output = gr.Markdown(label="")
                with gr.Column():
                    pass

            with gr.Tabs(visible=False) as admin_actions:
                with gr.Tab("Add document"):
                    with gr.Column():
                        pass
                    with gr.Column() as add_doc:
                        name = gr.Textbox(label="Name", interactive=True)
                        label = gr.Dropdown(choices=self.agent.db.themes, label="Label", interactive=True, allow_custom_value=True)

                        method = gr.Radio(choices=["URL", "Upload PDF"], label="Add document via", value="URL")
                        
                        url = gr.Textbox(label="PDF URL", interactive=True)
                        submit_button = gr.Button("Add document", variant="primary")

                        upload = gr.UploadButton("Upload PDF (may take a while)", file_count="single", variant="primary", visible=False)
                        
                        submit_output = gr.Markdown(label="")

                        def add_document(url, upload, name, label, method):
                            if name is None or label is None:
                                return "Name and label are required"
                            
                            if method == "URL":
                                return self.add_doc(url, None, name, label)
                            else:
                                # Handle file upload
                                file_path = upload
                                path = os.path.join(os.environ.get("PERSIST_PATH"), "pdfs", name.replace(" ", "_")+".pdf")
                                shutil.copyfile(file_path, path)
                                return self.add_doc(None, path, name, label)

                        method.change(fn=toggle_visibility, inputs=method, outputs=[url, submit_button, upload])
                        upload.upload(add_document, inputs=[url, upload, name, label, method], outputs=submit_output)


                        submit_button.click(fn=add_document, inputs=[url, upload, name, label], outputs=submit_output)
                    with gr.Column():
                        pass

                with gr.Tab("See documents"):
                    with gr.Blocks() as see_docs:
                        docs = load_filedocs()
                        string = "<ul>\n"
                        for doc in docs:
                            string += f"<li>{doc.name} ({doc.label})</li>\n"
                        string += "</ul>"
                        gr.Markdown(string)

            login_button.click(fn=check_password, inputs=password_input, outputs=[login_output, admin_actions, login])

        demo = gr.TabbedInterface([rag, admin_panel], tab_names=["Enstrag Bot", "Admin panel"])

        demo.launch(
            share=share,
            allowed_paths=[os.environ.get("PERSIST_PATH")],
        )