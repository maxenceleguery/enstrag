import gradio as gr
import os
import shutil
import time
import re
from functools import partial
from hashlib import sha256
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class FileDocument:
    url: str | None
    local_path: str | None
    name: str
    label: str

HASH_PASSWORD = os.getenv("HASH_PASSWORD")

def check_password(password: str):
    if sha256(password.encode('utf-8')).hexdigest() == HASH_PASSWORD:
        return "Login successful", gr.update(visible=True), gr.update(visible=False)
    else:
        time.sleep(1)
        return "Incorrect password", gr.update(visible=False), gr.update(visible=True)
    
def toggle_visibility(choice):
    if choice == "URL":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    
def add_doc(agent, url, path, name, label):
    if url is not None and re.match(r'^https?://.*\.pdf$', url, re.IGNORECASE) is None:
        return f"Invalid URL. Got {url}"

    filedoc = FileDocument(url, path, name, label)
    agent.add_filedoc(filedoc)
    return f"Document {name} added successfully"

def add_document(agent, url, upload, name, label, method):
    print(f"{url=}, {upload=}, {name=}, {label=}, {method=}")
    if name is None or label is None:
        return "Name and label are required"
    
    try:
        if method == "URL":
            return add_doc(agent, url, None, name, label)
        else:
            # Handle file upload
            file_path = upload
            path = os.path.join(os.environ.get("PERSIST_PATH"), "pdfs", name.replace(" ", "_")+".pdf")
            shutil.copyfile(file_path, path)
            return add_doc(agent, None, path, name, label)
    except Exception as e:
        return f"Error : {e}"

def build_admin_panel(agent):

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
                    label = gr.Dropdown(choices=agent.get_themes(), label="Label", interactive=True, allow_custom_value=True)

                    method = gr.Radio(choices=["URL", "Upload PDF"], label="Add document via", value="URL")
                    
                    url = gr.Textbox(label="PDF URL", interactive=True)
                    submit_button = gr.Button("Add document", variant="primary")

                    upload = gr.UploadButton("Upload PDF (may take a while)", file_count="single", variant="primary", visible=False)
                    
                    submit_output = gr.Markdown(label="")

                    method.change(fn=toggle_visibility, inputs=method, outputs=[url, submit_button, upload])
                    upload.upload(fn=partial(add_document, agent), inputs=[url, upload, name, label, method], outputs=submit_output)

                    submit_button.click(fn=partial(add_document, agent), inputs=[url, upload, name, label, method], outputs=submit_output)
                with gr.Column():
                    pass

            with gr.Tab("See documents"):
                with gr.Blocks() as see_docs:
                    docs = agent.get_docs()
                    string = "<ul>\n"
                    for doc in docs:
                        string += f"<li>{doc.name} ({doc.label})</li>\n"
                    string += "</ul>"
                    gr.Markdown(string)

        login_button.click(fn=check_password, inputs=password_input, outputs=[login_output, admin_actions, login])
    return admin_panel