import gradio as gr
import os
import shutil
import requests
import pwd
from gradio_pdf import PDF
from .base_front import Front

import pymupdf

# Function to highlight text in PDF
def highlight_text_in_pdf(pdf_file: str, highlight_text):
    page_number = 1
    doc = pymupdf.open(pdf_file)  # Open the PDF
    for page in doc:
        text_instances = page.search_for(highlight_text)  # Find text to highlight
        if len(text_instances) > 0:
            page_number = page.number
        for inst in text_instances:
            page.add_highlight_annot(inst)  # Highlight text
    # Save the modified PDF
    new_pdf_file = pdf_file.split("/")
    new_pdf_file[-1] = "new_" + new_pdf_file[-1]
    new_pdf_file = "/".join(new_pdf_file)

    doc.save(new_pdf_file)
    return new_pdf_file, page_number + 1

class GradioFront(Front):
    def ask(self, query):
        result, retrieved_context, sources, url = self.agent.answer_question(query, verbose=True)

        TMP_FOLDER = "/tmp/enstrag_"+str(pwd.getpwuid(os.getuid())[0])
        os.makedirs(TMP_FOLDER, exist_ok=True)

        with open(os.path.join(TMP_FOLDER, 'tmp.pdf'), 'wb') as f:
            try:
                response = requests.get(url)
                f.write(response.content)
            except Exception:
                print(f"Failed to download {url}.")            

        pdf, page_number = highlight_text_in_pdf(os.path.join(TMP_FOLDER, 'tmp.pdf'), retrieved_context[:25])
        return result, sources + f" - Page {page_number}", PDF(pdf, label="Document", starting_page=page_number, interactive=True)

    def launch(self):

        with gr.Blocks() as self.demo:
            title = gr.TextArea("Enstrag Bot", lines=1, interactive=False)
            with gr.Row():
                with gr.Column(scale=2):
                    input = gr.Textbox(label="Question", autofocus=True, interactive=True)
                    btn = gr.Button("Ask", variant="primary")
                    output = gr.Markdown(label="Anwser")
                with gr.Column(scale=2):
                    srcs = gr.Textbox(label="Sources", interactive=False)
                    pdf = PDF(label="Document")
                
            btn.click(fn=self.ask, inputs=input, outputs=[output, srcs, pdf])

        self.demo.launch(share=True)