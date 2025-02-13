import gradio as gr
import os
import shutil
import tqdm
import requests
import pwd
from gradio_pdf import PDF
from .base_front import Front

import pymupdf

# Function to highlight text in PDF
def highlight_text_in_pdf(pdf_file: str, highlight_text):
    page_number = 1
    doc = pymupdf.open(pdf_file)  # Open the PDF
    for page in tqdm.tqdm(doc, desc="Searching in doc..."):
        text_instances = page.search_for(highlight_text)  # Find text to highlight
        if len(text_instances) > 0:
            page_number = page.number
        for inst in text_instances:
            page.add_highlight_annot(inst)  # Highlight text

    tmp = [page_number-2, page_number-1, page_number, page_number+1, page_number+2]
    pages = []
    for page in tmp:
        if 0 <= page < len(doc):
            pages.append(page)
    print(pages)
    doc.select(pages)

    # Save the modified PDF
    new_pdf_file = pdf_file.split("/")
    new_pdf_file[-1] = "new_" + new_pdf_file[-1]
    new_pdf_file = "/".join(new_pdf_file)

    doc.save(new_pdf_file)
    return new_pdf_file, page_number + 1

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

    def launch(self):

        with gr.Blocks() as self.demo:
            title = gr.HTML(f"<center><h1>Enstrag Bot</h1> <h3>{', '.join(self.agent.db.themes)}</h3></center>")
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