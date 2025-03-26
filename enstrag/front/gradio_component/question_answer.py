import gradio as gr
from gradio_pdf import PDF

import requests
import pwd
import os
import re
from functools import partial

from ..utils import highlight_text_in_pdf

def ask(agent, query):
    result, retrieved_context, sources, (pdf_path, pdf_name, context_to_highlight) = agent.answer_question(query, verbose=True)

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
                raise RuntimeError(f"Failed to download {url}. Ignoring...")

    context_to_highlight = context_to_highlight[:25]
    pdf, page_number = highlight_text_in_pdf(pdf_path, context_to_highlight)
    print(pdf)
    agent.last_context = retrieved_context
    return result, sources + f" - Page {page_number}", gr.update(value=pdf, label=pdf_name, starting_page=3)

def clean_text(text):
    text = re.sub(r'[^\x20-\x7E\u00C0-\u00FF\u00B2\u00B3\u00B9\n]', '', text)
    text = re.sub(r'[^\S\n]+', ' ', text).strip()
    return text

def explain(agent, query, k, method):
    if not hasattr(agent, "last_context"):
        return "Ask a question before"
    
    chunks = agent.last_context.split("\n")

    text = " ".join(chunks)
    tokens = agent.top_k_tokens({"context": agent.last_context, "question": query}, int(k), method.lower())
    for token in tokens:
        token = clean_text(token)
        if not re.fullmatch(r"[^\w\s]+", token.strip()) and not len(token) < 3:
            text = text.replace(token, f"<span style='background-color:#ea580c; color:black'>{token}</span>")
    return text

def build_qa_panel(agent):
    with gr.Blocks() as rag:
        title = gr.HTML(f"<center><h1>Enstrag Bot</h1> <h3>{', '.join(agent.get_themes())}</h3></center>")
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
                with gr.Tabs():
                    with gr.Tab("Sources"):
                        srcs = gr.Textbox(label="Sources", interactive=False)
                        pdf = PDF(label="Document")
                    with gr.Tab("Explanation") as xai:
                        method = gr.Radio(choices=["Perturbation", "Gradient"], label="Method", value="Perturbation")
                        k = gr.Slider(minimum=1, maximum=30, value=10, step=1, label="Number of explanation tokens")
                        button_explain = gr.Button("Explain")
                        explanation = gr.Markdown()

                button_explain.click(fn=partial(explain, agent), inputs=[input, k, method], outputs=[explanation])
            
        btn.click(fn=partial(ask, agent), inputs=[input], outputs=[output, srcs, pdf])
    return rag