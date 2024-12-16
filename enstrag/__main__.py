print("Importing packages...")
from .rag import RagAgent
from .models import get_pipeline, RagEmbedding
from .data import VectorDB, Parser

import argparse
import gradio as gr

parser = argparse.ArgumentParser(prog='Enstrag')

parser.add_argument('-r', '--reset', action='store_true', help="Reset the vector database on start.")
args = parser.parse_args()

llm_folder = "Qwen2.5-1.5B-Instruct"
embedding_folder = "all-MiniLM-L6-v2"
persist_directory="/tmp/enstrag"

db = VectorDB(RagEmbedding(embedding_folder), persist_directory=persist_directory)

if args.reset:
    print("Resetting database...")
    db.db.reset_collection()

db.add_documents(
    Parser.get_documents_from_pdf_urls([
        "http://www.cs.man.ac.uk/~fumie/tmp/bishop.pdf",
        #"https://arxiv.org/pdf/1706.03762",
        #"https://arxiv.org/pdf/2106.09685"
    ])
)

agent = RagAgent(
    pipe=get_pipeline(llm_folder),
    db=db,
)

def ask(query, history):
    result, retrieved_context = agent.answer_question(query, verbose=True)
    return result

demo = gr.ChatInterface(fn=ask, type="messages", title="Enstrag Bot")
demo.launch(share=True)

"""
while True:
    query = input("Enter the question (Type exit to close)\n>>>")
    if query == "exit":
        break
    result, retrieved_context = agent.answer_question(query, verbose=True)
"""