import argparse
parser = argparse.ArgumentParser(prog='Enstrag')

parser.add_argument('-r', '--reset', action='store_true', help="Reset the vector database on start.")
parser.add_argument('-v', '--version', action='store_true', help="Show version")
args = parser.parse_args()

if args.version:
    from . import __version__
    print(__version__)
    exit(0)

from . import verify_execution
verify_execution()

print("Importing packages...")
from .rag import RagAgent
from .models import get_pipeline, RagEmbedding
from .data import VectorDB, Parser, FileDocument

import gradio as gr

llm_folder = "Qwen2.5-1.5B-Instruct"
embedding_folder = "all-MiniLM-L6-v2"
persist_directory = "/home/ensta/ensta-leguery/enstrag_folder"

db = VectorDB(RagEmbedding(embedding_folder), persist_directory=persist_directory)

if args.reset:
    print("Resetting database...")
    db.db.reset_collection()

db.add_documents(
    Parser.get_documents_from_filedocs([
        FileDocument("http://www.cs.man.ac.uk/~fumie/tmp/bishop.pdf", "ML Bishop", "Machine learning"),
        FileDocument("https://www.maths.lu.se/fileadmin/maths/personal_staff/Andreas_Jakobsson/StoicaM05.pdf", "SPECTRAL ANALYSIS OF SIGNALS", "Physics"),
        FileDocument("https://www.math.toronto.edu/khesin/biblio/GoldsteinPooleSafkoClassicalMechanics.pdf", "CLASSICAL MECHANICS", "Physics"),
        FileDocument("https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf", "Convex Optimization", "Maths"),
        #"https://www.damtp.cam.ac.uk/user/tong/qft/qft.pdf",
        FileDocument("http://students.aiu.edu/submissions/profiles/resources/onlineBook/Z6W3H3_basic%20algebra%20geometry.pdf", "Basic Algebraic Geometry", "Maths"),
        FileDocument("https://assets.openstax.org/oscms-prodcms/media/documents/OrganicChemistry-SAMPLE_9ADraVJ.pdf", "Organic Chemistry", "Chemistry"),
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