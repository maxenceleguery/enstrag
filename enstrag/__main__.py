print("Importing packages...")
from .rag import RagAgent
from .models import get_pipeline, RagEmbedding
from .data import VectorDB, Parser

# Explainable RAG
from .explanation.pipeline import XRAGPipeline
from .explanation.perturber import LeaveOneOutPerturber
from .explanation.generate import SimpleGenerator
from .explanation.compare import EmbeddingComparator

from transformers import AutoTokenizer, AutoConfig
import argparse
parser = argparse.ArgumentParser(prog='Enstrag')

parser.add_argument('-r', '--reset', action='store_true', help="Reset the vector database on start.")
parser.add_argument('-v', '--version', action='store_true', help="Show version")
parser.add_argument('--persist_dir', type=str, default="/home/ensta/ensta-leguery/enstrag_folder")
parser.add_argument('--models_path', type=str, default="/home/ensta/data")
parser.add_argument('--llm_folder', type=str, default="Qwen2.5-1.5B-Instruct")
parser.add_argument('--embedding_folder', type=str, default="all-MiniLM-L6-v2")
args = parser.parse_args()

if args.version:
    from . import __version__
    print(__version__)
    exit(0)

from . import verify_execution
verify_execution()

import os
os.environ["MODELS_PATH"] = args.models_path

print("Importing packages...")
from .rag import RagAgent
from .models import get_pipeline, RagEmbedding
from .data import VectorDB, Parser, FileDocument

import gradio as gr

llm_folder = args.llm_folder
embedding_folder = args.embedding_folder
persist_directory = args.persist_dir

embedding = RagEmbedding(embedding_folder)
db = VectorDB(embedding, persist_directory=persist_directory)

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
# Explainable RAG
tokenizer = AutoTokenizer.from_pretrained('/home/ensta/data/' + llm_folder)
config = AutoConfig.from_pretrained('/home/ensta/data/' + llm_folder)

perturber = LeaveOneOutPerturber()
generator = SimpleGenerator()
comparator = EmbeddingComparator()

pipeline_xrag = XRAGPipeline(perturber, generator, comparator, tokenizer, agent, embedding)

"""
def ask(query, history):
    result, retrieved_context = agent.answer_question(query, verbose=True)

    return result
with gr.Blocks() as demo:
    gr.ChatInterface(fn=ask, type="messages", title="Enstrag Bot")
    explain = gr.Button("Explain solution")
demo.launch(share=True)
"""

while True:
    query = input("Enter the question (Type exit to close)\n>>>")
    if query == "exit":
        break
    result, retrieved_context = agent.answer_question(query, verbose=True)

    # Explainable part
    k = int(input("How many top explicative tokens do you want?\n>>>"))
    prompt = {"context": retrieved_context, "question": query}
    top_tokens = pipeline_xrag.top_k_tokens(prompt, k)

    print(f"\nThe top {k} tokens are", top_tokens)
