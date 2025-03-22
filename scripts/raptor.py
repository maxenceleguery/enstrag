from enstrag import get_args

args = get_args()

from enstrag import verify_execution
verify_execution()

print("Importing packages...")
from enstrag.rag import RagAgent
from enstrag.models import get_pipeline
from enstrag.data import Parser, FileDocument, RAPTORVectorDB

import gradio as gr

llm_folder = args.llm_folder
embedding_folder = args.embedding_folder
persist_directory = args.persist_dir

db = RAPTORVectorDB()
db.add_documents(
    Parser.get_documents_from_filedocs([
        FileDocument("http://www.cs.man.ac.uk/~fumie/tmp/bishop.pdf", "ML Bishop", "Machine learning"),
        #FileDocument("https://www.maths.lu.se/fileadmin/maths/personal_staff/Andreas_Jakobsson/StoicaM05.pdf", "SPECTRAL ANALYSIS OF SIGNALS", "Physics"),
        #FileDocument("https://www.math.toronto.edu/khesin/biblio/GoldsteinPooleSafkoClassicalMechanics.pdf", "CLASSICAL MECHANICS", "Physics"),
        #FileDocument("https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf", "Convex Optimization", "Maths"),
        #"https://www.damtp.cam.ac.uk/user/tong/qft/qft.pdf",
        #FileDocument("http://students.aiu.edu/submissions/profiles/resources/onlineBook/Z6W3H3_basic%20algebra%20geometry.pdf", "Basic Algebraic Geometry", "Maths"),
        #FileDocument("https://assets.openstax.org/oscms-prodcms/media/documents/OrganicChemistry-SAMPLE_9ADraVJ.pdf", "Organic Chemistry", "Chemistry"),
        #"https://arxiv.org/pdf/1706.03762",
        #"https://arxiv.org/pdf/2106.09685"
    ], get_pages_num=False)
)

print(db.get_context_from_query("What is a guassian distribution"))

exit(0)

"""
if args.reset:
    print("Resetting database...")
    db.db.reset_collection()
"""


agent = RagAgent(
    pipe=get_pipeline(llm_folder),
    db=db,
)

def ask(query, history):
    result, retrieved_context, *_ = agent.answer_question(query, verbose=True)
    return result

demo = gr.ChatInterface(fn=ask, type="messages", title="Enstrag Bot")
demo.launch(share=True)
