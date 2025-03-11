from . import get_args
args = get_args()

from . import verify_execution
verify_execution()

print("Importing packages...")
from .rag import RagAgent
from .models import get_pipeline, RagEmbedding
from .data import VectorDB, load_filedocs, Parser, FileDocument
from .front import GradioFront, XAIConsoleFront

# Explainable RAG
if args.explained:
    from .explanation.pipeline import XRAGPipeline
    from .explanation.perturber import LeaveOneOutPerturber
    from .explanation.generate import SimpleGenerator
    from .explanation.compare import EmbeddingComparator

    from transformers import AutoTokenizer, AutoConfig

llm_folder = args.llm_folder
embedding_folder = args.embedding_folder
persist_directory = args.persist_dir

embedding = RagEmbedding(embedding_folder)
db = VectorDB(embedding, persist_directory=persist_directory)

if args.reset:
    print("Resetting database...")
    db.db.reset_collection()

db.add_documents(
    Parser.get_documents_from_filedocs(
        load_filedocs()
    )
)

"""
db.add_documents(
    Parser.get_documents_from_filedocs([
        FileDocument("http://www.cs.man.ac.uk/~fumie/tmp/bishop.pdf", None, "ML Bishop", "Machine learning"),
        FileDocument("https://www.maths.lu.se/fileadmin/maths/personal_staff/Andreas_Jakobsson/StoicaM05.pdf", None, "SPECTRAL ANALYSIS OF SIGNALS", "Physics"),
        FileDocument("https://www.math.toronto.edu/khesin/biblio/GoldsteinPooleSafkoClassicalMechanics.pdf", None, "CLASSICAL MECHANICS", "Physics"),
        FileDocument("https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf", None, "Convex Optimization", "Maths"),
        #"https://www.damtp.cam.ac.uk/user/tong/qft/qft.pdf",
        FileDocument("http://students.aiu.edu/submissions/profiles/resources/onlineBook/Z6W3H3_basic%20algebra%20geometry.pdf", None, "Basic Algebraic Geometry", "Maths"),
        FileDocument("https://assets.openstax.org/oscms-prodcms/media/documents/OrganicChemistry-SAMPLE_9ADraVJ.pdf", None, "Organic Chemistry", "Chemistry"),
        #"https://arxiv.org/pdf/1706.03762",
        #"https://arxiv.org/pdf/2106.09685"
    ])
)
"""

agent = RagAgent(
    pipe=get_pipeline(llm_folder),
    db=db,
)

if args.server:
    import uvicorn
    from .back.api import build_server

    app = build_server(agent)

    uvicorn.run(app)

else:
    if not args.explained:
        front = GradioFront(agent)
        front.launch(share=not args.local)
    else:
        tokenizer = AutoTokenizer.from_pretrained('/home/ensta/data/' + llm_folder)
        config = AutoConfig.from_pretrained('/home/ensta/data/' + llm_folder)

        perturber = LeaveOneOutPerturber()
        generator = SimpleGenerator()
        comparator = EmbeddingComparator()

        pipeline_xrag = XRAGPipeline(perturber, generator, comparator, tokenizer, agent, embedding)

        front = XAIConsoleFront(agent, pipeline_xrag)
        front.launch()