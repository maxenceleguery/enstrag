print("Importing packages...")
from .rag import RagAgent
from .models import get_pipeline, RagEmbedding
from .data import VectorDB, Parser

print("Main not ready yet")
exit(0)

llm_folder = ""
embedding_folder = ""
persist_directory="/tmp/enstrag"

db = VectorDB(RagEmbedding(embedding_folder), persist_directory=persist_directory)
db.add_documents(
    Parser.get_documents_from_pdf_url([
        "https://arxiv.org/pdf/1706.03762",
        "https://arxiv.org/pdf/2106.09685"
    ])
)

agent = RagAgent(
    pipe=get_pipeline(llm_folder),
    db=db,
)

while True:
    query = input("Enter the question(Type exit to close)\n")
    if query == "exit":
        break
    result, retrieved_context = agent.answer_question(query, verbose=True)