from .rag import RagAgent
from .models import get_pipeline, RagEmbeddings
from .data import VectorDB, Parser

print("Main not ready yet")
exit(0)

db = VectorDB(RagEmbeddings(), persist_directory="/tmp/enstrag")
db.add_documents(
    Parser.get_documents_from_pdf_url([
        "https://arxiv.org/pdf/1706.03762",
        "https://arxiv.org/pdf/2106.09685"
    ])
)

agent = RagAgent(
    pipe=get_pipeline(),
    db=VectorDB(
        
        RagEmbeddings()
    ),
)

while True:
    query = input("Enter the question(Type exit to close)\n")
    if query == "exit":
        break
    result, retrieved_context = agent.answer_question(query, verbose=True)