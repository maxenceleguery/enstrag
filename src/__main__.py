from .rag import RagAgent
from .models import get_pipeline, RagEmbeddings
from .data import get_retriever, get_documents_from_file

agent = RagAgent(
    pipe=get_pipeline(),
    retriever=get_retriever(get_documents_from_file(), RagEmbeddings()),
)

while True:
    query = input("Enter the question(Type exit to close)\n")
    if query == "exit":
        break
    result, retrieved_context = agent.answer_question(query, verbose=True)