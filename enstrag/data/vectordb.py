from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from ..models import RagEmbeddings


def get_retriever(documents: Document, embeddings: RagEmbeddings):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100
    )

    splits = text_splitter.split_documents(documents)

    """
    text_splitter = SentenceTransformersTokenTextSplitter(
        model_name = "sentence-transformers/all-MiniLM-l6-v2",
        tokens_per_chunk = 200,
    )
    splits = text_splitter.split_documents(documents)
    """

    db = Chroma.from_documents(
        splits,
        embeddings
    )
    return db.as_retriever()