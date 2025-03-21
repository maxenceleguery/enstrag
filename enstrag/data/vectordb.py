from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from ..models import RagEmbedding

from abc import ABC, abstractmethod
from typing import Literal, List

class DB(ABC):
    @abstractmethod
    def add_document(self, document: Document) -> None:
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass

    @abstractmethod
    def get_context_from_query(self, query:str, search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "similarity", topk: int = 4, fetch_k: int = 20 ) -> str:
        pass

class VectorDB(DB):
    def __init__(self, embedding: RagEmbedding, persist_directory: str = None):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100
        )

        """
        self.text_splitter = SentenceTransformersTokenTextSplitter(
            model_name = "sentence-transformers/all-MiniLM-l6-v2",
            tokens_per_chunk = 200,
        )
        splits = text_splitter.split_documents(documents)
        """
        self.embedding = embedding
        self.db = Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory,
        )
        self.themes = set()

    def add_document(self, document: Document) -> None:
        doc_hash = document.metadata["hash"]
        # Check if document is already in the database
        self.themes.add(document.metadata.get("label"))
        if len(self.db.get(
            where={"hash": doc_hash}
        )["documents"]) > 0:
            print(f"{document.metadata['name']} already in database. Ignoring...")
            return
        
        print(f"Adding {document.metadata['name']} in database...")
    
        splits = self.text_splitter.split_documents([document])
        self.db.add_documents(splits)

    def add_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            self.add_document(doc)

    def get_context_from_query(self, query: str, search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "similarity", topk: int = 4, fetch_k: int = 20 ) -> List[dict]:
        if search_type == "mmr":
            search_kwargs={'k': topk, 'fetch_k': fetch_k}
        else:
            search_kwargs={'k': topk}

        contexts = self.db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        ).invoke(query)
        assert len(contexts) == topk, f"{len(contexts)} != {topk}"

        chunks = []
        for ctx in contexts:
            chunks.append(
                {
                    "text" : ctx.page_content,
                    "name" : ctx.metadata.get("name"),
                    "url" : ctx.metadata.get("url"),
                    "path" : ctx.metadata.get("path"),
                }
            )

        return chunks
