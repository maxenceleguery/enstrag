from .raptor.cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig
from .raptor import BaseSummarizationModel, BaseEmbeddingModel
from ..models import get_pipeline
from .vectordb import DB

from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from typing import Literal, List
import os


class SummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name="Qwen2.5-1.5B-Instruct"):
        # Initialize the tokenizer and the pipeline for the GEMMA model
        self.summarization_pipeline = get_pipeline(model_name)
        self.tokenizer = self.summarization_pipeline.tokenizer

    def summarize(self, context, max_tokens=150):
        # Format the prompt for summarization
        messages=[
            {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate the summary using the pipeline
        outputs = self.summarization_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        # Extracting and returning the generated summary
        summary = outputs[0]["generated_text"].strip()
        return summary

class EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(os.path.join(os.environ["MODELS_PATH"], model_name), trust_remote_code=True)

    def create_embedding(self, text):
        return self.model.encode(text)


class RAPTORVectorDB(DB):
    def __init__(self):
        super().__init__()

        self.tree = None
        self.tree_builder = ClusterTreeBuilder(ClusterTreeConfig(
            summarization_model = SummarizationModel(),
            embedding_models = {"minilm": EmbeddingModel()},
            cluster_embedding_model = "minilm",
            max_tokens = 512,
            clustering_params = {"verbose": True},
        ))

    def add_document(self, document: Document) -> None:
        if self.tree is not None:
            user_input = input(
                "Warning: Overwriting existing tree. Did you mean to call 'add_to_existing' instead? (y/n): "
            )
            if user_input.lower() == "y":
                # self.add_to_existing(docs)
                return

        self.tree = self.tree_builder.build_from_text(text=document.page_content)
        self.retriever = TreeRetriever(TreeRetrieverConfig(), self.tree)

    def add_documents(self, documents: List[Document]) -> None:
        if self.tree is not None:
            user_input = input(
                "Warning: Overwriting existing tree. Did you mean to call 'add_to_existing' instead? (y/n): "
            )
            if user_input.lower() == "y":
                # self.add_to_existing(docs)
                return

        self.tree = self.tree_builder.build_from_text(text=" ".join(list(map(lambda doc: doc.page_content, documents))))
        self.retriever = TreeRetriever(TreeRetrieverConfig(), self.tree)

    def get_context_from_query(self, query: str, search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "similarity", topk: int = 4, fetch_k: int = 20 ) -> str:
        return self.retriever.retrieve(query, top_k=topk)