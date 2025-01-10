import os
from typing import List

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

from .available_models import get_available_models

class RagEmbedding(Embeddings):
    def __init__(self, model_name):
        if model_name not in get_available_models():
            raise ValueError(f"{model_name} is not a valid model. Choose one from {os.environ['MODELS_PATH']} or ask to add one.")

        self.model = SentenceTransformer(os.path.join(os.environ["MODELS_PATH"], model_name), trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()