import os
from typing import List

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_core.embeddings import Embeddings

def get_available_models():
    available = []
    for folder in os.listdir(os.environ["MODELS_PATH"]):
        if os.path.exists(os.path.join(os.environ["MODELS_PATH"], folder, "config.json")):
            available.append(folder)
    return available

class RagEmbedding(Embeddings):
        def __init__(self, model_name):
            if model_name not in get_available_models():
                raise ValueError(f"{model_name} is not a valid model. Choose one from {os.environ['MODELS_PATH']} or ask to add one.")

            self.model = SentenceTransformer(os.path.join(os.environ["MODELS_PATH"], model_name), trust_remote_code=True)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self.model.encode(texts).tolist()

        def embed_query(self, query: str) -> List[float]:
            return self.model.encode(query).tolist()

def get_pipeline(model_name: str):
    if model_name not in get_available_models():
        raise ValueError(f"{model_name} is not a valid model. Choose one from {os.environ['MODELS_PATH']} or ask to add one.")
    
    return pipeline(
        task="text-generation",
        model=os.path.join(os.environ["MODELS_PATH"], model_name),
        device=0,
        max_length=1024,
        truncation=True
    )

