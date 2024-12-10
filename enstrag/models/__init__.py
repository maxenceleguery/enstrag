import os
from typing import List

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_core.embeddings import Embeddings

MODELS_PATH = "/home/ensta/data/"

"""
TO ADD :
sentence-transformers/all-MiniLM-l6-v2
Qwen/Qwen2.5-1.5B-Instruct

"""

class RagEmbeddings(Embeddings):
        def __init__(self, model_name):
            if model_name not in os.listdir(MODELS_PATH):
                raise ValueError(f"{model_name} is not a valid model. Choose one from /home/ensta/data or ask to add one.")

            self.model = SentenceTransformer(model_name, trust_remote_code=True)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [self.model.encode(t).tolist() for t in texts]

        def embed_query(self, query: str) -> List[float]:
            return self.model.encode([query])[0].tolist()

def get_pipeline(model_name: str):
    if model_name not in os.listdir(MODELS_PATH):
        raise ValueError(f"{model_name} is not a valid model. Choose one from /home/ensta/data or ask to add one.")
    
    return pipeline(
        model=os.path.join(MODELS_PATH, model_name),
        device=0,
        max_length=1024,
        truncation=True
    )

