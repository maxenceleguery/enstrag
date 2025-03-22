import os
from transformers import pipeline
from .available_models import get_available_models

def get_pipeline(model_name: str):
    if model_name not in get_available_models():
        raise ValueError(f"{model_name} is not a valid model. Choose one from {os.environ['MODELS_PATH']} or ask to add one.")
    
    return pipeline(
        task="text-generation",
        model=os.path.join(os.environ["MODELS_PATH"], model_name),
        device=0,
        max_length=2048,
        truncation=True,
        batch_size=64,
    )