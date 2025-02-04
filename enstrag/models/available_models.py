import os

def get_available_models():
    available = []
    for folder in os.listdir(os.environ["MODELS_PATH"]):
        if os.path.exists(os.path.join(os.environ["MODELS_PATH"], folder, "config.json")):
            available.append(folder)
    return available