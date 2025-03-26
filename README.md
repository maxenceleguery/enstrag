# ENSTA RAG PROJECT

## Installation

```bash
git clone https://github.com/maxenceleguery/enstrag.git
cd enstrag
conda env update --name <your env> --file rag_env.yaml
python3 -m pip install .
```

For testing :
```bash
python3 -m pip install .[test]
```

### Without Docker
```bash
conda activate <your env>
python3 -m enstrag
```

### With Docker
 ```bash
docker compose up [-d]
```

For evaluation :
```bash
python3 -m enstrag.metrics.evaluation
```

## Dataset

```python
from datasets import load_dataset
dataset = load_dataset('Maxenceleguery/enstrag_dataset')
```

## Development

This repository is a python module to build a RAG application.<br>
When importing the module, the environment variables MODELS_PATH and PERSIST_PATH will be set to default. Feel free to override those to fit your use case. The module will only look to Huggingface models stored in MODELS_PATH directory.

```python
from enstrag.rag import RagAgent
from enstrag.models import get_pipeline, RagEmbedding
from enstrag.data import VectorDB, Parser, FileDocument
from enstrag.front import GradioFront, XAIConsoleFront

embedding = RagEmbedding("path/to/embedding/model")
db = VectorDB(embedding, persist_directory="...")
db.add_documents(
    Parser.get_documents_from_filedocs([
        FileDocument(pdf_url, None, name, label)
    ])
)

agent = RagAgent(
    pipe=get_pipeline("path/to/llm/model"),
    db=db,
)

front = GradioFront(agent) # or XAIConsoleFront
front.launch()
```

# ENSTA 2025 (CC BY-NC-SA 4.0)
Maxence Leguéry, Antoine Domingues, Albin Joyeux, Mattéo Denis, Simon Zarka