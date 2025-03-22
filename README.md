# ENSTA RAG PROJECT

## Installation

```bash
conda env update --name <your env> --file rag_env.yaml
python3 -m pip install .
```

For testing :
```bash
python3 -m pip install .[test]
```

```bash
srun --pty --time=00:30:00 --partition=ENSTA-h100 --gpus=1 bash
conda activate <your env>
python3 -m enstrag
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
