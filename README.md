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

For evaluation, add the dataset to the main folder and go to enstrag folder :
```bash
python3 -m enstrag.metrics.chunks_evaluation
```