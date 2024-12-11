# ENSTA RAG PROJECT

## Installation

```bash
python3 -m pip install .
```

```bash
srun --pty --time=00:30:00 --partition=ENSTA-h100 --gpus=1 bash
conda activate <your env>
python3 -m enstrag
```