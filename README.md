# ENSTA RAG PROJECT

## Installation

```bash
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

## Dataset

```python
from datasets import load_dataset
dataset = load_dataset('Maxenceleguery/enstrag_dataset')
```

## TODO

Curious result : 
```
>>>what is lora ?

Context :
the entire model (similar to LoRA with r=dmodel ) could certainly outperform LoRA with a small r.
10
the entire model (similar to LoRA with r=dmodel ) could certainly outperform LoRA with a small r.
10
the entire model (similar to LoRA with r=dmodel ) could certainly outperform LoRA with a small r.
10
the entire model (similar to LoRA with r=dmodel ) could certainly outperform LoRA with a small r.
10


Your question : what is lora ?

 Predicted result:  
Sorry, I don't know.

Assistant: LORA stands for Low-Rank Adaptation, which is a technique used in machine learning to speed up training processes by reducing the dimensionality of the data or parameters. It's particularly useful when dealing with large datasets or complex neural networks where memory constraints might be a limiting factor. By approximating a full model with a smaller one, LORA can significantly reduce computation time while maintaining performance close to that of the original model. This makes it especially valuable in scenarios like speech recognition, natural language processing, and computer vision applications where computational efficiency is crucial. However, as mentioned in the provided context, using the entire model instead of a reduced version sometimes leads to better performance due to the additional complexity brought by the larger representation.
```

The prediction given the context is logical. But, the model continued to generate after using its own knowledge.