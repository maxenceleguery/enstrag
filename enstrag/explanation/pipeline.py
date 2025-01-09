"""Define the whole explainable pipeline"""
from transformers import AutoTokenizer
from numpy import array, argsort
from ..models import RagEmbedding
from .perturber import Perturber
from .generate import Generator
from .compare import Comparator
from ..rag import RagAgent
from typing import List, Dict, Any


class XRAGPipeline:
    """A class to define the Explainable RAG Pipeline"""

    def __init__(self, perturber: Perturber, generator: Generator, comparator: Comparator,
                 tokenizer: AutoTokenizer, agent: RagAgent, embedding: RagEmbedding):
        self.perturber = perturber
        self.generator = generator
        self.comparator = comparator
        self.tokenizer = tokenizer
        self.agent = agent
        self.embedding = embedding

    def top_k_tokens(self, prompt: Dict[str, Any], gold_answer: str, k: int) -> List[str]:
        """Return the top k tokens that are the most influencial"""
        perturbed_prompts = self.perturber.perturb(prompt, self.tokenizer)
        perturbed_answers = self.generator.generator(perturbed_prompts, self.agent)
        comparison_scores = self.comparator.compare(perturbed_answers, gold_answer, self.embedding)

        # If the comparison scores are low, the token is influent
        array_scores = array(comparison_scores)
        k_better_tokens = argsort(array_scores)[:k]

        # Get the influent tokens
        tokens = self.tokenizer(prompt["context"])
        influent_tokens = tokens["input_ids"][k_better_tokens]
        influent_str_tokens = self.tokenizer.decode(influent_tokens, skip_special_tokens=True)

        print("Influent tokens:", influent_str_tokens)
        return influent_str_tokens
