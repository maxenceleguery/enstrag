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

    def top_k_tokens(self, prompt: Dict[str, Any], k: int) -> List[str]:
        """Return the top k tokens that are the most influencial"""
        print("Perturbing the prompt...")
        perturbations  = self.perturber.perturb(prompt, self.tokenizer)
        pertubated_tokens = [perturbation[0] for perturbation in perturbations]
        perturbated_prompts = [perturbation[1] for perturbation in perturbations]
        print("Generating perturbated answers...")
        perturbated_answers = self.generator.generator(perturbated_prompts, self.agent)
        print("Comparing to the original answer...")
        gold_answer = self.agent.prompt_llm(prompt)
        comparison_scores = self.comparator.compare(perturbated_answers, gold_answer, self.embedding)

        # If the comparison score is high, the token is influent
        array_scores = array(comparison_scores)
        k_better_tokens = argsort(array_scores)[-k:]

        # Get the influent tokens
        influent_str_tokens = [pertubated_tokens[tk] for tk in k_better_tokens]

        return influent_str_tokens
