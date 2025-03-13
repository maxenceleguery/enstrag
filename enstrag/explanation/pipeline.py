"""Define the whole explainable pipeline"""
from transformers import AutoTokenizer
from numpy import array, argsort, argpartition
from ..models import RagEmbedding
from ..rag.agent import RagAgent
from .perturber import Perturber
from .generate import Generator
from .compare import Comparator
from typing import Any
from abc import abstractmethod

class XRAGPipeline:
    """A generic class for explanations"""
    @abstractmethod
    def top_k_tokens(self, prompt: dict[str, object], tokenizer: AutoTokenizer) -> list[str]:
        ...


class PerturbationPipeline(XRAGPipeline):
    """A class to define the Explainable RAG Pipeline with perturbation"""

    def __init__(self, perturber: Perturber, generator: Generator, comparator: Comparator,
                 tokenizer: AutoTokenizer, agent:RagAgent, embedding: RagEmbedding):
        self.perturber = perturber
        self.generator = generator
        self.comparator = comparator
        self.tokenizer = tokenizer
        self.agent = agent
        self.embedding = embedding

    def top_k_tokens(self, prompt: dict[str, Any], k: int) -> list[str]:
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
    

class GradientPipeline(XRAGPipeline):
    """A class to define the Explainable RAG Pipeline with gradient"""

    def __init__(self, tokenizer: AutoTokenizer, agent:RagAgent, embedding: RagEmbedding):
        self.tokenizer = tokenizer
        self.model = agent.hf_pipeline.model
        self.embedding = embedding
    
    def top_k_tokens(self, prompt: dict[str, Any], k: int) -> list[str]:
        """Return the top k tokens that are the most influencial"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        embeddings = self.model.embeddings.word_embeddings(inputs['input_ids'])
        embeddings.retain_grad()

        outputs = self.model(inputs_embeds=embeddings)

        loss = outputs.last_hidden_state.sum()
        loss.backward()  

        gradients = embeddings.grad
        average_gradients = gradients[0].mean(dim=1).detach().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        scores = array(average_gradients)

        max_scores = argsort(scores)[-k:]
        # max_scores = argpartition(scores, -k)[-k:]

        return  [tokens[tk] for tk in max_scores]

