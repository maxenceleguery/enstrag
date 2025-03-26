"""Define the whole explainable pipeline"""
from transformers import AutoTokenizer, Pipeline, Qwen2ForCausalLM
import torch
from numpy import array, argsort, argpartition
from ..models import RagEmbedding
from .perturber import Perturber
from .generate import Generator
from .compare import Comparator
from typing import Any
import gc
from abc import abstractmethod

class XRAGPipeline:
    """A generic class for explanations"""
    @abstractmethod
    def top_k_tokens(self, prompt: dict[str, object], tokenizer: AutoTokenizer) -> list[str]:
        ...


class PerturbationPipeline(XRAGPipeline):
    """A class to define the Explainable RAG Pipeline with perturbation"""

    def __init__(self, perturber: Perturber, generator: Generator, comparator: Comparator,
                 tokenizer: AutoTokenizer, agent, embedding: RagEmbedding):
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
        gold_answer = self.agent.prompt_llm([prompt])
        comparison_scores = self.comparator.compare(perturbated_answers, gold_answer, self.embedding)

        # If the comparison score is high, the token is influent
        array_scores = array(comparison_scores)
        k_better_tokens = argsort(array_scores)[-k:]

        # Get the influent tokens
        influent_str_tokens = [pertubated_tokens[tk] for tk in k_better_tokens]
        print(influent_str_tokens)
        return influent_str_tokens
    

class GradientPipeline(XRAGPipeline):
    """A class to define the Explainable RAG Pipeline with gradient"""

    def __init__(self, pipe: Pipeline, embedding: RagEmbedding, prompt_template):
        self.tokenizer = pipe.tokenizer
        self.model = pipe.model
        self.embedding = embedding
        self.prompt_template = prompt_template

   
    def top_k_tokens(self, prompt: dict[str, Any], k: int) -> list[str]:
        """Return the top k tokens that are the most influential for the generated output."""
        prompt = self.prompt_template.format(context=prompt["context"], question=prompt["question"])
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        embeddings = self.model.model.embed_tokens(inputs['input_ids'].to(self.model.device))
        embeddings.retain_grad()
        
        if True:
            outputs = self.model(inputs_embeds=embeddings, output_hidden_states=True)
            
            logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)
            
            # Get predicted token probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert logits to probabilities
            target_ids = input_ids[:, 1:]  # Shift left to align with next token predictions
            target_probs = torch.gather(probs[:, :-1, :], 2, target_ids.unsqueeze(-1)).squeeze(-1)
            
            loss = -torch.sum(torch.log(target_probs))        

        else:
            outputs = self.model(inputs_embeds=embeddings, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            loss = last_hidden_state.sum()
        
        loss.backward()
        
        gradients = embeddings.grad
        average_gradients = gradients[0].mean(dim=1).detach().cpu().numpy()

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = None

        del gradients, embeddings
        loss.detach()
        del loss
        gc.collect()
        torch.cuda.empty_cache()

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        scores = array(average_gradients)

        max_scores = argsort(scores)[-k:]

        influent_str_tokens = [tokens[tk] for tk in max_scores]
        print(influent_str_tokens)
        return influent_str_tokens

