"""Defined the perturber of the explained pipeline"""
from copy import deepcopy
from abc import abstractmethod
from transformers import AutoTokenizer

import spacy

class Perturber:
    @abstractmethod
    def perturb(self, prompt: dict[str, object], tokenizer: AutoTokenizer) -> list[str]:
        ...


class LeaveOneOutPerturber(Perturber):
    """Perturber that removes one token from the prompt"""

    def perturb(self, prompt, tokenizer):
        """Generate pertubations by removing one token"""
        perturbations: list[tuple[str, dict[str, str]]] = []

        # Context perturber
        context_tokens = tokenizer(prompt["context"])
        n_tokens = len(context_tokens["input_ids"])
        for i in range(n_tokens):
            tmp_tokens = deepcopy(context_tokens["input_ids"])
            # We remove the ith token's context
            removed_token = tmp_tokens.pop(i)
            # Decode the modified context and token
            new_context = tokenizer.decode(tmp_tokens, skip_special_tokens=True)
            perturbated_token = tokenizer.decode(removed_token, skip_special_tokens=True)
            perturbations.append((perturbated_token, {"context": new_context, "question": prompt["question"]}))

        # Question perturber
        question_tokens = tokenizer(prompt["question"])
        n_tokens = len(question_tokens["input_ids"])
        for i in range(n_tokens):
            tmp_tokens = deepcopy(question_tokens["input_ids"])
            # We remove the ith token's question
            removed_token = tmp_tokens.pop(i)
            # Decode the modified question and token
            new_question = tokenizer.decode(tmp_tokens, skip_special_tokens=True)
            perturbated_token = tokenizer.decode(removed_token, skip_special_tokens=True)
            perturbations.append((perturbated_token, {"context": prompt["context"], "question": new_question}))
        return perturbations
    
class LeaveNounsOutPerturber(Perturber):
    """Perturber that removes tokens related to nouns from the prompt"""

    def perturb(self, prompt, tokenizer):
        """Generate pertubations by removing token's noun"""
        perturbations: list[tuple[str, dict[str, str]]] = []

        spacy.prefer_gpu()
        nlp = spacy.load('en_core_web_sm')

        doc_context = nlp(prompt["context"])
        doc_question = nlp(prompt["question"])
        context_nouns: list[str] = []
        question_nouns: list[str] = []
        for np in doc_context.noun_chunks:
            context_nouns.extend([token.text for token in np])
        for np in doc_question.noun_chunks:
            question_nouns.extend([token.text for token in np])
        
        # Avoid identical perturbations
        context_nouns = set(context_nouns)
        question_nouns = set(question_nouns)
        # Context perturber
        for noun in context_nouns:
            new_context = prompt["context"].replace(noun, "")
            perturbations.append((noun, {"context": new_context, "question": prompt["question"]}))
        
        # Question perturber
        for noun in question_nouns:
            new_question = prompt["question"].replace(noun, "")
            perturbations.append((noun, {"context": prompt["context"], "question": new_question}))

        return perturbations
