"""Defined the perturber of the explained pipeline"""
from copy import deepcopy
from abc import abstractmethod
from transformers import AutoTokenizer


class Perturber():
    @abstractmethod
    def perturb(self, prompt: dict[str], tokenizer: AutoTokenizer) -> list[str]:
        ...


class LeaveOneOutPerturber(Perturber):
    """Perturber that removes one token from the context's prompt"""

    def perturb(self, prompt, tokenizer):
        """Generate pertubations by removing one token"""
        context_tokens = tokenizer(prompt["context"])
        perturbations = []
        n_tokens = len(context_tokens["input_ids"])
        for i in range(n_tokens):
            tmp_tokens = deepcopy(context_tokens["input_ids"])
            # We remove the ith token's context
            tmp_tokens.pop(i)
            # Decode the modified context
            new_context = tokenizer.decode(tmp_tokens, skip_special_tokens=True)
            perturbations.append({"context": new_context, "question": prompt["question"]})
        return perturbations
