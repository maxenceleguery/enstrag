"""Defined the generator of the explained pipeline"""
from abc import abstractmethod
from ..rag import RagAgent
from typing import List


class Generator:
    @abstractmethod
    def generator(self, perturbations: List[str], agent: RagAgent) -> List[str]:
        ...


class SimpleGenerator(Generator):
    """This generator simply calls the LLM and retrieve the answer"""

    def generator(self, perturbations, agent):
        return [agent.prompt_llm(perturbation) for perturbation in perturbations]
