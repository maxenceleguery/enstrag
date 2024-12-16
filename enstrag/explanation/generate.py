"""Defined the generator of the explained pipeline"""
from abc import abstractmethod
from ..rag import RagAgent


class Generator():
    @abstractmethod
    def generator(self, perturbations: list[str], agent: RagAgent) -> list[str]:
        ...


class SimpleGenerator(Generator):
    """This generator simply calls the LLM and retrieve the answer"""

    def generator(self, perturbations, agent):
        return [agent.prompt_llm(perturbation) for perturbation in perturbations]
