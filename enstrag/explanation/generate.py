"""Defined the generator of the explained pipeline"""
from abc import abstractmethod


class Generator:
    @abstractmethod
    def generator(self, perturbations: list[str], agent) -> list[str]:
        ...


class SimpleGenerator(Generator):
    """This generator simply calls the LLM and retrieve the answer"""

    def generator(self, perturbations, agent):
        answers = [outputs for outputs in agent.prompt_llm(perturbations)]
        return answers
