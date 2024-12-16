"""Defined the comparator of the explained pipeline"""
from abc import abstractmethod
from typing import Callable
from ..models import RagEmbedding
from .utils import cosine_similarity


class Comparator():
    @abstractmethod
    def compare(self, perturbed_answers: list[str], gold_answer: str,
                embedding: RagEmbedding, similarity: Callable = cosine_similarity) -> list[float]:
        ...


class EmbeddingComparator(Comparator):
    """Use the embbedding model used to build the Vector DB to compare answers"""

    def compare(self, perturbed_answers, gold_answer, embedding, similarity):
        """Compare the perturbated answers to the gold_answer but computing a similarity score"""
        return [similarity(embedding.embed_query(p_answer), embedding.embed_query(g_answer))
                for p_answer, g_answer in zip(perturbed_answers, gold_answer)]
