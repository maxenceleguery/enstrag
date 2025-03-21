"""Defined the comparator of the explained pipeline"""
from abc import abstractmethod
from typing import Callable, List
from ..models import RagEmbedding
from .utils import cosine_similarity


class Comparator:
    @abstractmethod
    def compare(self, perturbed_answers: List[str], gold_answer: str,
                embedding: RagEmbedding, similarity: Callable = cosine_similarity) -> List[float]:
        ...


class EmbeddingComparator(Comparator):
    """Use the embbedding model used to build the Vector DB to compare answers"""

    def compare(self, perturbated_answers, gold_answer, embedding, similarity=cosine_similarity):
        """Compare the perturbated answers to the gold_answer but computing a similarity score"""
        comparison_scores =  [1 - similarity(embedding.embed_query(p_answer), embedding.embed_query(g_answer))
                for p_answer, g_answer in zip(perturbated_answers, gold_answer[0])]
        max_value = max(comparison_scores)
        return list(map(lambda x: x/max_value, comparison_scores))
