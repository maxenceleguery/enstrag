"""Utils function for the explainable part"""
from numpy import dot, array
from numpy.linalg import norm
from typing import List

def cosine_similarity(x: List[float], y: List[float]) -> float:
    """Compute the cosine similarity between two vectors"""
    a = array(x)
    b = array(y)
    return dot(a, b) / (norm(a) * norm(b))

def euclidian_similarity(x: List[float], y: List[float]) -> float:
    """Compute the cosine similarity between two vectors"""
    a = array(x)
    b = array(y)
    return norm(a-b)
