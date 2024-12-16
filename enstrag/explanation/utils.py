"""Utils function for the explainable part"""
from numpy import dot, array
from numpy.linalg import norm

def cosine_similarity(x: array, y: array):
    """Compute the cosine similarity between two vectors"""
    a = array(x)
    b = array(y)
    return dot(a, b) / (norm(a) * norm(b))
