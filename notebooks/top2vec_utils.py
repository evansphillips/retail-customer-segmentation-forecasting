from typing import List
import math
import numpy as np
from itertools import combinations
from collections import Counter
from top2vec import Top2Vec
from gensim.utils import simple_preprocess

def compute_cv_coherence(top2vec_model: Top2Vec, documents: List[str]) -> float:
    """
    Calculate C_V coherence for a given Top2Vec model and tokenized documents.

    Parameters:
    - top2vec_model (Top2Vec): Top2Vec model.
    - documents (list): List of tokenized documents.

    Returns:
    - float: C_V coherence score.
    """

    # Get word vectors from the Top2Vec model
    word_vectors = top2vec_model.word_vectors

    # Generate all possible word pairs
    word_pairs = list(combinations(top2vec_model.vocab, 2))

    # Initialize variable for C_V coherence calculation
    cv_sum = 0.0
    num_pairs = len(word_pairs)

    # Calculate C_V coherence for each word pair
    for pair in word_pairs:
        if pair[0] in word_vectors and pair[1] in word_vectors:
            # Calculate the cosine similarity between vectors of word1 and word2
            cos_similarity = np.dot(word_vectors[pair[0]], word_vectors[pair[1]]) / (
                    np.linalg.norm(word_vectors[pair[0]]) * np.linalg.norm(word_vectors[pair[1]]) + 1e-12
            )
            # Update the coherence score
            cv_sum += cos_similarity

    # Calculate the final C_V coherence score
    cv_coherence = cv_sum / num_pairs if num_pairs > 0 else 0.0
    return cv_coherence

