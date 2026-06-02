"""Linear preference probe — a logistic regression that learns the user's 'taste direction'.

Model-agnostic on the FEATURE vector: feed EmbeddingGemma embeddings now; swap in LFM2.5
hidden-state activations later (richer, more quality-aware) without changing this interface.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


class LinearPreferenceProbe:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        """X: (n, d) features. y: (n,) in {0, 1}."""
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model.fit(X, y)
        return self

    def score(self, X):
        """Return P(accept) in [0, 1] for each row of X."""
        classes = list(self.model.classes_)
        if 1 not in classes:
            return np.zeros(X.shape[0], dtype=float)
        return self.model.predict_proba(X)[:, classes.index(1)]
