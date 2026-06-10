"""Meta-scorer: fuse orthogonal per-response signals into one quality in [0, 1].

Gate is a hard multiplier (fail => 0). When a trained logistic model exists it fuses
[gate, propagation, probe, taste]; otherwise falls back to the mean of the geometric signals.
`taste` is the mechanical taste-axis fusion (RT-006, bespoke.eval.taste_axes.taste_score) — an
orthogonal, no-LLM signal that catches juandi's negative tells (emoji/exclamation/deflection) the
faint geometric probe misses. The fusion model itself trains on accept/reject labels later — no LLM.
"""
import numpy as np

FEATURE_ORDER = ("gate_passed", "propagation", "probe", "taste")
_GEOMETRIC = ("propagation", "probe", "taste")


class MetaScorer:
    def __init__(self):
        self.model = None

    def fit(self, F, y):
        """F: (n, 3) features in FEATURE_ORDER. y: (n,) in {0,1}."""
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(max_iter=1000).fit(F, y)
        return self

    def quality(self, features):
        """features: dict with keys FEATURE_ORDER. Returns quality in [0,1]."""
        if not features.get("gate_passed"):
            return 0.0
        if self.model is not None:
            f = np.array([[float(features[k]) for k in FEATURE_ORDER]])
            classes = list(self.model.classes_)
            if 1 in classes:
                return float(self.model.predict_proba(f)[0, classes.index(1)])
        return float(np.mean([features[k] for k in _GEOMETRIC]))
