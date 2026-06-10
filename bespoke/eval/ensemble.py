"""Geometric eval ensemble orchestrator.

score_items: per eval response, fuse gate + propagation + probe + taste -> quality, aggregate to a
scorecard. compare_geometric: keep/revert from reward delta with a noise band (work through
geometric noise to stability instead of trusting a paid judge).
"""
import numpy as np

from bespoke.eval.gates import run_gates
from bespoke.eval.meta import MetaScorer
from bespoke.eval.taste_axes import taste_score


def score_items(items, probe=None, propagation_scores=None, meta=None):
    """items: list of {prompt, output, domain, embedding(np.ndarray 768)}.

    probe: fitted LinearPreferenceProbe (or None -> probe signal 0.5).
    propagation_scores: list aligned to items (or None -> 0.5 each).
    meta: fitted MetaScorer (or None -> rule fallback).
    Returns scorecard dict.
    """
    meta = meta or MetaScorer()
    n = len(items)
    prop = propagation_scores if propagation_scores is not None else [0.5] * n

    per_item = []
    for i, it in enumerate(items):
        gate = run_gates(it["prompt"], it.get("output", ""), it.get("domain"))
        emb = it.get("embedding")
        probe_score = 0.5
        if probe is not None and emb is not None:
            probe_score = float(probe.score(np.asarray(emb, np.float32).reshape(1, -1))[0])
        features = {
            "gate_passed": 1 if gate["passed"] else 0,
            "propagation": float(prop[i]),
            "probe": probe_score,
            "taste": taste_score(it.get("output", "")),
        }
        per_item.append({
            "quality": meta.quality(features),
            "gate_passed": features["gate_passed"],
            "propagation": features["propagation"],
            "probe": features["probe"],
            "taste": features["taste"],
        })

    reward = float(np.mean([p["quality"] for p in per_item])) if per_item else 0.0
    gate_pass_rate = float(np.mean([p["gate_passed"] for p in per_item])) if per_item else 0.0
    return {
        "eval_set_size": n,
        "reward": reward,
        "gate_pass_rate": gate_pass_rate,
        "per_item": per_item,
    }


def compare_geometric(current, previous, noise=0.01):
    """Keep if reward improved or dropped within the noise band; else revert."""
    cur = current.get("reward", 0.0)
    prev = previous.get("reward", 0.0)
    delta = cur - prev
    if delta >= -noise:
        decision = "keep"
        reasoning = f"Reward {cur:.3f} vs {prev:.3f} (delta {delta:+.3f}, within/above noise {noise})."
    else:
        decision = "revert"
        reasoning = f"Reward dropped {cur:.3f} vs {prev:.3f} (delta {delta:+.3f}, beyond noise {noise})."
    return {"decision": decision, "reasoning": reasoning, "delta": delta}
