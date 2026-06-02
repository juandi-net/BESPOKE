"""Mechanical training-pair extraction — the part of Stage 2a that never needed an LLM.

The SFT pair is already in the transcript: instruction = user_message, response =
assistant_response. We COPY it (no generation), drop rejected answers, and attach a geometric
quality score (Track 1 probe). Only genuinely tangled multi-turn sessions are routed to a
(future, local) distiller — `is_tangled` flags them.
"""
import numpy as np

from bespoke.extract.conversations import trajectory_features, _cosine_dist

REJECT = {"reject", "strong_reject"}


def extract_sft_pairs(turns, probe=None, domain_cluster=None):
    """Copy each non-rejected turn into an SFT pair with a geometric quality score.

    Returns list of {instruction, response, quality, domain_cluster}.
    quality = probe P(accept) on the turn embedding if a fitted probe is given, else 0.5.
    """
    pairs = []
    for t in turns:
        if t.get("feedback_class") in REJECT:
            continue
        instr = (t.get("user_message") or "").strip()
        resp = (t.get("assistant_response") or "").strip()
        if not instr or not resp:
            continue
        quality = 0.5
        emb = t.get("emb")
        if probe is not None and emb is not None:
            quality = float(probe.score(np.asarray(emb, np.float32).reshape(1, -1))[0])
        pairs.append({"instruction": instr, "response": resp,
                      "quality": quality, "domain_cluster": domain_cluster})
    return pairs


def _mean_consecutive_dist(turns):
    embs = [t.get("emb") for t in turns]
    dists = [_cosine_dist(embs[i - 1], embs[i]) for i in range(1, len(embs))]
    return float(np.mean(dists)) if dists else 0.0


def is_tangled(turns, min_turns=8, topic_var=0.5):
    """Route decision: True => send this conversation to a (local) distiller, not mechanical
    extraction. Heuristic — long-and-unresolved, or high topic churn within the conversation.
    Thresholds get tuned on real data at the Phase 5 gate.
    """
    f = trajectory_features(turns)
    if f["n_turns"] >= min_turns and not f["ended_in_accept"]:
        return True
    if f["n_turns"] >= 4 and _mean_consecutive_dist(turns) > topic_var:
        return True
    return False
