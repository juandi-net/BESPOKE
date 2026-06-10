"""Mechanical taste-axis extractors (RT-006) — juandi's curation whys as cheap, deterministic signals.

RT-001..003 showed the single embedding axis is a ~0.62-0.69 DATA ceiling: accept/reject is barely
separable as one blurry direction. juandi's curation whys, though, name *specific, mechanical* tells —
"dont like the emojis", "dont need the absolutely right", "too verbose", "tries to deflect back to me",
"should have lead with the answer", "should have just done it". Each is a measurable axis. This module
turns them into a small interpretable feature vector (no LLM, no cloud) — the seed of a scalable
taste-specialist judge (RT-004's conclusion: a generic judge is miscalibrated 2x; we need one fit to him).

`python -m bespoke.eval.taste_axes` runs the RT-006 validation against ~/.bespoke curation.json:
does axes-only / axes+embedding beat the 0.69 single-axis ceiling, and which axes carry the signal.
"""
import re

# Emoji: pictographic + symbol/dingbat ranges + regional indicators + variation selectors.
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F000-\U0001F0FF"   # pictographs, symbols, supplemental
    "\U00002600-\U000027BF"                          # misc symbols + dingbats (✅ ✨ ⚡ ❌)
    "\U00002B00-\U00002BFF\U0001F1E6-\U0001F1FF]"    # arrows/stars, regional indicators
)

# Sycophancy / flattery openers and affirmations juandi explicitly rejects.
_SYCOPHANCY = [
    "great question", "good question", "excellent question", "absolutely right",
    "you're absolutely right", "you are absolutely right", "you're right", "you are right",
    "great point", "good point", "great choice", "great idea", "that's a great",
    "happy to help", "i'd be happy", "i would be happy", "glad to", "i'd love to",
    "excellent", "fantastic", "wonderful", "love it", "perfect choice",
]
_SYCO_RE = [re.compile(re.escape(p)) for p in _SYCOPHANCY]

# Hedging / deflection: punts the decision back to the user instead of answering.
_HEDGE = [
    "it depends", "depends on", "what would you like", "do you want", "would you like",
    "let me know", "feel free", "you might want", "you may want", "if you'd like",
    "if you want", "should i", "want me to", "up to you", "your call",
    "there are several options", "there are a few options",
]
_HEDGE_RE = [re.compile(re.escape(p)) for p in _HEDGE]

# Filler / weasel words — "added words that didnt matter", "too verbose".
_FILLER = [
    "i think", "i believe", "basically", "simply", "actually", "really",
    "very", "maybe", "perhaps", "essentially", "kind of", "sort of",
    "in order to", "it's worth noting", "it is worth noting", "probably", "somewhat",
    "just",
]
_FILLER_RE = [re.compile(r"\b" + re.escape(p) + r"\b") for p in _FILLER]

# Pleasantry openers — for "should have lead with the answer".
_PLEASANTRY_OPENERS = (
    "great question", "good question", "excellent question", "sure", "certainly",
    "absolutely", "of course", "happy to", "i'd be happy", "i would be happy",
    "thanks", "thank you", "great", "no problem", "got it", "that's a great",
    "i can help", "i'd love to",
)


def emoji_count(text):
    return len(_EMOJI.findall(text or ""))


def _count(text, patterns):
    low = (text or "").lower()
    return sum(len(p.findall(low)) for p in patterns)


def sycophancy_score(text):
    return _count(text, _SYCO_RE)


def hedge_score(text):
    return _count(text, _HEDGE_RE)


def filler_count(text):
    return _count(text, _FILLER_RE)


def exclamation_count(text):
    return (text or "").count("!")


def leading_pleasantry(text):
    """1 if the response opens with a pleasantry instead of leading with the answer."""
    low = (text or "").lstrip().lower()
    return 1 if low.startswith(_PLEASANTRY_OPENERS) else 0


def has_code(text):
    """1 if the response ships a concrete artifact (fenced code block) — the 'just do it' tell."""
    return 1 if "```" in (text or "") else 0


# Column order for the feature matrix — keep stable (drives RT-006 + any downstream scorer).
FEATURE_NAMES = [
    "emoji_count", "sycophancy_score", "exclamation_count", "hedge_score",
    "leading_pleasantry", "filler_count", "has_code", "length",
]


def taste_features(text):
    """All mechanical taste axes for one response, as a dict keyed by FEATURE_NAMES."""
    return {
        "emoji_count": emoji_count(text),
        "sycophancy_score": sycophancy_score(text),
        "exclamation_count": exclamation_count(text),
        "hedge_score": hedge_score(text),
        "leading_pleasantry": leading_pleasantry(text),
        "filler_count": filler_count(text),
        "has_code": has_code(text),
        "length": len(text or ""),
    }


def feature_vector(text):
    """Feature dict flattened to a list in FEATURE_NAMES order."""
    f = taste_features(text)
    return [f[k] for k in FEATURE_NAMES]


def taste_score(text):
    """Fuse the mechanical axes into one quality in [0,1] — 1.0 = none of juandi's negative tells.

    Penalty weights follow RT-006's per-axis separation on his curated labels: emoji and exclamation
    are the cleanest drop-tells (his keeps had exactly zero of both), then deflection/sycophancy. This
    is the intrinsic, no-LLM taste check the faint geometric probe couldn't be — it scores a single
    fresh response, deterministically, with no cloud call.
    """
    f = taste_features(text)
    penalty = (
        0.50 * (1 if f["emoji_count"] > 0 else 0)
        + 0.30 * min(f["exclamation_count"], 3) / 3
        + 0.30 * min(f["sycophancy_score"], 2) / 2
        + 0.20 * min(f["hedge_score"], 2) / 2
        + 0.10 * f["leading_pleasantry"]
    )
    return round(max(0.0, 1.0 - penalty), 4)


# --------------------------------------------------------------------------------------
# RT-006 validation — run the axes against juandi's real curated keep/drop labels.
# Question: do interpretable mechanical axes match/beat the 0.69 single-axis embedding
# ceiling (RT-005), and WHICH axes carry his taste? Pure measurement, no LLM.
# --------------------------------------------------------------------------------------
def _load_curation():
    import json
    from bespoke.benchmark.curate import _FILE
    a = json.loads(_FILE.read_text())
    keep = [it for it in a["items"] if it.get("verdict") == "keep"]
    drop = [it for it in a["items"] if it.get("verdict") == "drop"]
    return keep, drop


def _embeddings_for(ids):
    import numpy as np
    from bespoke.db.init import get_connection
    if not ids:
        return {}
    conn = get_connection()
    q = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT i.id AS id, v.interaction_embedding AS emb FROM interactions i "
        f"JOIN vec_interactions v ON v.rowid = i.id WHERE i.id IN ({q})", list(ids)).fetchall()
    conn.close()
    return {r["id"]: np.frombuffer(r["emb"], dtype=np.float32) for r in rows}


def _cv_auc(X, y):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    cv = max(2, min(5, int(np.sum(y == 1)), int(np.sum(y == 0))))
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000, class_weight="balanced"))
    return float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()), cv


def rt006_report():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    keep, drop = _load_curation()
    if len(keep) < 3 or len(drop) < 3:
        print(f"need >=3 keep AND >=3 drop (have {len(keep)} / {len(drop)})")
        return

    items = keep + drop
    y = np.array([1] * len(keep) + [0] * len(drop))
    A = np.array([feature_vector(it.get("response") or "") for it in items], dtype=np.float64)

    print("=" * 72)
    print(f"RT-006 — mechanical taste axes vs the 0.69 embedding ceiling")
    print(f"  {len(keep)} keep / {len(drop)} drop curated labels")
    print("=" * 72)

    # --- per-axis univariate separation (robust on small n; the interpretable headline) ---
    print("\nPer-axis univariate AUC (predicting KEEP). <0.50 ⇒ axis flags DROPs:")
    print(f"  {'axis':<20} {'AUC':>6}  {'keep_mean':>10} {'drop_mean':>10}  direction")
    order = []
    for j, name in enumerate(FEATURE_NAMES):
        col = A[:, j]
        try:
            auc = roc_auc_score(y, col)
        except ValueError:
            auc = 0.5
        sep = abs(auc - 0.5)
        km, dm = col[y == 1].mean(), col[y == 0].mean()
        arrow = "→drop" if auc < 0.5 else ("→keep" if auc > 0.5 else "—")
        order.append((sep, name, auc, km, dm, arrow))
    for sep, name, auc, km, dm, arrow in sorted(order, reverse=True):
        print(f"  {name:<20} {auc:>6.2f}  {km:>10.2f} {dm:>10.2f}  {arrow} ({sep:+.2f} from chance)")

    # --- combined models ---
    print("\nCross-validated logistic AUC (the eval-grade number):")
    auc_axes, cv = _cv_auc(A, y)
    print(f"  axes-only ({len(FEATURE_NAMES)} features)         AUC {auc_axes:.3f}  (cv={cv})")

    em = _embeddings_for([it["id"] for it in items])
    have = [i for i, it in enumerate(items) if it["id"] in em]
    auc_emb = auc_combo = None
    if len(have) == len(items) and len(have) >= 6:
        E = np.vstack([em[items[i]["id"]] for i in range(len(items))]).astype(np.float64)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        auc_emb, _ = _cv_auc(E, y)
        auc_combo, _ = _cv_auc(np.hstack([A, E]), y)
        print(f"  embedding-only (768-d, RT-005 repro)  AUC {auc_emb:.3f}")
        print(f"  axes + embedding                      AUC {auc_combo:.3f}")
    else:
        print(f"  (embedding compare skipped — {len(have)}/{len(items)} items have vectors)")

    # --- verdict ---
    print("\n" + "-" * 72)
    best = max(x for x in [auc_axes, auc_emb, auc_combo] if x is not None)
    if auc_emb is not None:
        if auc_axes >= auc_emb + 0.03:
            print(f"VERDICT: axes ({auc_axes:.2f}) BEAT the embedding ceiling ({auc_emb:.2f}) — "
                  "interpretable mechanical taste > one blurry geometric axis.")
        elif auc_combo is not None and auc_combo >= auc_emb + 0.03:
            print(f"VERDICT: axes ADD to the embedding ({auc_emb:.2f}→{auc_combo:.2f} combined) — "
                  "orthogonal signal beyond geometry.")
        else:
            print(f"VERDICT: axes ({auc_axes:.2f}) ≈ embedding ({auc_emb:.2f}); best {best:.2f}. "
                  "Headline = WHICH axes carry his taste (above), not the combined number on n=40.")
    print("Caveat: n is small — the per-axis table is the durable result; CV AUCs have wide CIs.")
    print("-" * 72)
    return {"auc_axes": auc_axes, "auc_emb": auc_emb, "auc_combo": auc_combo}


if __name__ == "__main__":
    rt006_report()
