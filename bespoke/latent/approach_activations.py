"""GOAL RUN / H3: does LFM2.5's OWN activation representation decode juandi's taste better than
EmbeddingGemma's ~0.62 ceiling?

Every linear+nonlinear probe on EmbeddingGemma plateaus at ~0.62 (Exp 1/2). The last lever is a
different representation: the base model's internal hidden states (which we LoRA-adapt anyway). Holding
the SAME topic partition fixed, we extract mean-pooled final-layer activations for the clean
accept/reject interactions and run the same cross-topic LOTO probe.

  activations break 0.62  -> the embedding was the limit; activation-based signals are the path.
  activations also ~0.62  -> the taste signal is ~irreducible in available representations.

Run:  python -m bespoke.latent.approach_activations [--max-tokens 256] [--limit N]
"""
import numpy as np
from bespoke.latent.approach import leiden_clusters, _l2norm
from bespoke.latent.approach_probe import _loto, _mean_diff, _logistic, _rbf_svm


def main():
    import argparse
    import mlx.core as mx
    from mlx_lm import load
    from bespoke.config import config
    from bespoke.db.init import get_connection
    from bespoke.eval.signals import get_labeled_embeddings
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="subsample clean labeled for speed")
    args = ap.parse_args()

    conn = get_connection()
    ids, Xemb, y = get_labeled_embeddings(conn)
    info = {r["id"]: (r["content_type"], (r["user_message"] or ""), (r["assistant_response"] or ""))
            for r in conn.execute("SELECT id, content_type, user_message, assistant_response FROM interactions").fetchall()}
    conn.close()

    keep = np.array([info.get(int(i), ("clean","",""))[0] in ("agentic","clean") for i in ids]) & (y != -1)
    ids, Xemb, y = ids[keep], Xemb[keep], y[keep]
    if args.limit and len(ids) > args.limit:
        sel = np.random.default_rng(0).permutation(len(ids))[:args.limit]
        ids, Xemb, y = ids[sel], Xemb[sel], y[sel]
    print(f"clean labeled: {len(ids)} ({int((y==1).sum())} acc / {int((y==0).sum())} rej); extracting activations...")

    model, tok = load(str(config.base_model.training_model_path))

    A = np.zeros((len(ids), 2048), dtype=np.float32)
    for k, iid in enumerate(ids.tolist()):
        _, um, ar = info[int(iid)]
        toks = tok.encode(f"{um}\n{ar}")[: args.max_tokens]
        if not toks:
            toks = [tok.eos_token_id or 0]
        h = model.model(mx.array(toks)[None])      # [1, seq, 2048]
        v = h.mean(axis=1)[0].astype(mx.float32)
        mx.eval(v)
        A[k] = np.array(v, dtype=np.float32)
        if (k + 1) % 1000 == 0:
            print(f"  {k+1}/{len(ids)}")

    # topics from EmbeddingGemma (same territory as Exp 1/2), evaluate ACTIVATIONS within them
    Xe = _l2norm(Xemb.astype(np.float64))
    An = _l2norm(A.astype(np.float64))
    labels = leiden_clusters(Xe, resolution=args.resolution)
    qual = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if (y[idx] == 1).sum() >= 8 and (y[idx] == 0).sum() >= 8:
            qual[int(c)] = idx
    print(f"topics: {len(qual)} qualifying\n")

    print("cross-topic LOTO AUC (clean; same topic partition):")
    print(f"  EmbeddingGemma  mean-diff : {_loto(Xe, y, qual, _mean_diff):.3f}   (baseline ~0.62)")
    print(f"  EmbeddingGemma  logistic  : {_loto(Xe, y, qual, _logistic):.3f}")
    print(f"  ACTIVATIONS     mean-diff : {_loto(An, y, qual, _mean_diff):.3f}   <-- H3")
    print(f"  ACTIVATIONS     logistic  : {_loto(An, y, qual, _logistic):.3f}   <-- H3")
    print(f"  ACTIVATIONS     RBF SVM   : {_loto(An, y, qual, _rbf_svm):.3f}   <-- H3")


if __name__ == "__main__":
    main()
