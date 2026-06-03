"""Top-K Sparse AutoEncoder over BESPOKE embeddings — interpretable 'latent terms'.

Top-K SAE (Gao et al. 2024), per the Latent Terms paper (arXiv:2605.29384). Encoder → ReLU →
keep the top-k activations per row → decoder; trained with MSE reconstruction. Sparsity is
enforced by the k cutoff (no L1 needed). Runs on the M4 GPU via MLX.

Adaptation: the paper trains on token-level hidden states (30B tokens); we run on our POOLED
EmbeddingGemma vectors (~40k) with a scaled-down dictionary. Expect dead/noisy features at this
scale — proving useful features emerge IS the experiment. Standalone; not on the critical path.
"""
import numpy as np


class TopKSAE:
    """Minimal Top-K SAE backed by MLX. Holds params; train via train_topk_sae()."""

    def __init__(self, d_in: int, d_hidden: int = 4096, k: int = 16, seed: int = 0):
        import mlx.core as mx
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = min(k, d_hidden)
        rng = np.random.RandomState(seed)
        # Decoder Kaiming-ish; encoder initialized as the transposed decoder (paper's scheme).
        W_dec = (rng.randn(d_hidden, d_in).astype(np.float32) / np.sqrt(d_in))
        self.W_dec = mx.array(W_dec)
        self.W_enc = mx.array(W_dec.T.copy())   # [d_in, d_hidden]
        self.b_enc = mx.zeros((d_hidden,))
        self.b_pre = mx.zeros((d_in,))

    def parameters(self):
        return {"W_enc": self.W_enc, "W_dec": self.W_dec,
                "b_enc": self.b_enc, "b_pre": self.b_pre}

    def update(self, params):
        self.W_enc, self.W_dec = params["W_enc"], params["W_dec"]
        self.b_enc, self.b_pre = params["b_enc"], params["b_pre"]

    def _encode(self, x, params=None):
        import mlx.core as mx
        p = params or self.parameters()
        a = mx.maximum((x - p["b_pre"]) @ p["W_enc"] + p["b_enc"], 0.0)   # ReLU
        # keep top-k per row (>= the k-th largest; ties may keep a few extra — fine)
        kth = mx.sort(a, axis=-1)[..., -self.k][..., None]
        return mx.where(a >= kth, a, 0.0)

    def _forward(self, x, params=None):
        import mlx.core as mx
        p = params or self.parameters()
        z = self._encode(x, p)
        recon = z @ p["W_dec"] + p["b_pre"]
        return recon, z

    def encode(self, X) -> np.ndarray:
        """Sparse feature activations for X: returns [N, d_hidden]."""
        import mlx.core as mx
        out = []
        for s in range(0, len(X), 2048):
            z = self._encode(mx.array(np.asarray(X[s:s + 2048], np.float32)))
            out.append(np.array(z))
        return np.vstack(out) if out else np.zeros((0, self.d_hidden), np.float32)


def train_topk_sae(X, d_hidden: int = 4096, k: int = 16, epochs: int = 30,
                   lr: float = 1e-3, batch: int = 512, seed: int = 0) -> TopKSAE:
    """Train a Top-K SAE on the (pooled) embedding matrix X [N, d]. Returns the fitted SAE."""
    import mlx.core as mx
    import mlx.optimizers as optim

    X = np.asarray(X, np.float32)
    sae = TopKSAE(X.shape[1], d_hidden=d_hidden, k=k, seed=seed)
    opt = optim.Adam(learning_rate=lr)

    def loss_fn(params, xb):
        recon, _ = sae._forward(xb, params)
        return mx.mean((recon - xb) ** 2)

    lvg = mx.value_and_grad(loss_fn)
    Xmx = mx.array(X)
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            xb = Xmx[mx.array(perm[s:s + batch])]
            loss, grads = lvg(sae.parameters(), xb)
            sae.update(opt.apply_gradients(grads, sae.parameters()))
            mx.eval(sae.parameters(), opt.state)
    return sae


def top_examples_for_feature(Z, ids, feature, top=8):
    """Which items most strongly activate a feature → read its meaning (naming)."""
    col = Z[:, feature]
    order = np.argsort(-col)
    return [(ids[i], float(col[i])) for i in order[:top] if col[i] > 0]


def feature_label_separation(Z, y):
    """Confound diagnostic: |mean activation on accepts − on rejects| per feature.
    A high value means that feature discriminates accept vs reject — i.e., the embedding DOES
    carry a quality-relevant axis (just not cosine-expressible). y in {1 accept, 0 reject}.
    """
    y = np.asarray(y)
    acc = Z[y == 1].mean(axis=0) if (y == 1).any() else np.zeros(Z.shape[1])
    rej = Z[y == 0].mean(axis=0) if (y == 0).any() else np.zeros(Z.shape[1])
    return np.abs(acc - rej)


def main():
    """Train a SAE on the warehouse embeddings and print an inspection report.

    Run:  python -m bespoke.latent.sae [d_hidden] [k]
    Read-only on the DB. The experiment: do interpretable features emerge at our scale, and
    does any feature separate accept vs reject (the confound)?
    """
    import sys
    from bespoke.db.init import get_connection
    from bespoke.eval.signals import get_labeled_embeddings

    d_hidden = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    conn = get_connection()
    ids, X, y = get_labeled_embeddings(conn)
    print(f"Training Top-K SAE on {len(X)} embeddings (d_hidden={d_hidden}, k={k})...")
    sae = train_topk_sae(X, d_hidden=d_hidden, k=k)
    Z = sae.encode(X)

    alive = int((Z.max(axis=0) > 0).sum())
    print(f"  alive features: {alive}/{d_hidden} ({100*alive/d_hidden:.0f}%)  "
          f"(dead = never fired — expected when data << dictionary)")

    labeled = y != -1
    if labeled.sum() > 2 and len(set(y[labeled].tolist())) == 2:
        sep = feature_label_separation(Z[labeled], y[labeled])
        order = np.argsort(-sep)[:8]
        print("\nTop features that separate ACCEPT vs REJECT (the confound test):")
        for f in order:
            ex = top_examples_for_feature(Z, ids, int(f), top=2)
            sample = conn.execute(
                "SELECT substr(user_message,1,70) FROM interactions WHERE id=?",
                (ex[0][0],)).fetchone()[0] if ex else ""
            print(f"  feat {int(f):>5}: sep={sep[f]:.3f}  e.g. “{sample}”")
    conn.close()


if __name__ == "__main__":
    main()
