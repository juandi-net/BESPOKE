# tests/test_latent_sae.py
"""Top-K SAE over embeddings (latent terms) — standalone experimental module."""
import numpy as np


class TestTopKSparsity:
    def test_keeps_at_most_k(self):
        from bespoke.latent.sae import TopKSAE
        sae = TopKSAE(d_in=8, d_hidden=16, k=4, seed=1)
        X = np.random.RandomState(0).randn(5, 8).astype(np.float32)
        Z = sae.encode(X)
        assert Z.shape == (5, 16)
        nz = (Z > 0).sum(axis=1)
        assert (nz <= 4).all()


class TestTraining:
    def test_reduces_reconstruction_loss(self):
        import mlx.core as mx
        from bespoke.latent.sae import TopKSAE, train_topk_sae
        rng = np.random.RandomState(0)
        basis = rng.randn(3, 8).astype(np.float32)         # data lives on a 3-dim subspace
        X = (rng.randn(200, 3).astype(np.float32) @ basis)

        def mse(sae):
            recon, _ = sae._forward(mx.array(X))
            return float(mx.mean((recon - mx.array(X)) ** 2))

        l0 = mse(TopKSAE(8, d_hidden=16, k=4, seed=0))
        trained = train_topk_sae(X, d_hidden=16, k=4, epochs=60, lr=1e-2, batch=64, seed=0)
        assert mse(trained) < l0


class TestConfoundDiagnostic:
    def test_feature_separation_finds_the_discriminating_feature(self):
        from bespoke.latent.sae import feature_label_separation
        Z = np.zeros((10, 4), np.float32)
        Z[:5, 0] = 1.0                       # feature 0 fires only on the accepts
        y = np.array([1] * 5 + [0] * 5)
        sep = feature_label_separation(Z, y)
        assert sep.argmax() == 0 and sep[0] > 0.5

    def test_top_examples(self):
        from bespoke.latent.sae import top_examples_for_feature
        Z = np.array([[0, 0], [0.9, 0], [0.5, 0]], np.float32)
        out = top_examples_for_feature(Z, ids=[10, 11, 12], feature=0, top=2)
        assert out[0][0] == 11 and out[1][0] == 12
