"""Best-checkpoint selection (RT-002 flag: best val-loss checkpoint ≠ last saved adapter).

Log format verified against a real run (~/.bespoke/logs/sft-v2-taste.log, 2026-06-10):
  Iter 1: Val loss 2.315, Val took 11.371s
  Iter 100: Val loss 2.102, Val took 10.2s
Checkpoints land every --save-every iters as {iter:07d}_adapters.safetensors.
"""
from pathlib import Path

from bespoke.train.checkpoints import parse_val_losses, select_best_checkpoint

LOG = """\
Iter 1: Val loss 2.315, Val took 11.371s
Iter 20: Train loss 2.653, Learning Rate 2.000e-05, It/sec 0.946
Iter 100: Val loss 2.102, Val took 10.2s
Iter 200: Val loss 2.050, Val took 10.1s
Iter 300: Val loss 2.120, Val took 10.3s
Iter 400: Val loss 2.090, Val took 10.0s
"""


def test_parse_val_losses():
    assert parse_val_losses(LOG) == {1: 2.315, 100: 2.102, 200: 2.050, 300: 2.120, 400: 2.090}


def test_parse_ignores_train_loss_lines():
    assert 20 not in parse_val_losses(LOG)


def test_select_best_checkpoint_picks_lowest_val_with_a_saved_file(tmp_path):
    # checkpoints exist only at 200/400 — best val (200 @ 2.050) wins over final (400 @ 2.090)
    for it in (200, 400):
        (tmp_path / f"{it:07d}_adapters.safetensors").write_bytes(b"x")
    (tmp_path / "adapters.safetensors").write_bytes(b"final")
    best = select_best_checkpoint(LOG, tmp_path)
    assert best == tmp_path / "0000200_adapters.safetensors"


def test_select_returns_none_when_final_is_best(tmp_path):
    log = "Iter 200: Val loss 2.3, Val took 1s\nIter 400: Val loss 2.0, Val took 1s\n"
    for it in (200, 400):
        (tmp_path / f"{it:07d}_adapters.safetensors").write_bytes(b"x")
    # 400 is the last save-point = identical to the final adapter -> nothing to swap
    assert select_best_checkpoint(log, tmp_path) is None


def test_select_handles_no_val_lines(tmp_path):
    assert select_best_checkpoint("no losses here", tmp_path) is None
