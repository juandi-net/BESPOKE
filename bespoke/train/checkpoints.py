"""Best-checkpoint selection from a captured mlx_lm training log.

RT-002 found the saved final adapter can be WORSE than an earlier checkpoint (val 2.000 @ it600
vs 2.132 @ it800 final) — mlx_lm keeps the last iteration, not the best. This parses the val-loss
lines from the run's captured stdout and picks the saved checkpoint with the lowest val loss.

Log format (verified on a real run, 2026-06-10):  `Iter 600: Val loss 2.000, Val took 10.2s`
Checkpoint files (--save-every):                  `0000600_adapters.safetensors`
"""
import re
from pathlib import Path

_VAL_LINE = re.compile(r"Iter (\d+): Val loss ([\d.]+)")


def parse_val_losses(log_text):
    """{iteration: val_loss} from mlx_lm stdout. Train-loss lines are ignored."""
    return {int(m.group(1)): float(m.group(2)) for m in _VAL_LINE.finditer(log_text or "")}


def select_best_checkpoint(log_text, adapter_dir):
    """Path of the saved checkpoint with the lowest val loss, or None if the final adapter
    already is the best (or no usable val data). Only iterations that actually have a
    {iter:07d}_adapters.safetensors on disk qualify (val evals are more frequent than saves).
    """
    losses = parse_val_losses(log_text)
    adapter_dir = Path(adapter_dir)
    saved = {it: lo for it, lo in losses.items()
             if (adapter_dir / f"{it:07d}_adapters.safetensors").exists()}
    if not saved:
        return None
    best_iter = min(saved, key=saved.get)
    if best_iter == max(saved):  # last save-point == what the final adapter already is
        return None
    return adapter_dir / f"{best_iter:07d}_adapters.safetensors"
