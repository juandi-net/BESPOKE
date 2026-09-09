# AGENTS.md — BESPOKE

Project guidance for any agent (Codex or other) working in this repo. This layers on top of
the user's global `~/.Codex/AGENTS.md` — don't repeat global rules here; this file is BESPOKE-specific.

**What BESPOKE is (one line):** a system that captures how frontier models reason about *your* work
and compresses it into a small, local, personal model — *"fit beats scale."* Full thesis in
`docs/bespoke-vision.md`.

---

## Start here — READ before any non-trivial work

You are NOT oriented until you've read these. Do it at the start of a session that touches design,
research, training, or the pipeline. Order matters (why → what → decisions → now → experiments):

1. `docs/bespoke-vision.md` — the thesis and what we're building toward (fit > scale; the fleet; the Mirror).
2. `ARCHITECTURE.md` (repo root) — current architecture & pipeline; `README.md` for the quick overview.
3. `strategy/program.md` — training strategy + the **decision history** (silver thread, topic-vs-quality
   confound as a *dial*, base-model = LFM2.5-1.2B, geometric eval). The single richest record of *why*.
4. `docs/bespoke-research-log.md` — every experiment we've run on our own data (RT-NNN), replicable.
5. Subsystem design notes as needed: `docs/bespoke-local-pipeline-design.md` (fully-local pipeline),
   `docs/bespoke-meta-approach-design.md` (topic-invariant "approach"/personality), `docs/bespoke-router-design.md`
   (fleet router, deferred), `docs/bespoke-track2-extraction.md`. External lit: `research/0*.md`.

> The orientation docs above (`docs/`, `strategy/program.md`, `research/`) are **internal / local-only**
> (gitignored — not in a fresh clone). The public record is the committed code + `ARCHITECTURE.md` + `README.md`.

## Where we are right now

Reconstruct the present from, in order: the recent **`git log`** (committed code = the durable record), the
top of **`docs/bespoke-research-log.md`** (latest experiments), and the newest decisions in
**`strategy/program.md`** and the `docs/bespoke-*-design.md` notes.

---

## Standing rules for this repo

1. **Orient first.** Read the docs above before proposing designs or running experiments. Don't
   reinvent decisions already recorded in `program.md` / the design notes.

2. **Log every research test → `docs/bespoke-research-log.md`.** ANY experiment you run against the
   warehouse/data (a probe, a clustering, an ablation, a measurement) gets a new `RT-NNN` entry with:
   question, how-we-got-here, method, the **exact replicate command**, results, interpretation
   (what it does and does NOT prove), caveats, next steps. If it can't be re-run from the entry, the
   entry isn't done. Newest at the top.

3. **Record every decision → the relevant design note.** When we reach a real decision (architecture,
   method, hyperparameters, scope, a pivot), write it down with the **reasoning and date** in the
   right place: training/strategy → `strategy/program.md`; a subsystem → its `docs/bespoke-*-design.md`
   (create one if none fits). A decision that isn't written down didn't happen — capture *why*, not just *what*.

4. **The internal docs ARE the durable memory.** `docs/`, `strategy/program.md`, and `research/` are
   gitignored (local-only, not published) by design. Keep them current as you work — they outlive any
   single session and are how the next agent gets up to speed. Code is committed normally.

---

## Environment / running things

- Python venv: **`.venv/bin/python`** (has `mlx_lm`, `mlx`, `igraph`/`leidenalg`, `sklearn`). The base
  `python` does NOT have these — always use `.venv/bin/python` (or `.venv/bin/bespoke`).
- Warehouse: SQLite at `~/.bespoke/warehouse.db` (interactions + `vec_interactions` sqlite-vec embeddings).
- CLI: `bespoke capture | extract | train | eval | benchmark` (see `bespoke/cli.py`). Standalone latent
  experiments live in `bespoke/latent/` (e.g. `python -m bespoke.latent.approach`).
- Apple Silicon / MLX, 16GB budget: training is batch=1 / 8 layers / grad-checkpoint (see `train_sft.py`).
