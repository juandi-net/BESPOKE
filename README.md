# BESPOKE

Compound frontier model interactions into bespoke AI experiences.

Every frontier model interaction generates high-quality reasoning, problem decomposition, tradeoff analysis, and structured solutions that disappear when the session ends. BESPOKE captures those demonstrations, extracts the reasoning patterns, and compresses them into local adapters that concentrate all capacity on the domains you actually work in.

## ⚠️ Before you start: your AI tools are deleting your history

This isn't hypothetical — **Claude Code auto-deletes your session transcripts.** By default it keeps only the **last 30 days** (`cleanupPeriodDays`) and removes older `.jsonl` files at startup. Your reasoning, decisions, and accept/reject signals are quietly thrown away on a rolling basis. *That* vanishing history is exactly what BESPOKE compresses into a model — so you have to preserve it **before it's gone.**

Do these two things **now**:

1. **Stop the deletion.** Set a long retention in `~/.claude/settings.json` (a large number ≈ "keep forever"):
   ```json
   { "cleanupPeriodDays": 365000 }
   ```
2. **Capture regularly** — run `bespoke capture` often (or on a schedule) so interactions reach your warehouse before any tool's cleanup window closes. The warehouse in `~/.bespoke/` is permanent; the source files are not.

If you've already been using Claude Code for a while, **most of your raw history beyond the last ~30 days is already gone from disk** — only what BESPOKE (or an export) captured still exists. Start capturing today.

## Usage

```bash
pip install -e .
```

```
bespoke capture              # ingest from Claude Code
bespoke capture --web        # ingest from claude.ai (web)
bespoke capture --all        # ingest from all sources
bespoke extract              # classify and curate training data
bespoke train                # fine-tune adapter (single run)
bespoke train --deadline 06:00  # search loop until 6 AM
bespoke train --max 3        # run exactly 3 experiments
bespoke eval                 # score adapter, keep or revert
bespoke serve                # start local model server
bespoke run                  # full pipeline end-to-end
bespoke run --search-deadline 06:00  # full pipeline with overnight search
bespoke benchmark interview  # define your quality standards
bespoke trajectory           # visualize your growth over time
```

Run `bespoke <command> --help` for flags.

## How It Works

```
Capture → Extract → Train → Serve → Repeat
```

**Capture** — Ingest interactions from Claude Code, claude.ai (web), Cursor, and other AI tools. Compute embeddings. Write to a local SQLite warehouse. Web capture extracts cookies from Claude Desktop's Keychain for authentication and supports incremental syncing.

**Extract** — Curation runs **locally and geometrically — no cloud LLM**. Sessions are segmented into coherent conversations (time gaps + embedding shifts + prompt-cache presence), training pairs are formed mechanically (the answer is already in the transcript), and each is labeled by where it sits in latent space: quality from a linear preference probe, domain from clustering, feedback from your own accept/reject signals. Curriculum weights come from community detection (Leiden), not a weekly LLM pass.

**Train** — Every night, LoRA fine-tuning runs autonomously on the curated data. If the adapter improves on the local geometric eval (label propagation + linear preference probe + programmatic gates, fused by a meta-scorer — no API calls), it ships. If not, it reverts. No human, and no paid judge, in the loop.

**Serve** — llama.cpp serves the base model with adapters hot-swapped per query. Any tool that speaks the OpenAI API format can use it.

**Repeat** — Your continued frontier model interactions and your interactions with deployed adapters both feed back into capture. Frontier sessions bring fresh reasoning patterns. Adapter sessions generate accept/reject signals that refine training. The benchmark tracks drift in your standards. Three curves compound: the warehouse grows, the benchmark refines, the adapters improve.

## Sovereign by Design — geometry instead of a cloud judge

Most "self-improving" AI systems quietly depend on a paid cloud model to *judge* their own output every cycle. That isn't a closed loop — it's a meter that never stops running, and it ships your data to someone else's server to grade it.

BESPOKE replaces that judgment with **geometry**. Your quality standards aren't an opinion an API has to re-derive each night — once *you* are fixed, your accepted answers form a low-dimensional, separable structure in latent space. Judging becomes *measuring*, and measuring is cheap, deterministic, and local:

- **Curate** without a cloud LLM — segment, copy, and label by position in latent space.
- **Evaluate** keep/revert with an ensemble of orthogonal signals — programmatic gates, a linear preference probe, graph label propagation, and **time** (how fast you accept, how long you engage) — fused by a small local model. No API calls.
- **Stay private** — your interactions, weights, and standards never leave `~/.bespoke/`.

The frontier still teaches BESPOKE — but only through the sessions you already generate by working, never through extra cloud calls re-sending your data to be graded. The result runs on the box on your desk (and, eventually, in your pocket), at the cost of electricity, not tokens.

## V0 Target

- **Hardware:** Mac Mini M4, 16GB unified memory
- **Base model:** LFM2.5-1.2B-Instruct (dense, 32K ctx, ~719MB on-device; Apache-2.0 fallback Qwen3-4B)
- **Training:** MLX with LoRA
- **Embeddings:** EmbeddingGemma 300M ONNX (768-dim, 2K context)
- **Database:** SQLite + sqlite-vec
- **Eval:** local geometric ensemble (scikit-learn LabelSpreading + LogisticRegression probe + Leiden), no LLM judge
- **Inference:** llama.cpp with LoRA hot-swap

## Status

Active development. V0 pipeline functional end-to-end.

## Data

All user data lives in `~/.bespoke/` — database, model weights, adapters, scorecards, benchmark. Nothing in that directory is tracked by git.

## License

MIT

---

**bespoke.sh** — bespoke ai experiences.
