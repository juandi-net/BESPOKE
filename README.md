# BESPOKE

Compound frontier model interactions into bespoke AI experiences.

Every frontier model interaction generates high-quality reasoning, problem decomposition, tradeoff analysis, and structured solutions that disappear when the session ends. BESPOKE captures those demonstrations, extracts the reasoning patterns, and compresses them into local adapters that concentrate all capacity on the domains you actually work in.

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

**Extract** — An LLM classifies each interaction nightly: domain, quality, reasoning primitives, user feedback signals. Produces clean training pairs. A second pass mines cross-corpus patterns and updates curriculum weights weekly.

**Train** — Every night, DoRA fine-tuning runs autonomously on the curated data. An evaluation benchmark, built from your stated preferences and your actual accept/reject behavior, governs keep/revert. If the adapter improves, it ships. If not, it reverts. No human in the loop.

**Serve** — llama.cpp serves the base model with adapters hot-swapped per query. Any tool that speaks the OpenAI API format can use it.

**Repeat** — Your continued frontier model interactions and your interactions with deployed adapters both feed back into capture. Frontier sessions bring fresh reasoning patterns. Adapter sessions generate accept/reject signals that refine training. The benchmark tracks drift in your standards. Three curves compound: the warehouse grows, the benchmark refines, the adapters improve.

## V0 Target

- **Hardware:** Mac Mini M4, 16GB unified memory
- **Base model:** Qwen3.5-4B (Q4_K_M, ~2.7GB)
- **Training:** MLX with QLoRA + DoRA + rsLoRA
- **Embeddings:** EmbeddingGemma 300M ONNX (768-dim, 2K context)
- **Database:** SQLite + sqlite-vec
- **Inference:** llama.cpp with Metal GPU acceleration

## Status

Active development. V0 pipeline functional end-to-end.

## Data

All user data lives in `~/.bespoke/` — database, model weights, adapters, scorecards, benchmark. Nothing in that directory is tracked by git.

## License

MIT

---

**bespoke.sh** — bespoke ai experiences.
