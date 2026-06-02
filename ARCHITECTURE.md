# BESPOKE — Architecture & Key Decisions

A concept-level record of *why* BESPOKE is built the way it is. For commands, see the README.

## Thesis
Frontier-model reasoning is ephemeral and rented. BESPOKE captures the demonstrations you
already generate by working, and compresses them into a local model that concentrates its
(smaller) capacity on the few domains you actually operate in. A specialist tuned on your
recurring work can match a general giant *on that work* — because the giant spreads capacity
across all of human knowledge, and the specialist spends all of its where it matters.

## The decision that defines the system: geometry instead of a cloud judge
Self-improving loops usually depend on a paid cloud model to judge their own output every
cycle. That is not a closed loop — it is a metered dependency that also ships your data out to
be graded. It fails three ways: **unsustainable** (recurring cost compounds against you as the
system works harder), **inaccessible** (the off-ramp from frontier prices handed back to a toll
booth), and **fragile** (a noisy judge over a small sample).

**Principle:** you don't make intelligence deterministic — you *shrink the surface that needs
intelligence* and push the rest to the cheapest source that's good enough. Most of "judgment"
here is geometry in disguise: once a single user is fixed, their accepted answers form a
low-dimensional, separable structure in latent space — the "silver thread." Judging becomes
measuring; measuring is cheap, deterministic, and local.

## The three loops, closed locally
| Stage | Was | Now |
|---|---|---|
| **Eval** (keep/revert) | cloud LLM-as-judge | geometric ensemble: programmatic gates + linear preference probe + graph label propagation + a meta-scorer (local) |
| **Extract** (curate pairs) | cloud LLM classifier | segment + mechanical copy + label by latent position (probe quality, cluster domain, rule-based feedback) |
| **Serve** (route, future fleet) | — | geometric routing among local adapters; frontier only on genuine out-of-distribution need |

The frontier still *teaches* BESPOKE — but only through the sessions you generate by working,
never through extra cloud calls that re-send your data to curate or grade it.

## The signals (all local, orthogonal)
- **Programmatic gates** — correctness is run, not opined (the un-foolable anchor).
- **Linear preference probe** — your "taste direction" from accept/reject; scores a single answer.
- **Graph label propagation** — spread a few labels across the embedding graph (scikit-learn).
- **Leiden clustering** — domain assignment + rare-example curriculum weighting (no naming needed).
- **Time** — a first-class, under-used eval signal a *personal* system uniquely has:
  time-to-accept, engagement/dwell (from prompt-cache presence), efficiency, temporal arc.

## Durable vs disposable
The **data warehouse** and the **quality standard** are the moat. **Adapters and the base model**
are disposable derivatives — cheap to regenerate. Swapping the base (currently LFM2.5-1.2B-Instruct,
chosen for being small, dense, and phone-deployable — but model-agnostic by design) loses nothing.

## Why this matters
The cost of improving drops to electricity, not tokens. Your data never leaves `~/.bespoke/`.
The system runs on the box on your desk — and is built to eventually run in your pocket.
