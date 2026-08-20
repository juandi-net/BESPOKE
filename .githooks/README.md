# Git hooks

This directory holds versioned git hooks that stop secrets and private files
from entering the public history.

## Setup (one command, per clone)

    git config core.hooksPath .githooks

For PRECEPT this also runs automatically via the `prepare` script on `bun install`.

## What `pre-commit` blocks

**Private paths** — `.env`, `.DS_Store`, `data/orgs/<org>/`, `.serena/`,
`.conductor/`, `conductor.json`, `fly.toml`, `docs/plans|superpowers|archive/`,
`strategy/program.md`, `strategy/LEARNING.md`, `research/`, `*.pem`, `*.key`,
`*.p12`, `*.pfx`, SSH private keys, `*.sqlite`, `secrets.*`.

**Secret content** — GitHub tokens, OpenAI/Anthropic keys, AWS access keys,
Google API keys, Slack tokens, Resend keys, JWT/Supabase keys, private key
blocks, and generic `api_key=`/`password=`/`token=` assignments.

Files ending in `.example`, `.sample`, `.template`, or `.dist` are skipped, and
obvious placeholders (`your-...`, `process.env.X`, `${VAR}`, `<value>`, code
references like `creds.apiKey`) do not trigger the generic rule.

Note that `git add -f` bypasses `.gitignore` but **not** this hook.

## If you hit a false positive

Use a placeholder value (`your-api-key-here`), move the sample into an
`*.example` file, or as a last resort `git commit --no-verify`.
GitHub push protection still runs server-side either way.

## Defense layers

1. `.gitignore` — keeps private files unstaged
2. `.githooks/pre-commit` — blocks the commit locally (this directory)
3. GitHub secret scanning + push protection — blocks the push server-side
