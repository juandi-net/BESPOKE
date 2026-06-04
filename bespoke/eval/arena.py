"""Blind thesis-eval arena: base vs adapter vs frontier (=captured response), judged by juandi.

The thesis is "fit beats scale on your work." The only ground-truth judge of *your* standard that is
neither faint (geometric ~0.62, RT-003) nor off-sovereignty (LLM) is YOU. And the frontier baseline is
free + sovereign: every captured assistant_response IS a frontier response (juandi's idea — no new cloud
calls). The held-out set is the training run's own valid.jsonl (the 10% split: representative high+med
prompts, already cleaned, provably not trained on; its assistant turn = the frontier answer).

Flow:  build  ->  rate (juandi picks blind)  ->  score (win-rates = the thesis number).
"""
import json
import random
from datetime import datetime
from pathlib import Path

from bespoke.config import config

MODELS = ("base", "adapter", "frontier")
_DEFAULT_DIR = Path.home() / ".bespoke" / "arena"
_VALID = Path.home() / ".bespoke" / "training_data" / "valid.jsonl"


def assemble_contestants(responses, seed):
    """Shuffle {base,adapter,frontier} into blind A/B/C. Returns (labeled, key={letter:model})."""
    rng = random.Random(seed)
    models = list(MODELS)
    rng.shuffle(models)
    key = {letter: m for letter, m in zip(("A", "B", "C"), models)}
    labeled = [{"label": L, "text": responses[key[L]]} for L in ("A", "B", "C")]
    return labeled, key


def score_verdicts(items):
    """Win-rates from rated items. Each: {key:{letter:model}, verdict: letter|'none'|None}."""
    wins = {m: 0 for m in MODELS}
    n_rated = none_acceptable = 0
    for it in items:
        v = it.get("verdict")
        if v is None:
            continue
        n_rated += 1
        if v == "none":
            none_acceptable += 1
            continue
        model = it["key"].get(v)
        if model in wins:
            wins[model] += 1
    win_rate = {m: (wins[m] / n_rated if n_rated else 0.0) for m in MODELS}
    return {"n_rated": n_rated, "wins": wins, "win_rate": win_rate, "none_acceptable": none_acceptable}


def select_arena_items(n=16, seed=0):
    """Held-out prompts + the frontier (captured) answer, from the training run's valid.jsonl split."""
    if not _VALID.exists():
        raise RuntimeError(f"No held-out set at {_VALID} — run `bespoke train` first (it writes valid.jsonl).")
    rows = []
    for line in _VALID.read_text().splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line).get("messages", [])
        prompt = next((m["content"] for m in msgs if m["role"] == "user"), "")
        frontier = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        if 20 <= len(prompt) <= 2000 and len(frontier) >= 20:
            rows.append({"prompt": prompt, "frontier": frontier})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def _context_messages(prompt, conn, max_prior=3):
    """Reconstruct prior-turn context for a held-out prompt from the warehouse (RT-004 fix).

    Many held-out prompts are mid-conversation turns; the frontier answered them WITH the full
    thread, so base/adapter must get the same context to be judged fairly. Returns a chat message
    list of up to max_prior prior (user, assistant) turns from the same session, [] if none/no match.
    """
    row = conn.execute(
        "SELECT session_id, captured_at FROM interactions WHERE user_message = ? "
        "AND session_id IS NOT NULL ORDER BY captured_at LIMIT 1", (prompt,)).fetchone()
    if not row:
        return []
    prior = conn.execute(
        "SELECT user_message, assistant_response FROM interactions "
        "WHERE session_id = ? AND captured_at < ? ORDER BY captured_at DESC LIMIT ?",
        (row["session_id"], row["captured_at"], max_prior)).fetchall()
    msgs = []
    for r in reversed(prior):
        um, ar = (r["user_message"] or "").strip(), (r["assistant_response"] or "").strip()
        if um:
            msgs.append({"role": "user", "content": um[:1500]})
        if ar:
            msgs.append({"role": "assistant", "content": ar[:1500]})
    return msgs


def generate_with_context(message_lists, model_path, adapter_path):
    """Generate one response per chat-message-list (multi-turn context). adapter_path="" -> base."""
    import gc
    import mlx_lm
    model, tok = (mlx_lm.load(model_path, adapter_path=adapter_path) if adapter_path
                  else mlx_lm.load(model_path))
    out = []
    for msgs in message_lists:
        try:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True)
        except Exception:
            p = msgs[-1]["content"]
        out.append(mlx_lm.generate(model, tok, p, max_tokens=512, verbose=False))
    del model, tok
    gc.collect()
    return out


def build_arena(n=16, out_dir=None, seed=0):
    """Select held-out items, reconstruct session context, generate base+adapter, blind-assemble, write."""
    from bespoke.db.init import get_connection

    items = select_arena_items(n=n, seed=seed)
    if not items:
        raise RuntimeError("No held-out items found to build an arena.")

    # Give base/adapter the same session context the frontier had (RT-004 methodology fix).
    conn = get_connection()
    msg_lists = []
    for it in items:
        ctx = _context_messages(it["prompt"], conn)
        it["context_turns"] = len(ctx) // 2
        msg_lists.append(ctx + [{"role": "user", "content": it["prompt"]}])
    conn.close()
    with_ctx = sum(1 for it in items if it["context_turns"] > 0)

    model_path = str(config.base_model.training_model_path)
    adapter_path = str(config.adapters_dir / "general-v1" / "sft")
    print(f"Generating BASE responses ({len(items)} prompts; {with_ctx} with reconstructed context)...")
    base = generate_with_context(msg_lists, model_path, "")
    print("Generating ADAPTER responses...")
    adapt = generate_with_context(msg_lists, model_path, adapter_path)

    arena = {"created": datetime.now().isoformat(timespec="seconds"),
             "model_path": model_path, "adapter_path": adapter_path, "items": []}
    for i, it in enumerate(items):
        responses = {"base": base[i].strip(), "adapter": adapt[i].strip(),
                     "frontier": (it["frontier"] or "").strip()}
        labeled, key = assemble_contestants(responses, seed=seed * 1000 + i)
        arena["items"].append({
            "prompt": it["prompt"], "context_turns": it.get("context_turns", 0),
            "responses": {x["label"]: x["text"] for x in labeled},
            "key": key, "verdict": None,
        })

    out = Path(out_dir or _DEFAULT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "arena.json").write_text(json.dumps(arena, indent=2))
    (out / "arena.md").write_text(_render_md(arena))
    print(f"\nArena written ({len(arena['items'])} items):\n  {out/'arena.json'}\n  {out/'arena.md'} (readable)")
    print("Rate it blind:  bespoke arena --rate    then    bespoke arena --score")
    return out / "arena.json"


def _render_md(arena):
    L = ["# BESPOKE thesis arena — blind rating", "",
         "Read each prompt + the 3 responses (A/B/C, models hidden). Pick the ONE you'd actually keep.",
         "Use `bespoke arena --rate` to record picks, then `bespoke arena --score`.", ""]
    for n, it in enumerate(arena["items"], 1):
        L.append(f"## {n}.\n**Prompt:** {it['prompt']}\n")
        for letter in ("A", "B", "C"):
            L.append(f"**[{letter}]**\n\n{it['responses'][letter]}\n")
        L.append("**Your pick: ___**\n\n---\n")
    return "\n".join(L)


def rate_arena(json_path=None):
    """Interactive blind rating: show prompt + A/B/C, capture juandi's pick into arena.json."""
    path = Path(json_path or (_DEFAULT_DIR / "arena.json"))
    arena = json.loads(path.read_text())
    todo = [it for it in arena["items"] if it.get("verdict") is None]
    print(f"{len(todo)} of {len(arena['items'])} items left to rate.\n")
    for n, it in enumerate(arena["items"], 1):
        if it.get("verdict") is not None:
            continue
        print(f"\n{'='*72}\nITEM {n}/{len(arena['items'])}\nPROMPT: {it['prompt']}\n")
        for letter in ("A", "B", "C"):
            print(f"--- [{letter}] {'-'*60}\n{it['responses'][letter]}\n")
        choice = input("Which would you KEEP? [A/B/C | none | s=skip | q=quit]: ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue
        it["verdict"] = "none" if choice == "none" else (choice.upper() if choice in ("a", "b", "c") else None)
        path.write_text(json.dumps(arena, indent=2))  # persist after each
    print("\nSaved. Score with: bespoke arena --score")


def score_arena(json_path=None):
    path = Path(json_path or (_DEFAULT_DIR / "arena.json"))
    arena = json.loads(path.read_text())
    rep = score_verdicts(arena["items"])
    print(f"\nThesis arena — {rep['n_rated']}/{len(arena['items'])} rated"
          f"  ({rep['none_acceptable']} 'none acceptable')")
    if rep["n_rated"]:
        print("win-rate (which you'd keep, blind):")
        for m in MODELS:
            print(f"  {m:9} {rep['wins'][m]:3}  ({rep['win_rate'][m]*100:.0f}%)")
        a, f, b = (rep["win_rate"][k] for k in ("adapter", "frontier", "base"))
        print(f"\n  adapter vs base:     adapter {a*100:.0f}% vs base {b*100:.0f}%  "
              f"-> {'fit improves over the raw base' if a > b else 'no gain over base' if a < b else 'tie'}")
        print(f"  adapter vs frontier: adapter {a*100:.0f}% vs frontier {f*100:.0f}%  "
              f"-> {'FIT >= SCALE on your work!' if a >= f else 'gap remains to frontier'}")
    return rep
