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
    """Held-out prompts + the frontier (captured) answer, from the training run's valid.jsonl split.

    Strips injected app boilerplate (Conductor <system_instruction>) so the rated prompt is the REAL
    ask; pure-boilerplate turns drop out (empty after strip). Keeps `orig` for session-context lookup.
    """
    from bespoke.extract.content_type import strip_conductor_boilerplate
    if not _VALID.exists():
        raise RuntimeError(f"No held-out set at {_VALID} — run `bespoke train` first (it writes valid.jsonl).")
    rows = []
    for line in _VALID.read_text().splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line).get("messages", [])
        orig = next((m["content"] for m in msgs if m["role"] == "user"), "")
        frontier = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        prompt = (strip_conductor_boilerplate(orig) or "")
        if 20 <= len(prompt) <= 2000 and len(frontier) >= 20:
            rows.append({"prompt": prompt, "orig": orig, "frontier": frontier})
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
        out.append(mlx_lm.generate(model, tok, p, max_tokens=320, verbose=False))  # shorter: faster + easier to rate
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
        ctx = _context_messages(it.get("orig", it["prompt"]), conn)  # match on the ORIGINAL (un-stripped) text
        it["context_turns"] = len(ctx) // 2
        msg_lists.append(ctx + [{"role": "user", "content": it["prompt"]}])  # generate on the stripped real ask
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


_RATING_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>BESPOKE — shape your model</title>
<style>
  body { font-family:'Times New Roman',Times,serif; background:#fdfcf8; color:#1a1a1a;
         max-width:760px; margin:0 auto; padding:48px 28px 110px; line-height:1.55; }
  h1 { font-size:30px; font-weight:normal; letter-spacing:.4px; margin:0 0 4px; }
  .sub { font-style:italic; color:#555; margin:0 0 26px; }
  .prog { font-size:14px; color:#555; margin-bottom:6px; }
  .bar { height:3px; background:#e7e3d7; margin-bottom:30px; }
  .bar > div { height:100%; background:#1a1a1a; width:0; transition:width .35s; }
  .ask { background:#f4f1e8; border-left:3px solid #1a1a1a; padding:12px 16px; margin-bottom:22px; }
  .ask .lbl { font-style:italic; color:#666; font-size:13px; display:block; margin-bottom:4px; }
  .card { border:1px solid #d8d3c4; padding:14px 18px; margin-bottom:14px; cursor:pointer;
          white-space:pre-wrap; transition:background .12s,border-color .12s; }
  .card:hover { background:#f7f4ec; border-color:#1a1a1a; }
  .card .tag { font-weight:bold; font-style:italic; margin-right:10px; }
  .none { text-align:center; color:#777; cursor:pointer; padding:12px; font-style:italic; }
  .none:hover { color:#1a1a1a; }
  .why { width:100%; font-family:inherit; font-size:15px; padding:9px 11px; box-sizing:border-box;
         border:1px solid #d8d3c4; background:#fff; margin-top:8px; }
  .why::placeholder { font-style:italic; color:#aaa; }
  .whylbl { font-style:italic; color:#666; font-size:13px; margin-top:14px; }
  .hint { font-size:13px; color:#999; margin-top:20px; text-align:center; }
  .done { text-align:center; padding-top:70px; }
  code { background:#f0ece0; padding:1px 5px; }
</style></head>
<body><div id="app"></div>
<script>
let items=[], cur=-1, total=0;
const app=document.getElementById('app');
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function rated(){return items.filter(it=>it.verdict!==null).length;}
async function load(){
  const d=await (await fetch('/api/arena')).json();
  items=d.items; total=items.length;
  cur=items.findIndex(it=>it.verdict===null);
  cur===-1 ? finish() : render();
}
async function pick(v){
  if(cur<0)return;
  const why=(document.getElementById('why')||{}).value||'';
  items[cur].verdict=v; items[cur].why=why;
  await fetch('/api/verdict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({index:cur,verdict:v,why:why})});
  cur=items.findIndex(it=>it.verdict===null);
  cur===-1 ? finish() : render();
}
function render(){
  const it=items[cur], done=rated();
  app.innerHTML=`<h1>Shape your model</h1>
  <p class="sub">Each pick teaches BESPOKE what <em>good</em> means to you — you're not testing it, you're shaping it.</p>
  <div class="prog">You've shaped ${done} of ${total}</div>
  <div class="bar"><div style="width:${(100*done/total).toFixed(1)}%"></div></div>
  <div class="ask"><span class="lbl">Your ask${it.context_turns?' (mid-conversation)':''}:</span>${esc(it.prompt)}</div>
  ${['A','B','C'].map(L=>`<div class="card" onclick="pick('${L}')"><span class="tag">${L}</span>${esc(it.responses[L])}</div>`).join('')}
  <div class="none" onclick="pick('none')">— I'd keep none of these —</div>
  <div class="whylbl">why? (optional — most useful when none fit: what's wrong, what you'd want instead)</div>
  <input class="why" id="why" autocomplete="off" placeholder="e.g. too verbose & sycophantic; didn't answer the question…">
  <div class="hint">click a card or — none —; keys A · B · C · N work when not typing here</div>`;
  window.scrollTo(0,0);
}
function finish(){
  app.innerHTML=`<div class="done"><h1>You've shaped all ${total}.</h1>
  <p class="sub">This is now part of how your model learns your standard.<br>Thank you for contributing.</p>
  <p class="hint">Run <code>bespoke arena --score</code> to see the result.</p></div>`;
}
document.addEventListener('keydown',e=>{
  if(document.activeElement&&document.activeElement.id==='why')return;  // don't hijack keys while typing why
  const k=e.key.toLowerCase();
  if(k==='a')pick('A');else if(k==='b')pick('B');else if(k==='c')pick('C');else if(k==='n')pick('none');});
load();
</script></body></html>"""


def serve_arena(json_path=None, port=8421):
    """Serve a minimal local rating page (Times New Roman, blind A/B/C); each pick saves to disk."""
    import http.server
    import socketserver
    import webbrowser
    from urllib.parse import urlparse

    path = Path(json_path or (_DEFAULT_DIR / "arena.json"))

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            p = urlparse(self.path).path
            if p in ("/", "/index.html"):
                self._send(200, _RATING_HTML, "text/html; charset=utf-8")
            elif p == "/api/arena":
                a = json.loads(path.read_text())
                items = [{"prompt": it["prompt"], "responses": it["responses"],  # blind: no key
                          "verdict": it["verdict"], "context_turns": it.get("context_turns", 0)}
                         for it in a["items"]]
                self._send(200, json.dumps({"items": items}))
            else:
                self._send(404, "{}")

        def do_POST(self):
            if urlparse(self.path).path == "/api/verdict":
                n = int(self.headers.get("Content-Length", 0) or 0)
                data = json.loads(self.rfile.read(n) or b"{}")
                a = json.loads(path.read_text())
                idx = int(data["index"])
                a["items"][idx]["verdict"] = data["verdict"]
                a["items"][idx]["why"] = (data.get("why") or "").strip()  # the high-resolution signal
                path.write_text(json.dumps(a, indent=2))  # persist every pick
                rated = sum(1 for it in a["items"] if it["verdict"] is not None)
                self._send(200, json.dumps({"ok": True, "rated": rated, "total": len(a["items"])}))
            else:
                self._send(404, "{}")

    url = f"http://localhost:{port}/"
    print(f"Rating page: {url}\n  open it, click through (or A/B/C/N) — each pick saves instantly. Ctrl-C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    with socketserver.TCPServer(("127.0.0.1", port), H) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped. Score with:  bespoke arena --score")


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
