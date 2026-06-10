"""One-time curation interview: mark your real interactions keep/drop + WHY, in a browser.

This is the SEED, not a chore (juandi, 2026-06-04): you curate once to (a) see your data and
(b) produce high-precision keep/drop labels + a WHY-rubric — replacing the noisy auto-accept/reject
that gave the faint ~0.62 signal. After this the system compounds autonomously from normal use.

No generation, no LLM — it shows your OWN real (prompt, response) pairs (boilerplate stripped, tool
dumps collapsed) and you judge them. Output: ~/.bespoke/curation/curation.json (keep/drop + why per
interaction) → feeds cleaner SFT data + the taste rubric.
"""
import json
import random
from pathlib import Path

from bespoke.config import config
from bespoke.db.init import get_connection

_DIR = Path.home() / ".bespoke" / "curation"
_FILE = _DIR / "curation.json"

# Stated-preferences interview (next-steps.md #1): a weak prior in juandi's own words, saved
# alongside the revealed keep/drop. The gap stated-vs-revealed is itself signal.
#
# It runs BEFORE keep/drop — he hasn't seen examples yet — so every prompt is OPEN free-text, not a
# forced scale/binary (juandi, 2026-06-04: scales pre-examples make him introspect in a vacuum and throw
# away nuance). He types the tradeoffs/context/anchors in his words; step #3 parses+weights each into the
# measurable axes. This list is the SINGLE source of truth — served to the page so UI/save/summary never
# drift. Swap wording freely; keep keys stable (they're the stored fields).
_STATED_QUESTIONS = [
    {"key": "good", "q": "When a response is exactly right — what makes it good?",
     "ph": ""},
    {"key": "bad", "q": "What makes you instantly drop one — the tells it isn't yours?",
     "ph": ""},
    {"key": "voice", "q": "How should a response sound or behave to feel like you?",
     "ph": ""},
    {"key": "tradeoffs", "q": "When your own rules collide — brevity vs completeness, just-do-it vs "
     "check-with-me-first, commit-to-one-answer vs lay-out-the-options — which wins, and when?",
     "ph": ""},
    {"key": "context", "q": "When does a long or deep answer earn its length — vs. when do you just "
     "want the short version?", "ph": ""},
    {"key": "level", "q": "Where are you a beginner vs. an expert? (so it can meet you where you are)",
     "ph": ""},
    {"key": "pushback", "q": "When you're wrong, or about to do something dumb — how should it handle that?",
     "ph": ""},
    {"key": "loved", "q": "Optional — paste a response you loved, and what made it land.", "ph": ""},
    {"key": "hated", "q": "Optional — paste one you hated, and what was off.", "ph": ""},
]
_STATED_FIELDS = tuple(q["key"] for q in _STATED_QUESTIONS)


def get_stated():
    """Return the stated-rubric answers ({} if the interview hasn't been done)."""
    if not _FILE.exists():
        return {}
    return json.loads(_FILE.read_text()).get("stated", {})


def save_stated(answers):
    """Persist the stated-preferences answers into curation.json (trimmed; marks done).

    Merges into the existing file — never clobbers the keep/drop items already on disk.
    """
    _DIR.mkdir(parents=True, exist_ok=True)
    a = json.loads(_FILE.read_text()) if _FILE.exists() else {"items": []}
    stated = {k: (str((answers or {}).get(k) or "")).strip() for k in _STATED_FIELDS}
    stated["done"] = True
    a["stated"] = stated
    _FILE.write_text(json.dumps(a, indent=2))
    return stated


def select_curation_items(n=40, seed=0):
    """Sample real interactions (boilerplate-stripped prompt + cleaned response) to curate."""
    from bespoke.extract.content_type import strip_conductor_boilerplate, clean_tool_blocks
    conn = get_connection()
    rows = conn.execute("""
        SELECT id, user_message, assistant_response, content_type
        FROM interactions
        WHERE content_type IN ('clean', 'agentic')
          AND length(user_message) BETWEEN 20 AND 2000
          AND length(assistant_response) BETWEEN 30 AND 4000
        ORDER BY RANDOM() LIMIT ?""", (n * 4,)).fetchall()
    items = []
    for r in rows:
        prompt = (strip_conductor_boilerplate(r["user_message"]) or "").strip()
        if not (20 <= len(prompt) <= 2000):
            continue
        resp = r["assistant_response"]
        if r["content_type"] == "agentic":
            resp = clean_tool_blocks(resp)
        resp = (resp or "").strip()
        if len(resp) < 30:
            continue
        items.append({"id": r["id"], "prompt": prompt, "response": resp, "verdict": None, "why": ""})
        if len(items) >= n:
            break
    conn.close()
    random.Random(seed).shuffle(items)
    return items


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>BESPOKE — curate your standard</title>
<style>
  body { font-family:'Times New Roman',Times,serif; background:#fdfcf8; color:#1a1a1a;
         max-width:760px; margin:0 auto; padding:48px 28px 110px; line-height:1.55; }
  h1 { font-size:30px; font-weight:normal; letter-spacing:.4px; margin:0 0 4px; }
  .sub { font-style:italic; color:#555; margin:0 0 26px; }
  .prog { font-size:14px; color:#555; margin-bottom:6px; }
  .bar { height:3px; background:#e7e3d7; margin-bottom:30px; }
  .bar > div { height:100%; background:#1a1a1a; width:0; transition:width .35s; }
  .ask { background:#f4f1e8; border-left:3px solid #1a1a1a; padding:12px 16px; margin-bottom:14px; }
  .ask .lbl { font-style:italic; color:#666; font-size:13px; display:block; margin-bottom:4px; }
  .resp { border:1px solid #d8d3c4; padding:14px 18px; white-space:pre-wrap; margin-bottom:18px; }
  .btns { display:flex; gap:12px; }
  .btn { flex:1; text-align:center; padding:12px; border:1px solid #1a1a1a; cursor:pointer; }
  .btn:hover { background:#1a1a1a; color:#fdfcf8; }
  .keep { } .drop { }
  .why { width:100%; font-family:inherit; font-size:15px; padding:9px 11px; box-sizing:border-box;
         border:1px solid #d8d3c4; background:#fff; margin-top:10px; }
  .why::placeholder { font-style:italic; color:#aaa; }
  .hint { font-size:13px; color:#999; margin-top:18px; text-align:center; }
  .done { text-align:center; padding-top:70px; }
  code { background:#f0ece0; padding:1px 5px; }
  .q { margin-bottom:22px; }
  .q label { display:block; font-size:16px; margin-bottom:7px; }
  .q label .note { font-style:italic; color:#888; font-size:13px; }
  .q textarea { width:100%; font-family:inherit; font-size:15px; line-height:1.5; padding:10px 12px;
                box-sizing:border-box; border:1px solid #d8d3c4; background:#fff; min-height:70px; resize:vertical; }
  .q textarea::placeholder { font-style:italic; color:#aaa; }
  .go { display:inline-block; padding:11px 22px; border:1px solid #1a1a1a; cursor:pointer; margin-top:4px; }
  .go:hover { background:#1a1a1a; color:#fdfcf8; }
  .skip { color:#999; font-size:13px; cursor:pointer; margin-left:18px; }
  .skip:hover { color:#1a1a1a; }
</style></head>
<body><div id="app"></div>
<script>
let items=[], cur=-1, total=0, stated={}, questions=[];
const app=document.getElementById('app');
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function done(){return items.filter(it=>it.verdict!==null).length;}
async function load(){
  const d=await (await fetch('/api/curation')).json();
  items=d.items; total=items.length; stated=d.stated||{}; questions=d.questions||[];
  if(!stated.done){ interview(); return; }
  beginCurating();
}
function beginCurating(){
  cur=items.findIndex(it=>it.verdict===null);
  cur===-1 ? finish() : render();
}
function interview(){
  const blocks=questions.map(q=>`<div class="q"><label>${esc(q.q)}</label>
    <textarea id="st_${q.key}" placeholder="${esc(q.ph||'')}"></textarea></div>`).join('');
  app.innerHTML=`<h1>First, in your words</h1>
  <p class="sub">Before you judge anything — say what you're looking for, in your own words. This is a weak
    prior; what you actually keep/drop is the real signal. The gap between them tells us something too.</p>
  ${blocks}
  <div><span class="go" onclick="submitStated()">Begin curating →</span>
    <span class="skip" onclick="submitStated()">skip the rest</span></div>`;
  window.scrollTo(0,0);
}
async function submitStated(){
  const body={};
  questions.forEach(q=>{ body[q.key]=(document.getElementById('st_'+q.key)||{}).value||''; });
  await fetch('/api/stated',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  stated.done=true; beginCurating();
}
async function mark(v){
  if(cur<0)return;
  const why=(document.getElementById('why')||{}).value||'';
  items[cur].verdict=v; items[cur].why=why;
  await fetch('/api/curate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({index:cur,verdict:v,why:why})});
  cur=items.findIndex(it=>it.verdict===null);
  cur===-1 ? finish() : render();
}
function render(){
  const it=items[cur], d=done();
  app.innerHTML=`<h1>Curate your standard</h1>
  <p class="sub">Mark the responses that are genuinely <em>your</em> kind of good. You're teaching the model what to learn from — once.</p>
  <div class="prog">You've curated ${d} of ${total}</div>
  <div class="bar"><div style="width:${(100*d/total).toFixed(1)}%"></div></div>
  <div class="ask"><span class="lbl">Your ask:</span>${esc(it.prompt)}</div>
  <div class="resp">${esc(it.response)}</div>
  <div class="btns"><div class="btn keep" onclick="mark('keep')">Keep — this is my standard</div>
    <div class="btn drop" onclick="mark('drop')">Drop — not my standard</div></div>
  <input class="why" id="why" autocomplete="off" placeholder="why? (optional — what makes it good, or what's off)">
  <div class="hint">keys: K (keep) · D (drop) — when not typing why</div>`;
  window.scrollTo(0,0);
}
async function finish(){
  const kept=items.filter(it=>it.verdict==='keep').length;
  app.innerHTML=`<div class="done"><h1>You've seeded your standard.</h1>
  <p class="sub">${kept} kept of ${total}. From here, the system learns from your normal use —<br>you won't have to do this again unless you want to.</p>
  <p id="sep" class="sub">measuring whether your standard separates in the latent space…</p>
  <p class="hint">Run <code>bespoke curate --summary</code> to review.</p></div>`;
  try{
    const s=await (await fetch('/api/separability')).json();
    const el=document.getElementById('sep');
    if(s.auc!==undefined){
      const read = s.auc>=0.72 ? 'clean — the automated latent-space eval is viable.'
        : s.auc>=0.62 ? 'above the ~0.62 auto-label baseline — the why-axes add the rest.'
        : 'faint — likely representation-limited.';
      el.innerHTML=`Your standard separates in the latent space at <b>AUC ${s.auc.toFixed(2)}</b> `
        +`(vs ~0.62 on the old auto-labels, ${s.n_keep} keep / ${s.n_drop} drop) — ${read}`;
    } else { el.textContent = s.error||''; }
  }catch(e){}
}
document.addEventListener('keydown',e=>{
  if(document.activeElement&&document.activeElement.id==='why')return;
  const k=e.key.toLowerCase();
  if(k==='k')mark('keep');else if(k==='d')mark('drop');});
load();
</script></body></html>"""


def serve_curation(n=40, port=8422, seed=0):
    """Build the curation set (once) and serve the browser curation page; each mark saves to disk."""
    import http.server
    import socketserver
    import webbrowser
    from urllib.parse import urlparse

    _DIR.mkdir(parents=True, exist_ok=True)
    if not _FILE.exists():
        items = select_curation_items(n=n, seed=seed)
        _FILE.write_text(json.dumps({"stated": {}, "items": items}, indent=2))
        print(f"Built curation set: {len(items)} of your real interactions.")

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
                self._send(200, _HTML, "text/html; charset=utf-8")
            elif p == "/api/curation":
                a = json.loads(_FILE.read_text())
                a["questions"] = _STATED_QUESTIONS  # single source of truth → page renders these
                self._send(200, json.dumps(a))
            elif p == "/api/separability":
                self._send(200, json.dumps(curation_separability()))
            else:
                self._send(404, "{}")

        def do_POST(self):
            p = urlparse(self.path).path
            n_ = int(self.headers.get("Content-Length", 0) or 0)
            data = json.loads(self.rfile.read(n_) or b"{}") if p in ("/api/curate", "/api/stated") else {}
            if p == "/api/curate":
                a = json.loads(_FILE.read_text())
                i = int(data["index"])
                a["items"][i]["verdict"] = data["verdict"]
                a["items"][i]["why"] = (data.get("why") or "").strip()
                _FILE.write_text(json.dumps(a, indent=2))
                self._send(200, json.dumps({"ok": True}))
            elif p == "/api/stated":
                save_stated(data)
                self._send(200, json.dumps({"ok": True}))
            else:
                self._send(404, "{}")

    url = f"http://localhost:{port}/"
    print(f"Curation page: {url}\n  open it, mark Keep/Drop (or K/D) + optional why — each saves instantly. Ctrl-C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    class _Server(socketserver.TCPServer):
        allow_reuse_address = True  # restartable without waiting out TIME_WAIT on the port

    with _Server(("127.0.0.1", port), H) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped. Review with:  bespoke curate --summary")


def curation_separability():
    """Does your CURATED keep/drop separate in the latent space? (the RT-003 test, but on YOUR
    deliberate labels instead of the noisy ~0.62 followup-heuristic labels). Cross-validated AUC.
    """
    import numpy as np
    if not _FILE.exists():
        return {"error": "no curation yet"}
    a = json.loads(_FILE.read_text())
    keep = [it["id"] for it in a["items"] if it["verdict"] == "keep"]
    drop = [it["id"] for it in a["items"] if it["verdict"] == "drop"]
    if len(keep) < 3 or len(drop) < 3:
        return {"error": f"need ≥3 keep AND ≥3 drop to measure (have {len(keep)} keep / {len(drop)} drop)"}
    ids = keep + drop
    conn = get_connection()
    q = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT i.id AS id, v.interaction_embedding AS emb FROM interactions i "
        f"JOIN vec_interactions v ON v.rowid = i.id WHERE i.id IN ({q})", ids).fetchall()
    conn.close()
    em = {r["id"]: np.frombuffer(r["emb"], dtype=np.float32) for r in rows}
    keep = [i for i in keep if i in em]
    drop = [i for i in drop if i in em]
    X = np.vstack([em[i] for i in keep + drop]).astype(np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    y = np.array([1] * len(keep) + [0] * len(drop))
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    cv = max(2, min(5, len(keep), len(drop)))
    auc = float(cross_val_score(
        LogisticRegression(max_iter=2000, class_weight="balanced"), X, y, cv=cv, scoring="roc_auc").mean())
    return {"auc": auc, "n_keep": len(keep), "n_drop": len(drop), "baseline": 0.62, "cv": cv}


def summarize_curation():
    if not _FILE.exists():
        print("No curation yet. Run `bespoke curate` first.")
        return
    a = json.loads(_FILE.read_text())
    items = a["items"]
    kept = [it for it in items if it["verdict"] == "keep"]
    dropped = [it for it in items if it["verdict"] == "drop"]
    rated = len(kept) + len(dropped)
    print(f"Curation: {rated}/{len(items)} rated — {len(kept)} keep, {len(dropped)} drop")
    stated = a.get("stated", {})
    if stated.get("done"):
        print("\nStated rubric (what you said up front — the weak prior):")
        for q in _STATED_QUESTIONS:
            v = (stated.get(q["key"]) or "").strip()
            if v:
                print(f"   {q['key']}: {v}")
    whys = [it["why"] for it in items if it.get("why")]
    print(f"  {len(whys)} reasons captured (seed for the WHY-rubric)")
    for w in whys[:8]:
        print(f"   • {w}")
    sep = curation_separability()
    if "auc" in sep:
        print(f"\nLatent-space separability of YOUR labels: AUC {sep['auc']:.2f} "
              f"(cv={sep['cv']}, {sep['n_keep']} keep / {sep['n_drop']} drop) — vs ~0.62 on the noisy auto-labels.")
        print("  >=0.72: clean (automated eval viable) | ~0.62: why-axes needed | <0.62: representation-limited")
    else:
        print(f"\nseparability: {sep.get('error')}")
