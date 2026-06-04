"""Geometric Stage 2a — local, deterministic extract (replaces the cloud LLM classifier).

Pipeline, all local:
  1. feedback_class  ← deterministic rules over user_followup (seeds everything)
  2. quality_score   ← Track 1 linear preference probe (trained on feedback labels), with a
                       time-to-accept nudge; feedback-based fallback when too few labels
  3. domain          ← 'general' for V0 (single adapter); fleet-era enables Leiden domains
  4. training_pairs  ← mechanical: segment each session into conversations, COPY non-rejected
                       self-contained turns into SFT pairs; tangled sessions kept conservatively

Incremental: processes interactions WHERE processed_2a_at IS NULL. Use `bespoke extract --reset`
for a full re-run. No cloud, no LLM.
"""
from datetime import datetime

from bespoke.db.init import get_connection
from bespoke.extract.feedback import classify_feedback
from bespoke.extract.content_type import classify_content, clean_tool_blocks, strip_conductor_boilerplate
from bespoke.extract.conversations import assemble_sessions, segment_conversation, _gap_seconds
from bespoke.extract.mechanical import is_tangled
from bespoke.eval.signals import get_labeled_embeddings
from bespoke.eval.probe import LinearPreferenceProbe

REJECT = {"reject", "strong_reject"}
QUALITY_FALLBACK = {"strong_accept": "high", "accept": "high", "neutral": "medium",
                    "reject": "low", "strong_reject": "exclude"}
FAST_ACCEPT_SECONDS = 180  # accepted within 3 min of the next turn = fast accept = quality boost


def _bucket(score):
    if score >= 0.66:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def run_geometric_extract(conn=None):
    """Run the fully-local geometric extract. Returns a stats dict."""
    close = conn is None
    conn = conn or get_connection()
    stats = {"interactions_labeled": 0, "pairs_written": 0, "tangled_sessions": 0,
             "probe_trained": False}

    unprocessed = {r["id"] for r in conn.execute(
        "SELECT id FROM interactions WHERE processed_2a_at IS NULL").fetchall()}
    if not unprocessed:
        if close:
            conn.close()
        return stats

    # 1. feedback_class from user_followup (unprocessed only)
    for r in conn.execute(
            "SELECT id, user_followup FROM interactions WHERE processed_2a_at IS NULL").fetchall():
        conn.execute("UPDATE interactions SET feedback_class=? WHERE id=?",
                     (classify_feedback(r["user_followup"]), r["id"]))
    conn.commit()

    # 2. train the preference probe on the WHOLE labeled corpus (revealed taste)
    ids, X, y = get_labeled_embeddings(conn)
    probe = None
    if len(X):
        labeled = y != -1
        if labeled.sum() >= 4 and len(set(y[labeled].tolist())) == 2:
            probe = LinearPreferenceProbe().fit(X[labeled], y[labeled])
            stats["probe_trained"] = True

    # 3. quality per interaction id (probe score → bucket; feedback fallback)
    quality = {}
    if probe is not None:
        for iid, s in zip(ids.tolist(), probe.score(X).tolist()):
            quality[iid] = _bucket(s)
    fb = {r["id"]: r["feedback_class"]
          for r in conn.execute("SELECT id, feedback_class FROM interactions").fetchall()}
    for iid, f in fb.items():
        quality.setdefault(iid, QUALITY_FALLBACK.get(f, "medium"))

    # 4. write interactions (unprocessed): quality, domain, processed_2a_at
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    for iid in unprocessed:
        conn.execute(
            "UPDATE interactions SET quality_score=?, "
            "domain=COALESCE(domain,'general'), processed_2a_at=? WHERE id=?",
            (quality.get(iid, "medium"), now, iid))
        stats["interactions_labeled"] += 1
    conn.commit()

    # 5. per-session: segment + mechanically write pairs (unprocessed turns only)
    for s in assemble_sessions(conn):
        turns = s["turns"]
        tangled = is_tangled(turns)
        if tangled:
            stats["tangled_sessions"] += 1
        tids = [t["id"] for t in turns]
        qm = ",".join("?" * len(tids))
        textmap = {r["id"]: (r["user_message"], r["assistant_response"], r["feedback_class"])
                   for r in conn.execute(
                       f"SELECT id, user_message, assistant_response, feedback_class "
                       f"FROM interactions WHERE id IN ({qm})", tids).fetchall()}
        turns_t = [{**t,
                    "user_message": textmap.get(t["id"], ("", "", None))[0],
                    "assistant_response": textmap.get(t["id"], ("", "", None))[1],
                    "feedback_class": textmap.get(t["id"], ("", "", None))[2]}
                   for t in turns]

        for seg in segment_conversation(turns_t):
            for i, t in enumerate(seg):
                if t["id"] not in unprocessed:
                    continue
                if t["feedback_class"] in REJECT:
                    continue
                # content-type filter: drop observer/tool-dump junk; strip raw tool blocks from agentic.
                ctype, _ = classify_content(t["user_message"], t["assistant_response"])
                if ctype in ("observer", "tool_result_only"):
                    continue
                # strip injected app boilerplate (Conductor <system_instruction>) -> keep the real ask;
                # pure-boilerplate turns become empty here and are dropped by the `not instr` check below.
                instr = (strip_conductor_boilerplate(t["user_message"]) or "").strip()
                resp = (t["assistant_response"] or "").strip()
                if ctype == "agentic":
                    resp = (clean_tool_blocks(resp) or "").strip()
                if not instr or not resp:
                    continue
                q = quality.get(t["id"], "medium")
                # time-to-accept nudge: accepted AND fast → boost to high
                if t["feedback_class"] in ("accept", "strong_accept") and i + 1 < len(seg):
                    if 0 < _gap_seconds(t, seg[i + 1]) <= FAST_ACCEPT_SECONDS:
                        q = "high"
                # tangled sessions: keep only high-quality pairs (conservative selection)
                if tangled and q != "high":
                    continue
                conn.execute(
                    "INSERT INTO training_pairs (interaction_id, pair_type, domain, "
                    "instruction, response, quality_score) VALUES (?, 'sft', ?, ?, ?, ?)",
                    (t["id"], t.get("domain") or "general", instr, resp, q))
                stats["pairs_written"] += 1
    conn.commit()

    if close:
        conn.close()
    return stats
