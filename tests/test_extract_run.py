# tests/test_extract_run.py
"""Track 2 / go-local: geometric extract orchestrator end-to-end (no cloud)."""
from unittest.mock import patch, MagicMock

from sqlite_vec import serialize_float32


class TestCliRoutesLocal:
    def test_cmd_extract_uses_geometric_when_flag_off(self):
        from bespoke import cli
        stats = {"interactions_labeled": 0, "pairs_written": 0,
                 "tangled_sessions": 0, "probe_trained": False}
        with patch("bespoke.extract.run.run_geometric_extract", return_value=stats) as geo, \
             patch("bespoke.teach.stage2a.run_stage_2a") as llm, \
             patch("bespoke.config.config") as cfg:
            cfg.pipeline.use_llm_extract = False
            cli.cmd_extract(MagicMock(reset=False))
        geo.assert_called_once()
        llm.assert_not_called()


def _ins(db, sid, ts, um, ar, followup, emb):
    cur = db.execute(
        "INSERT INTO interactions (provider,model,source,session_id,user_message,"
        "assistant_response,user_followup,captured_at) VALUES ('c','m','s',?,?,?,?,?)",
        (sid, um, ar, followup, ts))
    rid = cur.lastrowid
    db.execute("INSERT INTO vec_interactions (rowid,interaction_embedding) VALUES (?,?)",
               (rid, serialize_float32(emb)))
    return rid


GOOD = [1.0] + [0.0] * 767
BAD = [-1.0] + [0.0] * 767


class TestGeometricExtract:
    def test_end_to_end_local(self, db):
        from bespoke.extract.run import run_geometric_extract
        for i in range(6):
            _ins(db, "s1", f"2026-06-02T00:0{i}:00Z", "q", "a great answer", "perfect, do it", GOOD)
        for i in range(6):
            _ins(db, "s2", f"2026-06-02T01:0{i}:00Z", "q2", "a bad answer", "no that's wrong, fix it", BAD)

        stats = run_geometric_extract(conn=db)

        accepts = db.execute("SELECT COUNT(*) FROM interactions WHERE feedback_class IN "
                             "('accept','strong_accept')").fetchone()[0]
        rejects = db.execute("SELECT COUNT(*) FROM interactions WHERE feedback_class IN "
                             "('reject','strong_reject')").fetchone()[0]
        assert accepts == 6 and rejects == 6
        # every interaction got a quality_score + domain + processed
        assert db.execute("SELECT COUNT(*) FROM interactions WHERE quality_score IS NULL").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM interactions WHERE domain IS NULL").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM interactions WHERE processed_2a_at IS NULL").fetchone()[0] == 0
        # pairs only from the 6 non-rejected (rejects excluded)
        assert db.execute("SELECT COUNT(*) FROM training_pairs").fetchone()[0] == 6
        assert stats["pairs_written"] == 6
        assert stats["probe_trained"] is True

    def test_junk_excluded_and_agentic_tool_blocks_stripped(self, db):
        """observer/tool_result_only never become pairs; agentic pairs keep reasoning, drop dumps."""
        from bespoke.extract.run import run_geometric_extract
        # clean accepted prose -> a normal pair
        _ins(db, "s1", "2026-06-02T00:00:00Z", "real question", "a thoughtful answer", "perfect thanks", GOOD)
        # observer junk, accepted-looking -> must be EXCLUDED despite the accept signal
        _ins(db, "s2", "2026-06-02T00:00:00Z", "You are a Claude-Mem observer",
             "<observation>noted</observation>", "ok good", GOOD)
        # agentic: real reasoning + tool dump -> pair kept, blocks collapsed to [N tool calls]
        agentic_ar = ('<thinking>plan</thinking>'
                      '<tool_use name="Bash">{"command":"ls"}</tool_use><tool_result>f1</tool_result>'
                      ' final answer')
        _ins(db, "s3", "2026-06-02T00:00:00Z", "do a task", agentic_ar, "nice thanks", GOOD)

        run_geometric_extract(conn=db)

        responses = [r["response"] for r in
                     db.execute("SELECT response FROM training_pairs").fetchall()]
        # observer never becomes a training pair
        assert not any("<observation>" in r for r in responses)
        # no raw tool syntax survives into training data
        assert not any("<tool_use" in r or "<tool_result" in r for r in responses)
        # the agentic pair is kept, reasoning preserved, dump summarized
        agentic = [r for r in responses if "final answer" in r]
        assert len(agentic) == 1
        assert "[1 tool call]" in agentic[0]
        assert "<thinking>plan</thinking>" in agentic[0]

    def test_incremental_no_duplicate_pairs(self, db):
        from bespoke.extract.run import run_geometric_extract
        for i in range(5):
            _ins(db, "s1", f"2026-06-02T00:0{i}:00Z", "q", "a", "ok thanks", GOOD)
        run_geometric_extract(conn=db)
        p1 = db.execute("SELECT COUNT(*) FROM training_pairs").fetchone()[0]
        run_geometric_extract(conn=db)  # nothing unprocessed now
        p2 = db.execute("SELECT COUNT(*) FROM training_pairs").fetchone()[0]
        assert p1 == p2 and p1 == 5

    def test_fluffy_response_is_taste_demoted(self, db):
        """A response carrying juandi's negative tells (emoji/sycophancy) is demoted one bucket so
        clear fluff drops out of the high+medium training set — the adapter shouldn't learn to emit it."""
        from bespoke.extract.run import run_geometric_extract
        # clean accepted prose -> stays high (fast-accept chain)
        for i in range(5):
            _ins(db, "clean", f"2026-06-02T00:0{i}:00Z", "real q", "a direct, useful answer",
                 "perfect, do it", GOOD)
        # a fluffy accepted response in its own self-contained session
        _ins(db, "fluff", "2026-06-02T02:00:00Z", "real q",
             "Great question! 🚀 You're absolutely right!", "perfect, do it", GOOD)
        _ins(db, "fluff", "2026-06-02T02:01:00Z", "next", "ok", "thanks", GOOD)

        stats = run_geometric_extract(conn=db)

        assert stats["taste_demoted"] >= 1
        fluffy = db.execute(
            "SELECT quality_score FROM training_pairs WHERE response LIKE '%🚀%'").fetchone()
        assert fluffy is not None and fluffy["quality_score"] != "high"


    def test_taste_demoted_stat_counts_tangled_drops(self, db):
        """In a tangled session (kept conservatively: high only) a demoted fluffy pair is DROPPED —
        right behavior, but the stat must still count the demotion or the extract under-reports
        how much fluff the taste filter caught (found live: 76 reported vs ~900 actual)."""
        from bespoke.extract.run import run_geometric_extract
        # 8 turns, last followup is a reject -> not ended_in_accept -> tangled
        for i in range(7):
            fu = "perfect, do it" if i != 1 else "perfect, do it"
            _ins(db, "tg", f"2026-06-02T00:0{i}:00Z", f"q{i}",
                 "Great question! 🚀 You're absolutely right!" if i == 1 else "a direct answer",
                 fu, GOOD)
        _ins(db, "tg", "2026-06-02T00:07:00Z", "q7", "a", "no that's wrong, fix it", GOOD)

        stats = run_geometric_extract(conn=db)

        assert stats["taste_demoted"] >= 1  # the fluffy turn was demoted (then tangled-dropped)
        fluffy = db.execute("SELECT 1 FROM training_pairs WHERE response LIKE '%🚀%'").fetchone()
        assert fluffy is None               # ...and correctly kept OUT of the training set


class TestTasteDemote:
    def test_clean_response_keeps_bucket(self):
        from bespoke.extract.run import taste_demote
        assert taste_demote("high", "Set the flag to false and redeploy.") == "high"

    def test_fluffy_response_drops_one_bucket(self):
        from bespoke.extract.run import taste_demote
        assert taste_demote("high", "Great question! 🚀 You're absolutely right!") == "medium"

    def test_low_stays_low(self):
        from bespoke.extract.run import taste_demote
        assert taste_demote("low", "🚀🚀🚀 amazing!!!") == "low"
