# tests/test_geometric_interview.py
"""Geometric (local, no-LLM) benchmark interview."""
import numpy as np
import yaml
from sqlite_vec import serialize_float32


def _ins(db, um, ar, fc, vec):
    cur = db.execute("INSERT INTO interactions (provider,model,source,user_message,"
                     "assistant_response,feedback_class) VALUES ('c','m','s',?,?,?)",
                     (um, ar, fc))
    rid = cur.lastrowid
    db.execute("INSERT INTO vec_interactions (rowid,interaction_embedding) VALUES (?,?)",
               (rid, serialize_float32(vec)))
    return rid


class TestRetrieveMatches:
    def test_returns_nearest(self, db):
        from bespoke.benchmark.geometric_interview import retrieve_matches
        _ins(db, "near", "good ans", "accept", [1.0] + [0.0] * 767)
        _ins(db, "far", "bad ans", "reject", [-1.0] + [0.0] * 767)
        matches = retrieve_matches(np.array([1.0] + [0.0] * 767, np.float32), db, k=1)
        assert len(matches) == 1
        assert matches[0]["user_message"] == "near"
        assert matches[0]["feedback_class"] == "accept"


class TestAlignment:
    def test_fraction_accepted(self):
        from bespoke.benchmark.geometric_interview import alignment_score
        matches = [{"feedback_class": "accept"}, {"feedback_class": "reject"},
                   {"feedback_class": "strong_accept"}, {"feedback_class": "neutral"}]
        assert alignment_score(matches) == 0.5
        assert alignment_score([]) == 0.0


class TestAnchorsAndWrite:
    def test_build_and_write(self, tmp_path):
        from bespoke.benchmark.geometric_interview import build_anchor_examples, write_geometric_benchmark
        anchors = build_anchor_examples(["a crisp answer", "  "], ["a vague one"])
        assert anchors == {"good": ["a crisp answer"], "bad": ["a vague one"]}
        p = write_geometric_benchmark(anchors, 0.75, path=tmp_path / "benchmark.yaml")
        data = yaml.safe_load(p.read_text())
        assert data["benchmark"]["anchor_examples"]["good"] == ["a crisp answer"]
        assert data["benchmark"]["stated_vs_revealed_alignment"] == 0.75
        assert data["benchmark"]["source"] == "geometric_interview"


class TestDriver:
    def test_run_produces_benchmark(self, db, tmp_path):
        from bespoke.benchmark.geometric_interview import run_geometric_interview
        _ins(db, "q", "a great accepted answer", "accept", [1.0] + [0.0] * 767)

        answers = iter([
            "I build local AI systems",          # domains (open)
            "clear, decomposed, explains why",   # good_example
            "vague hand-wavy filler",            # bad_example
            "depth and directness",              # values (open)
        ])
        fake_svc = type("S", (), {
            "embed": lambda self, t, prefix="document": (np.array([1.0] + [0.0] * 767, np.float32), 1)
        })()
        out = []
        path = run_geometric_interview(
            conn=db, input_fn=lambda _: next(answers), output_fn=out.append,
            embedding_svc=fake_svc, benchmark_path=tmp_path / "benchmark.yaml")
        data = yaml.safe_load(path.read_text())
        assert data["benchmark"]["anchor_examples"]["good"] == ["clear, decomposed, explains why"]
        assert data["benchmark"]["anchor_examples"]["bad"] == ["vague hand-wavy filler"]
        # retrieved an accepted match → alignment > 0
        assert data["benchmark"]["stated_vs_revealed_alignment"] > 0
