"""Stated-preferences interview: free-text rubric saved alongside keep/drop in curation.json.

The stated rubric is a weak prior (what juandi SAYS makes a response good/bad); the revealed
keep/drop + why is ground truth. The gap between them is itself signal (next-steps.md #1).
"""
import json

import pytest

from bespoke.benchmark import curate


@pytest.fixture
def curfile(tmp_path, monkeypatch):
    """Point curate at a throwaway curation dir/file."""
    d = tmp_path / "curation"
    monkeypatch.setattr(curate, "_DIR", d)
    monkeypatch.setattr(curate, "_FILE", d / "curation.json")
    return curate._FILE


def test_get_stated_empty_when_no_file(curfile):
    assert curate.get_stated() == {}


def test_save_stated_persists_trimmed_and_marks_done(curfile):
    curate.save_stated({"good": "  direct, no fluff  ", "bad": "sycophancy ", "voice": ""})
    s = curate.get_stated()
    assert s["good"] == "direct, no fluff"
    assert s["bad"] == "sycophancy"
    assert s["voice"] == ""
    assert s["done"] is True


def test_save_stated_preserves_existing_keep_drop_items(curfile):
    # juandi's real case: 40 rated items already on disk, no "stated" key yet.
    curfile.parent.mkdir(parents=True, exist_ok=True)
    curfile.write_text(json.dumps({"items": [
        {"id": 1, "prompt": "p", "response": "r", "verdict": "keep", "why": "good"},
        {"id": 2, "prompt": "p", "response": "r", "verdict": "drop", "why": ""},
    ]}))
    curate.save_stated({"good": "signal over noise", "bad": "emojis"})
    a = json.loads(curfile.read_text())
    assert len(a["items"]) == 2
    assert a["items"][0]["verdict"] == "keep"
    assert a["stated"]["good"] == "signal over noise"


def test_save_stated_ignores_unknown_fields(curfile):
    curate.save_stated({"good": "x", "injected": "nope"})
    s = curate.get_stated()
    assert "injected" not in s
    assert s["good"] == "x"


def test_save_stated_persists_expanded_open_fields(curfile):
    # the seed moment: more open prompts than just good/bad/voice (juandi, 2026-06-04).
    curate.save_stated({
        "good": "g", "tradeoffs": "brevity wins unless it breaks",
        "context": "depth when it's novel", "level": "noob at ml",
        "pushback": "just tell me i'm wrong", "loved": "the diesel answer", "hated": "great question!",
    })
    s = curate.get_stated()
    assert s["tradeoffs"] == "brevity wins unless it breaks"
    assert s["context"] == "depth when it's novel"
    assert s["level"] == "noob at ml"
    assert s["pushback"] == "just tell me i'm wrong"
    assert s["loved"] == "the diesel answer"
    assert s["hated"] == "great question!"


def test_stated_questions_are_the_single_source_of_fields(curfile):
    # every served question maps to a saved field, and vice-versa — no drift.
    keys = [q["key"] for q in curate._STATED_QUESTIONS]
    assert set(keys) == set(curate._STATED_FIELDS)
    assert {"good", "bad", "voice", "tradeoffs", "context", "level", "pushback"} <= set(keys)
    for q in curate._STATED_QUESTIONS:
        assert q["q"].strip()  # every prompt has text


def test_summarize_prints_stated_rubric(curfile, capsys):
    curfile.parent.mkdir(parents=True, exist_ok=True)
    curfile.write_text(json.dumps({
        "stated": {"good": "just do it", "bad": "too soft, too much noise", "done": True},
        "items": [{"id": 1, "prompt": "p", "response": "r", "verdict": "keep", "why": ""}],
    }))
    curate.summarize_curation()
    out = capsys.readouterr().out
    assert "just do it" in out
    assert "too soft, too much noise" in out
