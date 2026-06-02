# tests/test_extract_cache.py
"""Track 2: capture the prompt-cache signal (presence detection)."""
import json


class TestJsonlIngestCache:
    def test_reads_cache_fields(self, tmp_path):
        from bespoke.capture.parsers import ingest_extraction_jsonl
        p = tmp_path / "x.jsonl"
        p.write_text(json.dumps({
            "user_message": "u", "assistant_response": "a",
            "cache_read_input_tokens": 120, "cache_creation_input_tokens": 30,
        }) + "\n")
        out = ingest_extraction_jsonl(p)
        assert out[0].cache_read_tokens == 120
        assert out[0].cache_creation_tokens == 30

    def test_defaults_none_when_absent(self, tmp_path):
        from bespoke.capture.parsers import ingest_extraction_jsonl
        p = tmp_path / "y.jsonl"
        p.write_text(json.dumps({"user_message": "u", "assistant_response": "a"}) + "\n")
        out = ingest_extraction_jsonl(p)
        assert out[0].cache_read_tokens is None
        assert out[0].cache_creation_tokens is None


class TestPipelinePersistsCache:
    def test_persists_cache_fields(self, db):
        from bespoke.capture.parsers import Interaction
        from bespoke.capture.pipeline import capture_interaction
        it = Interaction(
            provider="claude", model="m", source="s", session_id="sess",
            system_prompt=None, user_message="u", assistant_response="a",
            input_tokens=10, output_tokens=5, timestamp="2026-06-02T00:00:00Z",
            cache_read_tokens=120, cache_creation_tokens=30,
        )
        rid = capture_interaction(db, it, embedding_svc=None, stats=None)
        row = db.execute(
            "SELECT cache_read_tokens, cache_creation_tokens FROM interactions WHERE id=?",
            (rid,)).fetchone()
        assert row["cache_read_tokens"] == 120
        assert row["cache_creation_tokens"] == 30
