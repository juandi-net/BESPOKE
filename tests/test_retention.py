# tests/test_retention.py
"""Source-retention guard: keep Claude Code from auto-deleting transcripts."""
import json


class TestEnsureClaudeRetention:
    def test_sets_when_missing(self, tmp_path):
        from bespoke.capture.retention import ensure_claude_retention, DEFAULT_MIN_DAYS
        p = tmp_path / "settings.json"
        p.write_text(json.dumps({"theme": "light"}))
        changed, val = ensure_claude_retention(settings_path=p, min_days=DEFAULT_MIN_DAYS)
        assert changed is True and val == DEFAULT_MIN_DAYS
        assert json.loads(p.read_text())["cleanupPeriodDays"] == DEFAULT_MIN_DAYS
        # preserves existing keys
        assert json.loads(p.read_text())["theme"] == "light"

    def test_raises_when_too_low(self, tmp_path):
        from bespoke.capture.retention import ensure_claude_retention
        p = tmp_path / "settings.json"
        p.write_text(json.dumps({"cleanupPeriodDays": 30}))
        changed, val = ensure_claude_retention(settings_path=p, min_days=365000)
        assert changed is True and val == 365000

    def test_noop_when_already_high(self, tmp_path):
        from bespoke.capture.retention import ensure_claude_retention
        p = tmp_path / "settings.json"
        p.write_text(json.dumps({"cleanupPeriodDays": 500000}))
        changed, val = ensure_claude_retention(settings_path=p, min_days=365000)
        assert changed is False and val == 500000

    def test_noop_when_no_file(self, tmp_path):
        from bespoke.capture.retention import ensure_claude_retention
        changed, val = ensure_claude_retention(settings_path=tmp_path / "nope.json")
        assert changed is False and val is None
