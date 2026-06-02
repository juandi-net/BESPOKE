"""Source-retention guard.

AI tools auto-delete their local history. Claude Code's `cleanupPeriodDays` defaults to 30
days — it removes session transcripts older than that at startup. If BESPOKE doesn't capture an
interaction before that window closes, the raw source is gone forever. So BESPOKE actively keeps
the retention window wide open.

Disk tradeoff: transcripts are small plain-text JSONL, so even years of history is modest (tens
to low-hundreds of MB) — cheap insurance against losing the data the whole system is built on.
The threshold is configurable.
"""
import json
from pathlib import Path

# ~1000 years ≈ "keep forever". Plenty of headroom; transcripts are tiny text.
DEFAULT_MIN_DAYS = 365000


def ensure_claude_retention(settings_path=None, min_days=DEFAULT_MIN_DAYS):
    """Ensure Claude Code's cleanupPeriodDays is at least `min_days`.

    Returns (changed: bool, value). changed=True means we raised it. No-op (changed=False) if it
    was already high enough, or if the settings file doesn't exist.
    """
    path = Path(settings_path) if settings_path else Path.home() / ".claude" / "settings.json"
    if not path.exists():
        return (False, None)
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return (False, None)

    current = data.get("cleanupPeriodDays")
    if isinstance(current, int) and current >= min_days:
        return (False, current)

    data["cleanupPeriodDays"] = min_days
    path.write_text(json.dumps(data, indent=2) + "\n")
    return (True, min_days)
