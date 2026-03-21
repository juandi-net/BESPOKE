# bespoke/capture/web_capture.py
"""Capture claude.ai conversations via the desktop app's session cookie."""

import json
import time
import logging
import requests

from datetime import datetime, timezone
from typing import List

from bespoke.capture.parsers import Interaction
from bespoke.capture.pipeline import capture_interaction
from bespoke.db.init import get_connection


def _extract_assistant_text(message: dict) -> str:
    """Extract text from an assistant message's content blocks.

    Assistant messages use a `content` array: [{"type": "text", "text": "..."}].
    Falls back to `text` field for older format.
    """
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    # Fallback: older format with top-level text field
    return message.get("text", "")


def _extract_human_text(message: dict) -> str:
    """Extract text from a human message.

    Human messages typically have a top-level `text` field.
    Falls back to content blocks if present.
    """
    text = message.get("text")
    if text is not None:
        return text
    # Fallback: content blocks (same format as assistant)
    return _extract_assistant_text(message)


def parse_conversation(conv: dict) -> List[Interaction]:
    """Parse a claude.ai conversation into BESPOKE Interactions.

    Follows the active branch only (via current_leaf_message_uuid).
    Pairs human->assistant messages into Interactions.
    Backfills user_followup for Stage 2a extraction quality.
    """
    messages = conv.get("chat_messages", [])
    if not messages:
        return []

    # Build lookup by UUID (defensive: skip messages without uuid)
    by_uuid = {m["uuid"]: m for m in messages if "uuid" in m}

    # Reconstruct active branch from leaf to root
    leaf_uuid = conv.get("current_leaf_message_uuid")
    if not leaf_uuid or leaf_uuid not in by_uuid:
        # Fallback: use messages in index order
        branch = sorted(messages, key=lambda m: m.get("index", 0))
    else:
        branch = []
        current = leaf_uuid
        while current and current in by_uuid:
            branch.append(by_uuid[current])
            current = by_uuid[current].get("parent_message_uuid")
        branch.reverse()  # root to leaf

    # Pair human->assistant
    interactions = []
    model = conv.get("model", "unknown")
    session_id = conv.get("uuid", "")

    i = 0
    while i < len(branch) - 1:
        if branch[i].get("sender") == "human" and branch[i + 1].get("sender") == "assistant":
            human_text = _extract_human_text(branch[i])
            assistant_text = _extract_assistant_text(branch[i + 1])
            if human_text and assistant_text:
                interactions.append(Interaction(
                    provider="claude",
                    model=model,
                    source="claude-web",
                    session_id=session_id,
                    system_prompt=None,
                    user_message=human_text,
                    assistant_response=assistant_text,
                    input_tokens=None,
                    output_tokens=None,
                    timestamp=branch[i].get("created_at", datetime.now().isoformat()),
                ))
            i += 2
        else:
            i += 1

    # Backfill user_followup
    for j in range(len(interactions) - 1):
        interactions[j].user_followup = interactions[j + 1].user_message

    return interactions


logger = logging.getLogger(__name__)


def _headers(cookies: dict) -> dict:
    """Build request headers with all cookies from Claude Desktop."""
    cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Cookie": cookie_str,
    }


def get_session_cookie() -> dict:
    """Extract and decrypt cookies from Claude Desktop's cookie store.

    Claude Desktop is an Electron app that stores its encryption key under
    'Claude Safe Storage' in the macOS Keychain (not Chrome's key).

    Returns all cookies as a dict (the API requires more than just __ssid).
    """
    import os
    import subprocess
    from pycookiecheat import chrome_cookies

    # Get Claude's encryption key from Keychain (not Chrome's)
    result = subprocess.run(
        ["security", "find-generic-password", "-s", "Claude Safe Storage", "-w"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not read 'Claude Safe Storage' from Keychain. "
            "Is Claude Desktop installed and logged in?"
        )
    password = result.stdout.strip()

    cookie_db = os.path.expanduser("~/Library/Application Support/Claude/Cookies")
    cookies = chrome_cookies(
        "https://claude.ai",
        cookie_file=cookie_db,
        password=password,
    )

    if not cookies.get("sessionKey"):
        raise RuntimeError("No sessionKey cookie found. Is Claude Desktop logged in?")
    return cookies


def get_org_id(session_cookie: dict) -> str:
    """Get the organization UUID."""
    r = requests.get(
        "https://claude.ai/api/organizations",
        headers=_headers(session_cookie),
    )
    if r.status_code == 401:
        raise RuntimeError("Session expired. Open Claude Desktop to refresh.")
    r.raise_for_status()
    return r.json()[0]["uuid"]


def list_conversations(session_cookie: dict, org_id: str) -> list:
    """List all conversations, newest first."""
    r = requests.get(
        f"https://claude.ai/api/organizations/{org_id}/chat_conversations",
        headers=_headers(session_cookie),
    )
    if r.status_code == 401:
        raise RuntimeError("Session expired. Open Claude Desktop to refresh.")
    r.raise_for_status()
    return r.json()


def fetch_conversation(session_cookie: dict, org_id: str, conv_id: str) -> dict:
    """Fetch a single conversation with all messages.

    Retries on 429 with exponential backoff (2s, 4s, 8s).
    """
    url = f"https://claude.ai/api/organizations/{org_id}/chat_conversations/{conv_id}"
    for attempt in range(4):  # 1 initial + 3 retries
        r = requests.get(url, headers=_headers(session_cookie))
        if r.status_code == 401:
            raise RuntimeError("Session expired. Open Claude Desktop to refresh.")
        if r.status_code == 429 and attempt < 3:
            delay = 2 ** (attempt + 1)  # 2, 4, 8
            logger.warning(f"Rate limited (429). Retrying in {delay}s...")
            time.sleep(delay)
            continue
        r.raise_for_status()
        return r.json()


def run_web_capture() -> dict:
    """Capture new/updated claude.ai conversations.

    Incremental: only fetches conversations updated since last run.
    Per-conversation checkpointing: safe to interrupt and resume.
    """
    stats = {
        "conversations_checked": 0,
        "conversations_fetched": 0,
        "interactions_captured": 0,
        "interactions_skipped": 0,
    }

    cookie = get_session_cookie()
    org_id = get_org_id(cookie)

    conn = get_connection()

    try:
        # Get last capture timestamp
        row = conn.execute(
            "SELECT last_successful_run FROM pipeline_state WHERE stage = 'web_capture'"
        ).fetchone()
        last_run = None
        if row and row["last_successful_run"]:
            last_run = datetime.fromisoformat(row["last_successful_run"])

        conversations = list_conversations(cookie, org_id)
        stats["conversations_checked"] = len(conversations)

        for conv_summary in conversations:
            updated = conv_summary.get("updated_at", "")
            if last_run and updated:
                try:
                    # Normalize: strip Z suffix, treat as UTC (matches our UTC checkpoint)
                    updated_str = updated.rstrip("Z")
                    conv_updated = datetime.fromisoformat(updated_str).replace(tzinfo=timezone.utc)
                    if conv_updated <= last_run:
                        continue
                except ValueError:
                    pass  # Can't parse date — fetch to be safe

            # Fetch full conversation
            conv = fetch_conversation(cookie, org_id, conv_summary["uuid"])
            stats["conversations_fetched"] += 1

            # Parse and ingest
            interactions = parse_conversation(conv)
            for interaction in interactions:
                capture_interaction(conn, interaction, stats=stats)

            # Per-conversation checkpoint (UTC for consistent timezone handling)
            conn.execute(
                "INSERT OR REPLACE INTO pipeline_state (stage, last_successful_run, metadata) "
                "VALUES ('web_capture', ?, ?)",
                (datetime.now(timezone.utc).isoformat(), json.dumps(stats)),
            )
            conn.commit()

            # Rate limit
            time.sleep(0.5)
    finally:
        conn.close()

    return stats
