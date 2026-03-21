"""Session data ingestion for BESPOKE.

Uses ai-data-extraction (github.com/0xSero/ai-data-extraction) for the hard
work of parsing provider-specific session file formats. BESPOKE adds:
- Embedding computation at ingest time
- SQLite + sqlite-vec writes
- Feedback signal extraction from user follow-ups
- Session linking across multi-turn conversations
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Generator
from datetime import datetime


@dataclass
class Interaction:
    """A single user-assistant interaction in BESPOKE's internal format."""
    provider: str              # 'claude', 'openai', etc.
    model: str                 # 'claude-opus-4-6', etc.
    source: str                # 'claude-code', 'cursor', etc.
    session_id: str            # Links multi-turn conversations
    system_prompt: Optional[str]
    user_message: str
    assistant_response: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    timestamp: str             # ISO format
    user_followup: Optional[str] = None


def ingest_extraction_jsonl(jsonl_path: Path) -> List[Interaction]:
    """Convert ai-data-extraction JSONL output to BESPOKE Interactions.

    The JSONL format from ai-data-extraction includes fields like:
    - user_message / human_message
    - assistant_response / ai_response
    - model, provider, session_id, timestamp
    - token counts (if available)

    Field names may vary by version — this function normalizes them.
    """
    interactions = []

    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  Warning: skipping malformed JSON on line {line_num}")
                continue

            # Normalize field names (ai-data-extraction format may vary)
            user_msg = (
                record.get("user_message") or
                record.get("human_message") or
                record.get("prompt") or
                ""
            )
            assistant_resp = (
                record.get("assistant_response") or
                record.get("ai_response") or
                record.get("response") or
                ""
            )

            if not user_msg or not assistant_resp:
                continue

            interactions.append(Interaction(
                provider=record.get("provider", "unknown"),
                model=record.get("model", "unknown"),
                source=record.get("source", record.get("tool", "unknown")),
                session_id=record.get("session_id", record.get("conversation_id", "")),
                system_prompt=record.get("system_prompt"),
                user_message=user_msg,
                assistant_response=assistant_resp,
                input_tokens=record.get("input_tokens"),
                output_tokens=record.get("output_tokens"),
                timestamp=record.get("timestamp", datetime.now().isoformat()),
                user_followup=record.get("user_followup"),
            ))

    return interactions


def _is_real_user_turn(msg: dict) -> bool:
    """Check if a message is a real user turn (not a tool_result)."""
    msg_type = msg.get("type")
    if msg_type not in ("user", "human"):
        return False

    inner = msg.get("message", msg)
    content = inner.get("content", "")

    if isinstance(content, str):
        return bool(content.strip())

    if isinstance(content, list):
        for block in content:
            if isinstance(block, str) and block.strip():
                return True
            if isinstance(block, dict) and block.get("type") == "text":
                return True
        return False

    return False


def _extract_all_content(msg: dict) -> str:
    """Extract all content from a message, preserving thinking, tool_use, and tool_result blocks.

    Stage 1 keeps everything — full reasoning traces, full tool calls and results.
    Stage 2a handles curation.
    """
    inner = msg.get("message", msg)
    content = inner.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "thinking":
                    thinking = block.get("thinking", "")
                    if thinking:
                        parts.append(f"<thinking>\n{thinking}\n</thinking>")
                elif block_type == "tool_use":
                    tool_name = block.get("name", "unknown")
                    tool_input = json.dumps(block.get("input", {}))
                    parts.append(f"<tool_use name=\"{tool_name}\">\n{tool_input}\n</tool_use>")
                elif block_type == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_parts = []
                        for rb in result_content:
                            if isinstance(rb, str):
                                result_parts.append(rb)
                            elif isinstance(rb, dict) and rb.get("type") == "text":
                                result_parts.append(rb.get("text", ""))
                        result_content = "\n".join(result_parts)
                    parts.append(f"<tool_result>\n{result_content}\n</tool_result>")
        return "\n".join(p for p in parts if p)

    return str(content) if content else ""


def ingest_claude_code_direct(session_path: Path) -> List[Interaction]:
    """Parse a Claude Code JSONL session file directly.

    Single-pass streaming parser — reads one JSON object at a time and
    discards it after extracting content. Avoids loading the full message
    history into memory.

    Between two real user turns, everything (assistant thinking, text,
    tool calls, tool results) is concatenated into a single response.
    """
    session_id = session_path.stem
    interactions = []

    # Streaming state for current turn
    current_user_text = None
    current_user_timestamp = None
    response_parts = []
    resp_model = "unknown"
    resp_input_tokens = None
    resp_output_tokens = None

    with open(session_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if _is_real_user_turn(msg):
                # Finalize previous turn
                if current_user_text is not None:
                    assistant_text = "\n".join(response_parts)
                    if assistant_text:
                        interactions.append(Interaction(
                            provider="claude",
                            model=resp_model,
                            source="claude-code",
                            session_id=session_id,
                            system_prompt=None,
                            user_message=current_user_text,
                            assistant_response=assistant_text,
                            input_tokens=resp_input_tokens,
                            output_tokens=resp_output_tokens,
                            timestamp=current_user_timestamp,
                        ))

                # Start new turn
                current_user_text = _extract_text(msg)
                current_user_timestamp = msg.get("timestamp", datetime.now().isoformat())
                response_parts = []
                resp_model = "unknown"
                resp_input_tokens = None
                resp_output_tokens = None

            elif current_user_text is not None:
                msg_type = msg.get("type", "")
                if msg_type == "assistant":
                    content = _extract_all_content(msg)
                    if content:
                        response_parts.append(content)
                    inner = msg.get("message", msg)
                    if inner.get("model"):
                        resp_model = inner["model"]
                    usage = inner.get("usage", {})
                    if usage.get("input_tokens"):
                        resp_input_tokens = (resp_input_tokens or 0) + usage["input_tokens"]
                    if usage.get("output_tokens"):
                        resp_output_tokens = (resp_output_tokens or 0) + usage["output_tokens"]
                elif msg_type in ("user", "human"):
                    # tool_result messages (not real user turns) — include in response
                    content = _extract_all_content(msg)
                    if content:
                        response_parts.append(content)

    # Finalize last turn
    if current_user_text is not None:
        assistant_text = "\n".join(response_parts)
        if assistant_text:
            interactions.append(Interaction(
                provider="claude",
                model=resp_model,
                source="claude-code",
                session_id=session_id,
                system_prompt=None,
                user_message=current_user_text,
                assistant_response=assistant_text,
                input_tokens=resp_input_tokens,
                output_tokens=resp_output_tokens,
                timestamp=current_user_timestamp,
            ))

    # Add followups (next user turn's text = feedback signal)
    for i in range(len(interactions) - 1):
        interactions[i].user_followup = interactions[i + 1].user_message

    return interactions


def _extract_text(msg: dict) -> str:
    """Extract only text content from a message (for user messages and followups)."""
    inner = msg.get("message", msg)
    content = inner.get("content", inner.get("text", ""))

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts)

    return str(content) if content else ""


def find_claude_code_sessions(
    base_dir: Optional[Path] = None,
    since: Optional[datetime] = None
) -> Generator[Path, None, None]:
    """Find Claude Code session files, optionally filtered by modification time."""
    if base_dir is None:
        base_dir = Path.home() / ".claude" / "projects"

    if not base_dir.exists():
        return

    for project_dir in base_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for session_file in project_dir.glob("*.jsonl"):
            if since is not None:
                mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                if mtime < since:
                    continue
            yield session_file
