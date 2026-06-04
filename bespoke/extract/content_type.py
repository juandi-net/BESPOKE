"""Deterministic content-type classifier + tool-block cleaner (no LLM).

Most of the captured corpus is NOT the user's reasoning: ~2/3 is claude-mem observer traffic
(`<observation>` XML) and tool-result dumps. This tags each interaction so the warehouse stays
whole (raw + tag) while extract can drop the junk and KEEP the signal.

Categories:
  observer          claude-mem observer turns (a different agent talking to itself) — drop at extract
  tool_result_only  tool output / local-command stdout with no real assistant content — drop at extract
  agentic           real reasoning interleaved with tool calls — KEEP, but strip the raw tool blocks
                    down to a `[N tool calls]` summary (see clean_tool_blocks)
  clean             ordinary prose reasoning — keep as-is

Markers validated against the live warehouse (2026-06-03; see docs/bespoke-research-log.md RT-002).
"""
import re

# A tool turn: <tool_use name="X">...</tool_use> or <tool_result>...</tool_result> (incl. empty).
_TOOL_BLOCK = re.compile(
    r"<tool_use\b[^>]*>.*?</tool_use>|<tool_result\b[^>]*>.*?</tool_result>"
    r"|<tool_use\b[^>]*/>|<tool_result\b[^>]*/>",
    re.DOTALL,
)
_TOOL_USE_OPEN = re.compile(r"<tool_use\b")


def classify_content(user_message, assistant_response):
    """Return (content_type, tool_call_count) for one interaction. Pure, deterministic."""
    um = user_message or ""
    ar = assistant_response or ""
    um_low = um.lower()

    # observer (claude-mem) — checked first; an observer turn may also contain tool-ish markers.
    if ("<observation>" in ar
            or "<observed_from_primary_session>" in um
            or "you are a claude-mem" in um_low
            or "memory agent" in um_low):
        return ("observer", 0)

    # tool-result-only turns — no genuine assistant content.
    if (ar.strip() == "No response requested."
            or "<local-command-stdout>" in um
            or "<local-command-stderr>" in um):
        return ("tool_result_only", 0)

    # agentic — real reasoning interleaved with tool calls.
    n = len(_TOOL_USE_OPEN.findall(ar))
    if n > 0:
        return ("agentic", n)

    return ("clean", 0)


def clean_tool_blocks(text):
    """Strip raw <tool_use>/<tool_result> blocks, collapse them to one `[N tool call(s)]` marker.

    Keeps the surrounding reasoning (<thinking>, prose). The marker lands where the first tool
    block was; remaining blocks are removed (N is the total). No-op when there are no tool calls.
    """
    if not text:
        return text
    n = len(_TOOL_USE_OPEN.findall(text))
    if n == 0:
        return text
    marker = f"[{n} tool call{'s' if n != 1 else ''}]"
    sentinel = "\x00"
    cleaned = _TOOL_BLOCK.sub(sentinel, text)
    # First run of sentinels (and the whitespace around it) becomes the single marker.
    cleaned = re.sub(r"\s*" + sentinel + r"(\s*" + sentinel + r")*\s*", f" {marker} ", cleaned, count=1)
    cleaned = cleaned.replace(sentinel, "")  # drop any later sentinels
    return cleaned.strip()
