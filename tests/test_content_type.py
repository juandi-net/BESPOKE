"""Tests for the deterministic content-type classifier + tool-block cleaner."""
from bespoke.extract.content_type import classify_content, clean_tool_blocks


# --- classify_content: content_type ---

def test_observer_when_observation_xml_in_response():
    ct, _ = classify_content("anything", "<thinking>x</thinking><observation>note</observation>")
    assert ct == "observer"


def test_observer_when_claude_mem_prompt_in_user_message():
    ct, _ = classify_content("You are a Claude-Mem, a specialized observer tool", "<thinking>...</thinking>")
    assert ct == "observer"


def test_observer_when_observed_from_primary_session():
    ct, _ = classify_content("<observed_from_primary_session><what_happened>Glob</what_happened>", "ok")
    assert ct == "observer"


def test_tool_result_only_when_no_response_requested():
    ct, _ = classify_content("<local-command-stdout>## Context Usage</local-command-stdout>", "No response requested.")
    assert ct == "tool_result_only"


def test_tool_result_only_when_local_command_stdout_user_message():
    ct, _ = classify_content("<local-command-stdout>foo</local-command-stdout>", "")
    assert ct == "tool_result_only"


def test_agentic_when_tool_use_in_response():
    ar = ('<thinking>plan</thinking>'
          '<tool_use name="Bash">{"command":"ls"}</tool_use><tool_result>f1</tool_result>'
          '<tool_use name="Glob">{}</tool_use><tool_result>f2</tool_result>')
    ct, n = classify_content("do the thing", ar)
    assert ct == "agentic"
    assert n == 2


def test_clean_prose_is_clean_with_zero_tool_calls():
    ct, n = classify_content("what is 2+2?", "It's 4 — here's the reasoning.")
    assert ct == "clean"
    assert n == 0


def test_observer_takes_precedence_over_tool_use():
    ct, _ = classify_content("hi", '<observation>x</observation><tool_use name="Bash">{}</tool_use>')
    assert ct == "observer"


def test_handles_none_inputs():
    ct, n = classify_content(None, None)
    assert ct == "clean"
    assert n == 0


# --- clean_tool_blocks: strip dumps, keep reasoning, summarize as [N tool calls] ---

def test_clean_tool_blocks_replaces_single_with_count_and_keeps_prose():
    ar = ('<thinking>plan</thinking>\n'
          '<tool_use name="Bash">{"command":"ls"}</tool_use>\n<tool_result>file1</tool_result>\n'
          'Done.')
    cleaned = clean_tool_blocks(ar)
    assert "<tool_use" not in cleaned and "<tool_result" not in cleaned
    assert "[1 tool call]" in cleaned
    assert "<thinking>plan</thinking>" in cleaned
    assert "Done." in cleaned


def test_clean_tool_blocks_pluralizes_count():
    ar = ('<tool_use name="A">{}</tool_use><tool_result>x</tool_result>'
          '<tool_use name="B">{}</tool_use><tool_result>y</tool_result>')
    cleaned = clean_tool_blocks(ar)
    assert "[2 tool calls]" in cleaned


def test_clean_tool_blocks_noop_when_no_tools():
    assert clean_tool_blocks("just prose, no tools") == "just prose, no tools"


def test_clean_tool_blocks_handles_empty_and_none():
    assert clean_tool_blocks("") == ""
    assert clean_tool_blocks(None) is None
