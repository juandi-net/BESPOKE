"""Tests for stripping injected Conductor <system_instruction> boilerplate, keeping the real ask."""
from bespoke.extract.content_type import strip_conductor_boilerplate


def test_strips_underscore_block_keeps_real_ask():
    t = ("<system_instruction>\nYou are working inside Conductor, a Mac app… target branch main.\n"
         "</system_instruction>\nTwo fixes: center the login page and remove the dev indicator.")
    out = strip_conductor_boilerplate(t)
    assert "Conductor" not in out
    assert "system_instruction" not in out
    assert out == "Two fixes: center the login page and remove the dev indicator."


def test_strips_multiple_blocks_underscore_and_hyphen():
    t = ("<system_instruction>boilerplate one</system_instruction>\n"
         "<system-instruction>branch rename rules…</system-instruction>\n"
         "weekly report did not run. why? fix it")
    out = strip_conductor_boilerplate(t)
    assert out == "weekly report did not run. why? fix it"


def test_pure_boilerplate_becomes_empty():
    t = "<system_instruction>You are working inside Conductor…</system_instruction>"
    assert strip_conductor_boilerplate(t).strip() == ""


def test_no_boilerplate_unchanged():
    t = "Wouldn't it make sense to only show 10-K?"
    assert strip_conductor_boilerplate(t) == t


def test_case_insensitive_and_multiline():
    t = "<SYSTEM_INSTRUCTION>\nline1\nline2\n</SYSTEM_INSTRUCTION>\nreal ask here"
    assert strip_conductor_boilerplate(t) == "real ask here"


def test_handles_none_and_empty():
    assert strip_conductor_boilerplate(None) is None
    assert strip_conductor_boilerplate("") == ""
