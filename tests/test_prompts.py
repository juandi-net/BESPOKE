from bespoke.teach.prompts import make_session_extraction_prompt


def test_basic_session_prompt():
    """A 2-turn session produces numbered turns in <session> tags."""
    turns = [
        {"user_message": "Fix the bug in auth.py", "assistant_response": "I found the issue..."},
        {"user_message": "Great, deploy it", "assistant_response": "Deployed to staging..."},
    ]
    messages = make_session_extraction_prompt(turns)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "--- Turn 1 ---" in messages[1]["content"]
    assert "--- Turn 2 ---" in messages[1]["content"]
    assert "<session>" in messages[1]["content"]
    assert "</session>" in messages[1]["content"]
    assert "Fix the bug in auth.py" in messages[1]["content"]


def test_context_only_turns():
    """Context-only turns get the marker in their header."""
    turns = [
        {"user_message": "First message", "assistant_response": "First response"},
        {"user_message": "Second message", "assistant_response": "Second response"},
        {"user_message": "Third message", "assistant_response": "Third response"},
    ]
    messages = make_session_extraction_prompt(turns, context_only_turns={1, 2})
    content = messages[1]["content"]
    assert "[CONTEXT ONLY" in content
    assert "Turn 1 [CONTEXT ONLY" in content
    assert "Turn 2 [CONTEXT ONLY" in content
    assert "--- Turn 3 ---" in content  # no marker
    # Turn 3 should NOT have CONTEXT ONLY
    assert "Turn 3 [CONTEXT ONLY" not in content


def test_system_prompt_has_key_instructions():
    """System prompt contains critical directives."""
    turns = [{"user_message": "test", "assistant_response": "test"}]
    messages = make_session_extraction_prompt(turns)
    system = messages[0]["content"]
    assert "turn_number" in system
    assert "SELF-CONTAINED" in system
    assert "JSON array" in system
    assert "CONTEXT ONLY" in system
    assert "INPUT DATA" in system  # prompt injection defense
