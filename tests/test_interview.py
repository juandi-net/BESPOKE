import json

from bespoke.benchmark.interview import (
    INTERVIEW_SYSTEM,
    OUTPUT_SCHEMA,
    SCHEMA_TEMPLATE,
    build_interview_system_prompt,
)


def test_interview_system_prompt_has_placeholders():
    """New system prompt contains the three format placeholders."""
    assert "{prescan_summary}" in INTERVIEW_SYSTEM
    assert "{schema_template}" in INTERVIEW_SYSTEM
    assert "{output_schema}" in INTERVIEW_SYSTEM


def test_build_interview_system_prompt():
    """build_interview_system_prompt formats prescan data into system prompt."""
    prescan = {
        "total_interactions": 100,
        "total_sessions": 20,
        "avg_user_message_length": 250,
        "sample_prompts": [{"message": "Write a parser", "length": 50, "has_followup": True}],
    }

    result = build_interview_system_prompt(prescan)

    assert "100" in result  # total_interactions injected
    assert "operating_altitude" in result  # output schema present
    assert "evaluation_sequence" in result  # schema template present
    # No unformatted placeholders
    assert "{prescan_summary}" not in result
    assert "{schema_template}" not in result
    assert "{output_schema}" not in result


def test_output_schema_is_valid_json():
    """OUTPUT_SCHEMA constant is parseable as JSON."""
    parsed = json.loads(OUTPUT_SCHEMA)
    assert "operating_altitude" in parsed
    assert "domains" in parsed
    assert "universal_gates" in parsed
