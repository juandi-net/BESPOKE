"""Data-informed benchmark interview — discovers how the user actually evaluates AI output."""

import json
from bespoke.teach.llm_client import chat, chat_json
from bespoke.config import config


INTERVIEW_SYSTEM = """You are conducting a structured assessment to build a judgment model for how
this user evaluates AI output. You are not asking what they prefer -- you are
discovering how they actually make accept/reject decisions.

You have access to a statistical summary of their interaction history:

{prescan_summary}

Your internal tracking structure (never shown to the user):

{schema_template}

PHASE 0 — Operating Altitude (1-2 turns)
Determine how this user relates to AI output. Do they:
- Evaluate at the artifact level (reads code, checks implementation details)
- Evaluate at the intent level (checks if output matches what they asked for)
- Evaluate at the ideation level (uses AI for brainstorming, executes independently)

Use the prescan data to form a hypothesis first. Their average prompt length,
the ratio of short directives to long-form descriptions, and session patterns
all indicate altitude. Confirm with one targeted question.

PHASE 1 — Approach Diagnosis (3-4 turns)
Present scenarios from the user's actual domains (use sample_prompts from the
prescan to identify their domain language). Don't ask "what do you value" --
describe a situation and ask "what do you do?"

Examples calibrated to altitude:
- Intent-level user: "You described a feature. The AI built it, but it solves
  a slightly different problem. Works perfectly for that other problem though.
  What do you do?"
- Artifact-level user: "The function works but the error handling only covers
  the happy path. Ship or send back?"

Map each response to the schema: what did they check first (evaluation_sequence),
what was non-negotiable (gates), what did they tolerate (threshold).

PHASE 2 — Flaw Detection (2-3 turns)
Describe output with embedded flaws of different types and severity. Ask
"what's wrong with this?" What they catch = active standard (gate or quality
check). What they miss = not on their radar (exclude from benchmark). What they
notice and shrug off = pragmatic threshold.

Generate flaws relevant to their altitude:
- Intent-level: misunderstood goal, overengineered, missing edge case in design
- Artifact-level: naming issues, missing tests, inefficient algorithm

PHASE 3 — Data Confrontation (2-3 turns)
Use the prescan behavioral signals. If the data shows high accept rate on
short responses, ask about it. If the data shows they rarely push back, probe
whether that's satisfaction or low expectations.

Frame as genuine curiosity, not challenge:
"Your history shows you accept most AI output without pushback. Is that because
the output is generally good enough, or because you fix things yourself after?"

PHASE 4 — Bar Articulation (1-2 turns)
Synthesize a judgment profile and play it back:
"Here's how I think you evaluate work: you check X first, you have a hard line
on Y, you tolerate Z up to this threshold, and your standards differ between
domain A and domain B. Does this sound right?"

User validates or corrects. Then output the final JSON.

OUTPUT FORMAT:
After Phase 4, output the judgment model as a JSON block:

{output_schema}

Rules:
- Keep turns SHORT. One question at a time. No lists of questions.
- Never explain the methodology. Never say "I'm trying to discover your..."
- Use the user's own domain language (from prescan samples).
- Track populated vs empty schema fields internally. Steer toward gaps.
- Every question must map to a specific schema field. No decorative questions."""


SCHEMA_TEMPLATE = """{
    "operating_altitude": "intent | artifact | ideation",
    "domains": [
        {
            "id": "domain_id",
            "evaluation_sequence": ["first check", "then this"],
            "gates": [{"check": "pass/fail question", "severity": "hard_fail"}],
            "quality_checks": [{"check": "quality question"}],
            "tolerance_threshold": "where the hard line softens"
        }
    ],
    "universal_gates": [{"check": "applies regardless of domain"}],
    "pragmatic_tradeoffs": ["what they sacrifice for what"]
}"""


OUTPUT_SCHEMA = json.dumps({
    "operating_altitude": "intent | artifact | ideation",
    "altitude_evidence": "one sentence explaining why",
    "domains": [
        {
            "id": "domain_snake_case",
            "name": "Human readable",
            "evaluation_sequence": ["what gets checked first", "then this", "then this"],
            "gates": [
                {
                    "id": "gate_id",
                    "check": "Binary pass/fail question",
                    "severity": "hard_fail",
                }
            ],
            "quality_checks": [
                {
                    "id": "check_id",
                    "check": "Binary yes/no quality question",
                    "user_evidence": "what the user said or did that revealed this",
                }
            ],
            "aspiration_targets": [
                {
                    "id": "target_id",
                    "description": "Directional goal, not enforced",
                    "user_evidence": "what revealed this",
                }
            ],
            "tolerance_threshold": "description of where hard line softens",
        }
    ],
    "universal_gates": [
        {
            "id": "gate_id",
            "check": "Binary pass/fail that applies regardless of domain",
        }
    ],
    "pragmatic_tradeoffs": [
        "what they'll sacrifice for what -- e.g. 'will accept messy naming under deadline pressure'"
    ],
    "data_confrontation_notes": [
        "divergences between stated and revealed behavior, with user's response"
    ],
}, indent=2)


def build_interview_system_prompt(prescan: dict) -> str:
    """Format the interview system prompt with prescan data."""
    return INTERVIEW_SYSTEM.format(
        prescan_summary=json.dumps(prescan, indent=2),
        schema_template=SCHEMA_TEMPLATE,
        output_schema=OUTPUT_SCHEMA,
    )


def _extract_interview_summary(messages: list, final_response: str) -> str:
    """Extract the interview summary for the separate JSON generation call.

    Prefers the model's Phase 4 plaintext playback (text before the JSON block).
    Falls back to formatting the full conversation if the playback is too short.
    """
    if "```json" in final_response:
        summary = final_response[:final_response.index("```json")].strip()
        if len(summary) > 50:
            return summary

    turns = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        role = "Interviewer" if msg["role"] == "assistant" else "User"
        turns.append(f"{role}: {msg['content']}")
    return "\n\n".join(turns)


def _generate_benchmark_json(summary_text: str) -> dict:
    """Generate benchmark JSON in a separate API call with full token budget.

    Avoids truncation from sharing a token-limited call with conversation history.
    One retry on parse failure; if that fails, surfaces the error instead of looping.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Convert this interview summary into the benchmark schema. "
                "Output only valid JSON, nothing else.\n\n"
                f"Schema:\n{OUTPUT_SCHEMA}"
            ),
        },
        {
            "role": "user",
            "content": summary_text,
        },
    ]

    try:
        return chat_json(
            messages=messages,
            model=config.llm.model_benchmark,
            temperature=0.2,
            max_tokens=128_000,
        )
    except (ValueError, json.JSONDecodeError):
        print("\n(Failed to parse benchmark JSON, retrying once...)\n")
        try:
            return chat_json(
                messages=messages,
                model=config.llm.model_benchmark,
                temperature=0.2,
                max_tokens=128_000,
            )
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Failed to generate benchmark JSON after 2 attempts: {e}"
            )


def run_interview(prescan: dict = None) -> dict:
    """Run the data-informed benchmark interview in the terminal.

    If prescan is None, generates it from the DB.

    Returns the extracted judgment model as a dict.
    """
    if prescan is None:
        from bespoke.benchmark.prescan import generate_prescan_summary
        prescan = generate_prescan_summary()

    print("\nBESPOKE Benchmark Interview")
    print("=" * 50)
    print("This interview will take about 10-15 minutes.")
    print("I'll discover how you actually evaluate AI output.")
    print("Type your answers naturally. Type 'done' when you want to finish early.\n")

    system = build_interview_system_prompt(prescan)
    messages = [{"role": "system", "content": system}]

    # Kick off the interview
    messages.append({"role": "user", "content": "Please begin the interview."})

    while True:
        response = chat(
            messages=messages,
            model=config.llm.model_benchmark,
            temperature=0.7,
            max_tokens=1000,
        )

        print(f"\nBESPOKE: {response}\n")
        messages.append({"role": "assistant", "content": response})

        # Check if the LLM has produced the final JSON output.
        # Split JSON generation into a separate API call to avoid truncation —
        # the conversation turn's 1000-token limit is too small for the schema.
        if "```json" in response and '"operating_altitude"' in response:
            summary = _extract_interview_summary(messages, response)
            print("(Generating benchmark JSON...)\n")
            return _generate_benchmark_json(summary)

        # Get user input
        user_input = input("You: ").strip()
        if user_input.lower() == "done":
            messages.append({"role": "user", "content": "Please finalize the judgment model based on what we've discussed so far. Output the JSON."})
        else:
            messages.append({"role": "user", "content": user_input})

    return {}


if __name__ == "__main__":
    result = run_interview()
    print(f"\nExtracted judgment model")
    print(json.dumps(result, indent=2))
