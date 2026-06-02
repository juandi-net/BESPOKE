"""Deterministic feedback classification from the user's next message (user_followup).

Rule-based accept/reject cue detection. This is the SEED signal — the geometric quality
probe trains on these labels. Crude by design; tunable on real data (Phase 5). No LLM.
"""
import re

# Short cue words checked on word boundaries (avoid matching 'now'/'note'/'notion').
ACCEPT_WORDS = {"perfect", "great", "thanks", "awesome", "exactly", "nice", "yes", "yep",
                "yeah", "correct", "good", "lgtm", "brilliant", "amazing", "love"}
REJECT_WORDS = {"no", "not", "wrong", "incorrect", "broke", "broken", "fix", "instead",
                "fails", "failed", "error", "undo", "revert", "stop", "nope"}

# Multi-word phrases checked as substrings.
ACCEPT_PHRASES = ["do it", "go ahead", "ship it", "looks good", "that works",
                  "makes sense", "love it", "thank you", "exactly what"]
REJECT_PHRASES = ["doesn't", "does not", "didn't", "that's not", "not what", "not right"]
STRONG_ACCEPT = ["perfect", "love it", "ship it", "exactly what", "brilliant", "amazing"]
STRONG_REJECT = ["completely wrong", "totally wrong", "terrible", "awful", "way off"]


def _has_word(words, vocab):
    return bool(words & vocab)


def _has_phrase(text, phrases):
    return any(p in text for p in phrases)


def classify_feedback(user_followup):
    """Return one of: strong_accept, accept, neutral, reject, strong_reject."""
    if not user_followup or not user_followup.strip():
        return "neutral"
    t = user_followup.strip().lower()
    words = set(re.findall(r"[a-z']+", t))

    acc = _has_word(words, ACCEPT_WORDS) or _has_phrase(t, ACCEPT_PHRASES)
    rej = _has_word(words, REJECT_WORDS) or _has_phrase(t, REJECT_PHRASES)

    # Strong signals (only when unambiguous).
    if _has_phrase(t, STRONG_REJECT) and not acc:
        return "strong_reject"
    if _has_phrase(t, STRONG_ACCEPT) and not rej:
        return "strong_accept"

    if acc and not rej:
        return "accept"
    if rej and not acc:
        return "reject"
    return "neutral"
