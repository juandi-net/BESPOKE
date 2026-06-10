"""Mechanical taste-axis extractors (RT-006): turn juandi's curation whys into cheap,
deterministic signals. Each test is grounded in a real `drop` why from curation.json:

  "dont like the emojis"                         -> emoji_count
  "great question!" / "dont need the absolutely right" -> sycophancy_score
  "too excited"                                  -> exclamation_count
  "tries to deflect back to me"                  -> hedge_score
  "should have lead with the answer"             -> leading_pleasantry
  "added words that didnt matter" / "too verbose"-> filler_count
  "should have just done it"                     -> has_code

No LLM, pure functions — the point is a scalable judge that needs no cloud call.
"""
from bespoke.eval import taste_axes as t


# ---- emoji_count ----
def test_emoji_count_counts_emoji():
    assert t.emoji_count("Done ✅ shipped 🚀") == 2

def test_emoji_count_zero_on_plain_text():
    assert t.emoji_count("just the facts, no decoration") == 0


# ---- sycophancy_score ----
def test_sycophancy_flags_great_question():
    assert t.sycophancy_score("Great question! Here's the answer.") >= 1

def test_sycophancy_flags_absolutely_right():
    assert t.sycophancy_score("You're absolutely right, my mistake.") >= 1

def test_sycophancy_clean_answer_scores_zero():
    assert t.sycophancy_score("The bug is on line 12; fix the off-by-one.") == 0


# ---- exclamation_count ----
def test_exclamation_count():
    assert t.exclamation_count("Wow! Amazing! Let's go!") == 3

def test_exclamation_count_zero():
    assert t.exclamation_count("Set the flag to false.") == 0


# ---- hedge_score (deflection / non-answers) ----
def test_hedge_flags_deflection_back_to_user():
    # "not a clear answer and tries to deflect back to me"
    assert t.hedge_score("It depends on your needs. What would you like to do?") >= 1

def test_hedge_direct_answer_scores_zero():
    assert t.hedge_score("Use Postgres. It handles your concurrency needs.") == 0


# ---- leading_pleasantry (should have led with the answer) ----
def test_leading_pleasantry_when_starts_with_fluff():
    assert t.leading_pleasantry("Great question! The answer is 42.") == 1

def test_leading_pleasantry_zero_when_answer_first():
    assert t.leading_pleasantry("The answer is 42.") == 0


# ---- filler_count (verbose, words that didn't matter) ----
def test_filler_count_counts_weasel_words():
    # "added words that didnt matter to me" / "too verbose"
    assert t.filler_count("I think you could basically just simply try it") >= 3

def test_filler_count_zero_on_tight_prose():
    assert t.filler_count("Run the migration, then restart.") == 0


# ---- has_code (should have just done it) ----
def test_has_code_detects_fenced_block():
    assert t.has_code("Here:\n```py\nx = 1\n```") == 1

def test_has_code_zero_without_code():
    assert t.has_code("You should write a function that does it.") == 0


# ---- taste_features assembles the vector ----
def test_taste_features_returns_all_axes():
    f = t.taste_features("Great question! 🚀 You could maybe try it.")
    for axis in ("emoji_count", "sycophancy_score", "exclamation_count",
                 "hedge_score", "leading_pleasantry", "filler_count",
                 "has_code", "length"):
        assert axis in f
    assert f["emoji_count"] == 1
    assert f["sycophancy_score"] >= 1

def test_taste_features_order_is_stable():
    # FEATURE_NAMES drives the matrix columns — must be deterministic.
    assert t.feature_vector("hi") == [t.taste_features("hi")[k] for k in t.FEATURE_NAMES]


# ---- taste_score: mechanical axes fused into one [0,1] quality (1.0 = no negative tells) ----
def test_taste_score_clean_answer_is_one():
    assert t.taste_score("Use Postgres. It handles your concurrency needs.") == 1.0

def test_taste_score_in_unit_interval():
    assert 0.0 <= t.taste_score("Great question! 🚀 You're absolutely right!!!") <= 1.0

def test_taste_score_penalizes_negative_tells():
    clean = t.taste_score("Set the flag to false and redeploy.")
    fluffy = t.taste_score("Great question! 🚀 You're absolutely right!")
    assert fluffy < clean

def test_taste_score_monotonic_in_severity():
    clean = t.taste_score("The answer is 42.")
    one_tell = t.taste_score("The answer is 42!")            # one exclamation
    many_tells = t.taste_score("Great question! 🚀 Amazing!!!")  # emoji + sycophancy + bangs
    assert clean > one_tell > many_tells
