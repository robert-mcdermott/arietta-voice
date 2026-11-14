import pytest

from arietta_voice.clauses import CLAUSE_CHARS, SpeechClauses


def collect(parts):
    clauses = SpeechClauses()
    result = [clause for part in parts for clause in clauses.feed(part)]
    tail = clauses.finish()
    return result + ([tail] if tail else [])


def test_boundaries_survive_single_character_deltas_and_preserve_decimals():
    text = "Dr. Jones measured 3.14 units. **Next** result! Final tail"
    assert collect(list(text)) == ["Dr. Jones measured 3.14 units.", "Next result!", "Final tail"]
    assert collect([text]) == collect(list(text))


def test_long_unpunctuated_reply_is_bounded_and_not_lost():
    text = "word " * 180
    result = collect(list(text))
    assert max(map(len, result)) <= CLAUSE_CHARS
    assert " ".join(result) == text.strip()
    assert max(map(len, collect(["x" * 800]))) <= CLAUSE_CHARS


def test_oversized_reply_rejected_before_further_speech():
    clauses = SpeechClauses()
    with pytest.raises(ValueError, match="limit"):
        list(clauses.feed("x" * 12001))


def test_first_clause_may_end_at_a_comma_once_long_enough():
    text = "The capital of France is Paris, which sits on the Seine, a long river. It is old."
    assert collect(list(text)) == [
        "The capital of France is Paris,",
        "which sits on the Seine, a long river.",
        "It is old.",
    ]
    assert collect([text]) == collect(list(text))


def test_short_openers_and_numbers_do_not_split_the_first_clause():
    assert collect(list("Well, yes, it costs 1,250 dollars. Next.")) == [
        "Well, yes, it costs 1,250 dollars.",
        "Next.",
    ]
    assert collect(list("Sure \u2014 the answer is four.")) == ["Sure \u2014 the answer is four."]


def test_first_clause_may_end_at_a_dash_and_later_clauses_ignore_commas():
    text = "Photosynthesis is how plants eat \u2014 they turn light into sugar. Then, they grow."
    assert collect(list(text)) == [
        "Photosynthesis is how plants eat \u2014",
        "they turn light into sugar.",
        "Then, they grow.",
    ]
