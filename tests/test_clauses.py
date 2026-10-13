import pytest
from arietta_voice.clauses import SpeechClauses, CLAUSE_CHARS


def collect(parts):
    clauses = SpeechClauses()
    result = [clause for part in parts for clause in clauses.feed(part)]
    tail = clauses.finish()
    return result + ([tail] if tail else [])


def test_boundaries_survive_single_character_deltas_and_preserve_decimals():
    text = 'Dr. Jones measured 3.14 units. **Next** result! Final tail'
    assert collect(list(text)) == ['Dr. Jones measured 3.14 units.', 'Next result!', 'Final tail']
    assert collect([text]) == collect(list(text))


def test_long_unpunctuated_reply_is_bounded_and_not_lost():
    text = 'word ' * 180
    result = collect(list(text))
    assert max(map(len, result)) <= CLAUSE_CHARS
    assert ' '.join(result) == text.strip()
    assert max(map(len, collect(['x' * 800]))) <= CLAUSE_CHARS


def test_oversized_reply_rejected_before_further_speech():
    clauses = SpeechClauses()
    with pytest.raises(ValueError, match='limit'):
        list(clauses.feed('x' * 12001))
