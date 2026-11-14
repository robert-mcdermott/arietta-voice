import json

from bench.answer_quality import QUESTIONS, is_correct, is_refusal, summarize


def test_refusals_are_recognised():
    assert is_refusal("I don't have that information right now.")
    assert is_refusal("I’m not sure, sorry.")
    assert not is_refusal("Spiders have eight legs.")


def test_facts_are_checked_against_a_pattern():
    assert is_correct("It's Canberra.", "canberra") is True
    assert is_correct("It's Sydney.", "canberra") is False
    assert is_correct("Anything.", None) is None


def test_summary_counts():
    rows = [
        {"refused": True, "correct": None, "first_s": 1.0},
        {"refused": False, "correct": True, "first_s": 0.5},
        {"refused": False, "correct": False, "first_s": 2.0},
    ]
    assert summarize(rows) == {
        "answers": 3,
        "refused": 1,
        "facts": 2,
        "facts_right": 1,
        "median_first_s": 1.0,
        "smarter_answers": 0,
    }


def test_the_question_set_is_well_formed():
    import re

    questions = json.loads(QUESTIONS.read_text())
    assert len(questions) >= 40
    for item in questions:
        assert item["q"].strip()
        if "a" in item:
            re.compile(item["a"])
