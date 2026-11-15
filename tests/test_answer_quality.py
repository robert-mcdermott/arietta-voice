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


def test_summary_splits_answers_with_and_without_notes():
    rows = [
        {"refused": False, "correct": True, "first_s": 2.0, "notes": True},
        {"refused": False, "correct": True, "first_s": 1.0, "notes": True},
        {"refused": False, "correct": None, "first_s": 0.5, "notes": False},
    ]
    summary = summarize(rows)
    assert summary["median_first_with_notes_s"] == 1.5
    assert summary["median_first_without_notes_s"] == 0.5


def test_the_made_up_notes_and_their_questions_are_well_formed():
    import re

    from bench.answer_quality import ROOT

    questions = json.loads((ROOT / "evals" / "notes_questions.json").read_text())
    assert len(questions) >= 12 and all(re.compile(item["a"]) for item in questions)
    notes = sorted((ROOT / "evals" / "fixture_notes").glob("*.md"))
    assert notes and all("Made-up notes" in path.read_text() for path in notes)
