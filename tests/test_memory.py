from pathlib import Path
from unittest.mock import Mock

import pytest

from arietta_voice.memory import MemoryStore


def test_interrupted_memory_replacement_retains_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "MEMORY.md"
    original = "# Memory\n- Existing fact\n"
    path.write_text(original)
    memory = MemoryStore(path, True, 1)
    generator = Mock(generate=Mock(return_value="# Memory\n- Consolidated fact"))
    monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("simulated write failure")))
    with pytest.raises(OSError):
        memory.maybe_consolidate(generator, 1)
    assert path.read_text() == original
    assert not path.with_name(".MEMORY.md.working").exists()


def test_memory_updates_are_private_and_atomic(tmp_path):
    path = tmp_path / "MEMORY.md"
    memory = MemoryStore(path, True, 1)
    memory.ensure_exists()
    assert (
        memory.update_from_turn(
            Mock(generate=Mock(return_value="- New fact")), "Question", "Answer"
        )
        == 1
    )
    assert path.read_text() == "# Memory\n\n- New fact\n"
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "enabled, turn, output, calls, changed",
    [
        (False, 3, "# Memory\n- Updated", 0, False),
        (True, 0, "# Memory\n- Updated", 0, False),
        (True, 2, "# Memory\n- Updated", 0, False),
        (True, 3, "Invalid header", 1, False),
        (True, 3, "# Memory\n- Updated", 1, True),
    ],
)
def test_consolidation_requires_opt_in_cadence_and_valid_output(
    tmp_path, enabled, turn, output, calls, changed
):
    path = tmp_path / "MEMORY.md"
    original = "# Memory\n- Existing synthetic fact\n"
    path.write_text(original)
    store = MemoryStore(path, enabled, 3)
    generator = Mock(generate=Mock(return_value=output))
    assert store.maybe_consolidate(generator, turn) is changed
    assert generator.generate.call_count == calls
    assert path.read_text() == (output + "\n" if changed else original)


def test_failed_extraction_keeps_previous_memory(tmp_path):
    path = tmp_path / "MEMORY.md"
    original = "# Memory\n- Existing synthetic fact\n"
    path.write_text(original)
    generator = Mock(generate=Mock(side_effect=RuntimeError("synthetic failure")))
    with pytest.raises(RuntimeError):
        MemoryStore(path, True, 3).update_from_turn(generator, "Question", "Answer")
    assert path.read_text() == original


def prepare(store, outputs, turn=1):
    from arietta_voice.llm import ModelEvent

    outputs = iter(outputs)

    def stream(*a, **kw):
        yield ModelEvent("text", next(outputs))
        yield ModelEvent("done")

    gen = store.prepare_update(Mock(stream=stream), "Synthetic user", "Answer", turn)
    while True:
        try:
            next(gen)
        except StopIteration as done:
            return done.value


def test_background_proposal_has_no_writes_until_commit_and_consolidates_once(tmp_path):
    path = tmp_path / "MEMORY.md"
    store = MemoryStore(path, True, 2)
    proposal = prepare(store, ["- New fact", "# Memory\n- Consolidated fact"], turn=2)
    assert not path.exists()
    result = store.commit_update(proposal)
    assert result == dict(facts_added=1, consolidated=True, conflict=False)
    assert path.read_text() == "# Memory\n- Consolidated fact\n"
    assert path.stat().st_mode & 0o777 == 0o600


def test_edit_during_generation_is_preserved_and_proposal_discarded(tmp_path):
    path = tmp_path / "MEMORY.md"
    path.write_text("# Memory\n- Original\n")
    store = MemoryStore(path, True, 3)
    proposal = prepare(store, ["- Generated fact"])
    path.write_text("# Memory\n- Edited externally\n")
    assert store.commit_update(proposal)["conflict"]
    assert path.read_text() == "# Memory\n- Edited externally\n"


def test_cancel_or_failed_consolidation_never_commits_extracted_facts(tmp_path):
    from arietta_voice.llm import ModelEvent

    path = tmp_path / "MEMORY.md"
    path.write_text("# Memory\n- Original\n")
    store = MemoryStore(path, True, 1)
    closed = []

    def tokens(*a, **kw):
        try:
            yield ModelEvent("text", "- New fact")
            raise RuntimeError("Synthetic failure")
        finally:
            closed.append(True)

    gen = store.prepare_update(Mock(stream=tokens), "Synthetic", "Answer", 1)
    next(gen)
    gen.close()
    assert closed == [True]
    assert path.read_text() == "# Memory\n- Original\n"
    with pytest.raises(RuntimeError):
        list(store.prepare_update(Mock(stream=tokens), "Synthetic", "Answer", 1))
    assert path.read_text() == "# Memory\n- Original\n"


def test_background_exact_none_and_disabled_memory(tmp_path):
    path = tmp_path / "MEMORY.md"
    store = MemoryStore(path, True, 3)
    proposal = prepare(store, ["- Favorite author is None Such"])
    assert proposal["facts_added"] == 1
    proposal = prepare(store, ["NONE"])
    assert proposal["facts_added"] == 0
    disabled = MemoryStore(path, False, 3)
    assert prepare(disabled, []) is None
    disabled.commit_update(None)
    assert not path.exists()


def test_failed_second_generation_keeps_original_memory(tmp_path):
    from arietta_voice.llm import ModelEvent

    path = tmp_path / "MEMORY.md"
    original = "# Memory\n- Original\n"
    path.write_text(original)
    calls = []

    def tokens(*a, **kw):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("Synthetic consolidation failure")
        yield ModelEvent("text", "- New fact")
        yield ModelEvent("done")

    with pytest.raises(RuntimeError):
        list(MemoryStore(path, True, 1).prepare_update(Mock(stream=tokens), "User", "Answer", 1))
    assert len(calls) == 2
    assert path.read_text() == original


def test_oversized_existing_memory_is_preserved_without_model_call(tmp_path):
    path = tmp_path / "MEMORY.md"
    original = "x" * 128001
    path.write_text(original)
    model = Mock()
    with pytest.raises(ValueError, match="limit"):
        list(MemoryStore(path, True, 1).prepare_update(model, "User", "Answer", 1))
    model.stream.assert_not_called()
    assert path.read_text() == original


def test_correct_and_forget_duplicates_without_changing_freeform_markdown(tmp_path):
    from arietta_voice.memory import MemoryConflict, prompt_memory

    path = tmp_path / "MEMORY.md"
    original = b"# Memory\r\n\r\nA handwritten paragraph.\r\n\r\n"
    fenced = b"\r\n```\r\n- A code example\r\n```\r\n"
    path.write_bytes(original + b"- Likes tea\r\n* Likes tea\r\n" + fenced)
    store = MemoryStore(path, True, 2)
    before = store.inspect()
    (fact,) = before["entries"]
    assert fact["origin"] == "file"
    store.edit("Likes coffee", revision=before["revision"], entry_id=fact["id"])
    after = store.inspect()
    (fixed,) = after["entries"]
    assert fixed["edited"] and fixed["text"] == "Likes coffee"
    assert "arietta-memory" not in prompt_memory(after["content"])
    with pytest.raises(MemoryConflict):
        store.edit(None, revision=before["revision"], entry_id=fact["id"])
    store.edit(None, revision=after["revision"], entry_id=fixed["id"])
    assert path.read_bytes() == original + fenced
    assert store.inspect()["entries"] == []


def test_delayed_learning_cannot_recreate_forgotten_preference(tmp_path):
    import time

    store = MemoryStore(tmp_path / "MEMORY.md", True, 10)
    store.edit("- Prefers jasmine tea.\n")
    old_turn = time.time_ns()
    memory = store.inspect()
    store.edit(None, revision=memory["revision"], entry_id=0)
    # A queued voice or Chat job can start after forgetting and see the new
    # text as its expected snapshot, even though it describes an earlier turn.
    proposal = prepare(store, ["- Prefers jasmine tea."])
    proposal["submitted_ns"] = old_turn
    assert store.commit_update(proposal)["conflict"]
    assert store.inspect()["entries"] == []


def test_consolidation_preserves_corrected_fact_and_its_origin(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md", True, 2)
    store.edit("# Memory\n- Prefers oolong tea. <!-- arietta-memory:chat:edited -->\n- Gardens.\n")
    model = Mock(generate=Mock(return_value="# Memory\n- Grows herbs."))
    assert store.maybe_consolidate(model, 2)
    assert "oolong" not in str(model.generate.call_args)
    fixed = next(entry for entry in store.inspect()["entries"] if entry["edited"])
    assert (fixed["text"], fixed["origin"]) == ("Prefers oolong tea.", "chat")


def test_dated_save_preserves_freeform_and_same_text_on_other_dates(tmp_path):
    from datetime import date

    from arietta_voice.memory import prompt_memory

    store = MemoryStore(tmp_path / "MEMORY.md", True, 1)
    value = "The review is in the west hall."
    original = (
        f"# Memory\r\n- {value} <!-- arietta-memory:voice:saved=2000-03-14 -->\r\n"
        "\r\nHandwritten notes.\r\n```\r\n- an unfinished example"
    ).encode()
    store.path.write_bytes(original)
    store.save_dated(value, origin="chat")
    store.save_dated(value, origin="chat")
    memory = store.inspect()
    today, older = memory["entries"]
    assert (today["saved_on"], today["origin"]) == (date.today().isoformat(), "chat")
    assert older["saved_on"] == "2000-03-14"
    assert today["saved_on"] in prompt_memory(memory["content"])
    assert value not in prompt_memory(memory["content"], include_dated=False)
    store.edit(None, entry_id=today["id"], revision=memory["revision"])
    assert store.path.read_bytes() == original
    assert [entry["saved_on"] for entry in store.inspect()["entries"]] == ["2000-03-14"]


def test_consolidation_preserves_explicit_memories(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md", True, 1)
    store.edit("# Memory\n- Grows herbs.\n")
    store.save_dated("The decision was violet.", origin="voice")
    before = next(e for e in store.inspect()["entries"] if e["saved_on"])
    model = Mock(generate=Mock(return_value="# Memory\n- Grows basil."))
    assert store.maybe_consolidate(model, 1)
    assert "violet" not in str(model.generate.call_args)
    after = next(e for e in store.inspect()["entries"] if e["saved_on"])
    assert (after["text"], after["saved_on"], after["origin"]) == (
        before["text"],
        before["saved_on"],
        "voice",
    )
