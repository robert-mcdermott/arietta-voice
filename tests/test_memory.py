from pathlib import Path
from unittest.mock import Mock

import pytest

from arietta_voice.memory import MemoryStore


def test_interrupted_memory_replacement_retains_previous_file(tmp_path, monkeypatch):
    path = tmp_path / 'MEMORY.md'
    original = '# Memory\n- Existing fact\n'
    path.write_text(original)
    memory = MemoryStore(path, True, 1)
    generator = Mock(generate=Mock(return_value='# Memory\n- Consolidated fact'))
    monkeypatch.setattr(Path, 'replace', Mock(side_effect=OSError('simulated write failure')))
    with pytest.raises(OSError):
        memory.maybe_consolidate(generator, 1)
    assert path.read_text() == original
    assert not path.with_name('.MEMORY.md.working').exists()


def test_memory_updates_are_private_and_atomic(tmp_path):
    path = tmp_path / 'MEMORY.md'
    memory = MemoryStore(path, True, 1)
    memory.ensure_exists()
    assert memory.update_from_turn(Mock(generate=Mock(return_value='- New fact')), 'Question', 'Answer') == 1
    assert path.read_text() == '# Memory\n\n- New fact\n'
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("enabled, turn, output, calls, changed", [
    (False, 3, "# Memory\n- Updated", 0, False),
    (True, 0, "# Memory\n- Updated", 0, False),
    (True, 2, "# Memory\n- Updated", 0, False),
    (True, 3, "Invalid header", 1, False),
    (True, 3, "# Memory\n- Updated", 1, True),
])
def test_consolidation_requires_opt_in_cadence_and_valid_output(tmp_path, enabled, turn, output, calls, changed):
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
        yield ModelEvent('text', next(outputs))
        yield ModelEvent('done')
    gen = store.prepare_update(Mock(stream=stream), 'Synthetic user','Answer',turn)
    while True:
        try:
            next(gen)
        except StopIteration as done:
            return done.value


def test_background_proposal_has_no_writes_until_commit_and_consolidates_once(tmp_path):
    path = tmp_path/'MEMORY.md'
    store = MemoryStore(path,True,2)
    proposal = prepare(store,['- New fact','# Memory\n- Consolidated fact'],turn=2)
    assert not path.exists()
    result = store.commit_update(proposal)
    assert result == dict(facts_added=1,consolidated=True,conflict=False)
    assert path.read_text() == '# Memory\n- Consolidated fact\n'
    assert path.stat().st_mode & 0o777 == 0o600


def test_edit_during_generation_is_preserved_and_proposal_discarded(tmp_path):
    path = tmp_path/'MEMORY.md'
    path.write_text('# Memory\n- Original\n')
    store = MemoryStore(path,True,3)
    proposal = prepare(store,['- Generated fact'])
    path.write_text('# Memory\n- Edited externally\n')
    assert store.commit_update(proposal)['conflict']
    assert path.read_text() == '# Memory\n- Edited externally\n'


def test_cancel_or_failed_consolidation_never_commits_extracted_facts(tmp_path):
    from arietta_voice.llm import ModelEvent
    path = tmp_path/'MEMORY.md'
    path.write_text('# Memory\n- Original\n')
    store = MemoryStore(path,True,1)
    closed = []
    def tokens(*a,**kw):
        try:
            yield ModelEvent('text','- New fact')
            raise RuntimeError('Synthetic failure')
        finally:
            closed.append(True)
    gen = store.prepare_update(Mock(stream=tokens),'Synthetic','Answer',1)
    next(gen)
    gen.close()
    assert closed == [True]
    assert path.read_text() == '# Memory\n- Original\n'
    with pytest.raises(RuntimeError):
        list(store.prepare_update(Mock(stream=tokens),'Synthetic','Answer',1))
    assert path.read_text() == '# Memory\n- Original\n'


def test_background_exact_none_and_disabled_memory(tmp_path):
    path = tmp_path/'MEMORY.md'
    store = MemoryStore(path,True,3)
    proposal = prepare(store,['- Favorite author is None Such'])
    assert proposal['facts_added'] == 1
    proposal = prepare(store,['NONE'])
    assert proposal['facts_added'] == 0
    disabled = MemoryStore(path,False,3)
    assert prepare(disabled,[]) is None
    disabled.commit_update(None)
    assert not path.exists()


def test_failed_second_generation_keeps_original_memory(tmp_path):
    from arietta_voice.llm import ModelEvent
    path = tmp_path/'MEMORY.md'
    original = '# Memory\n- Original\n'
    path.write_text(original)
    calls = []
    def tokens(*a,**kw):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError('Synthetic consolidation failure')
        yield ModelEvent('text','- New fact')
        yield ModelEvent('done')
    with pytest.raises(RuntimeError):
        list(MemoryStore(path,True,1).prepare_update(Mock(stream=tokens),'User','Answer',1))
    assert len(calls) == 2
    assert path.read_text() == original


def test_oversized_existing_memory_is_preserved_without_model_call(tmp_path):
    path = tmp_path/'MEMORY.md'
    original = 'x'*128001
    path.write_text(original)
    model = Mock()
    with pytest.raises(ValueError,match='limit'):
        list(MemoryStore(path,True,1).prepare_update(model,'User','Answer',1))
    model.stream.assert_not_called()
    assert path.read_text() == original
