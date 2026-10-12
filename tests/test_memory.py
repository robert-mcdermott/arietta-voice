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
