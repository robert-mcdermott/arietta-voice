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
