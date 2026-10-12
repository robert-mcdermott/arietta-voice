from unittest.mock import Mock

import numpy as np

from arietta_voice.config import default_config
from arietta_voice.inference import InferenceEngine
from arietta_voice.models import ToolResponse


def test_tts_pieces_are_bounded_and_closing_clears_generator(tmp_path):
    engine = InferenceEngine(default_config(tmp_path))
    closed = []
    async def stream(text):
        try:
            yield np.arange(10_000, dtype=np.float32), 24000
        finally:
            closed.append(True)
    engine._speaker = Mock(create_stream=stream)
    try:
        engine.handle('tts_start', 'Synthetic')
        pieces = [engine.handle('tts_next', None) for _ in range(3)]
        assert [len(piece[0]) for piece in pieces] == [4096, 4096, 1808]
        np.testing.assert_equal(np.concatenate([piece[0] for piece in pieces]), np.arange(10_000))
        engine.handle('tts_close', None)
        assert closed == [True]
        assert engine._tts is engine._tts_pending is None
    finally:
        engine._loop.close()


def test_voice_tool_instances_survive_successful_turns(tmp_path, monkeypatch):
    from arietta_voice import tools
    tool = Mock()
    tool.maybe_handle.side_effect = [ToolResponse('counter', 'First'), ToolResponse('counter', 'Second')]
    constructor = Mock(return_value=tool)
    monkeypatch.setattr(tools, 'ToolRegistry', constructor)
    engine = InferenceEngine(default_config(tmp_path))
    def payload():
        return dict(message='Synthetic', history=[], enabled_tools=['counter'], enabled_model_tools=[], voice_turn=True)
    try:
        assert engine.handle('chat', payload())['response'] == 'First'
        assert engine.handle('chat', payload())['response'] == 'Second'
        constructor.assert_called_once()
        assert engine.service()._llm is None
    finally:
        engine.service().worker.close()
        engine._loop.close()
