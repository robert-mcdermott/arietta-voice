import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from arietta_voice.stt import MoonshineSpeechToText

# conftest blocks real model loading during tests; keep the real method for one test.
REAL_ENSURE_LOADED = MoonshineSpeechToText.ensure_loaded


class FakeStream:
    def __init__(self, lines, fail=False):
        self.lines, self.fail = lines, fail
        self.calls = []

    def start(self):
        self.calls.append("start")

    def add_audio(self, samples, rate):
        self.calls.append(("add_audio", len(samples), rate))
        if self.fail:
            raise RuntimeError("native failure")

    def stop(self):
        self.calls.append("stop")
        return SimpleNamespace(lines=[SimpleNamespace(text=text) for text in self.lines])

    def close(self):
        self.calls.append("close")


def loaded(streams):
    stt = MoonshineSpeechToText("en")
    stt._transcriber = Mock(create_stream=Mock(side_effect=streams))
    stt.ensure_loaded = lambda: None
    return stt


def test_each_utterance_gets_its_own_stream_and_lines_are_joined():
    first = FakeStream(["What is the capital", "", "of France?"])
    second = FakeStream(["Next question."])
    stt = loaded([first, second])
    audio = np.zeros(16000, dtype=np.float32)
    assert stt.transcribe(audio, 16000) == "What is the capital of France?"
    assert stt.transcribe(audio, 16000) == "Next question."
    assert first.calls == ["start", ("add_audio", 16000, 16000), "stop", "close"]
    assert second.calls[-1] == "close"
    # No partial updates: the whole utterance is decoded once at stop().
    for call in stt._transcriber.create_stream.call_args_list:
        assert call.kwargs["update_interval"] >= 60


def test_a_failed_stream_is_still_closed():
    failing = FakeStream([], fail=True)
    stt = loaded([failing])
    with pytest.raises(RuntimeError):
        stt.transcribe(np.zeros(512, dtype=np.float32), 16000)
    assert failing.calls[-1] == "close"


def test_loading_disables_speaker_identification(monkeypatch):
    created = {}

    def transcriber(**options):
        created.update(options)
        return Mock()

    module = types.ModuleType("moonshine_voice")
    module.Transcriber = transcriber
    module.get_model_for_language = lambda language: (f"/models/{language}", "ARCH")
    monkeypatch.setitem(sys.modules, "moonshine_voice", module)
    stt = MoonshineSpeechToText("en")
    REAL_ENSURE_LOADED(stt)
    assert stt.is_loaded()
    assert created == {
        "model_path": "/models/en",
        "model_arch": "ARCH",
        "options": {"identify_speakers": "false"},
    }
