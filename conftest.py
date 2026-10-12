"""Shared isolation for unit tests and synthetic evaluations (never live models)."""
import os
import socket

import pytest


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    from arietta_voice import config

    original_locate = config.locate_project_root

    def explicit_root(start=None):
        if start is None:
            raise AssertionError("Tests must pass an explicit temporary config root/path")
        return original_locate(start)

    monkeypatch.setattr(config, "locate_project_root", explicit_root)
    for name in tuple(os.environ):
        if name.startswith(("AWS_", "ARIETTA_ADMIN_")) or name in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
            monkeypatch.delenv(name)
    for name in ("AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE"):
        monkeypatch.setenv(name, str(tmp_path / "empty-aws-config"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))

    def synthetic_only(*args, **kwargs):
        raise AssertionError("Live models, cloud clients and audio devices must be replaced with test fakes")

    for target in (
        "arietta_voice.llm.GemmaChatModel.ensure_loaded",
        "arietta_voice.stt.MoonshineSpeechToText.ensure_loaded",
        "arietta_voice.tts.KokoroTextToSpeech.ensure_loaded",
        "arietta_voice.turn_detection.SileroVoiceActivityDetector.ensure_loaded",
        "arietta_voice.turn_detection.SmartTurnEndpointDetector.ensure_loaded",
        "arietta_voice.knowledge.BgeTextEmbedder.__init__",
        "boto3.Session",
        "sounddevice.InputStream", "sounddevice.OutputStream", "sounddevice.play",
    ):
        monkeypatch.setattr(target, synthetic_only)

    connect = socket.socket.connect

    def no_network(sock, address):
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            raise AssertionError("Tests must not make network connections")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", no_network)
