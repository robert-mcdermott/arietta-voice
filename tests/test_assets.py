from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Barrier

import pytest

from arietta_voice.assets import ensure_download


def response(body, length):
    stream = BytesIO(body)
    stream.headers = {"Content-Length": str(length)}
    return stream


def test_incomplete_download_can_be_retried(monkeypatch, tmp_path):
    path = tmp_path / "model.onnx"
    replies = iter([response(b"ab", 3), response(b"abc", 3)])
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: next(replies))
    with pytest.raises(OSError, match="Incomplete download"):
        ensure_download("https://example.org/model", path)
    assert list(tmp_path.iterdir()) == []
    assert ensure_download("https://example.org/model", path).read_bytes() == b"abc"
    assert ensure_download("https://example.org/model", path) == path


def test_simultaneous_first_downloads_publish_complete_files(monkeypatch, tmp_path):
    both_started = Barrier(2)

    def open_model(*args, **kwargs):
        both_started.wait(timeout=5)
        return response(b"complete", 8)

    monkeypatch.setattr("urllib.request.urlopen", open_model)
    path = tmp_path / "model.onnx"
    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [pool.submit(ensure_download, "https://example.org/model", path) for _ in range(2)]
        assert all(call.result() == path for call in calls)
    assert path.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [path]
