"""Arietta Voice."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("arietta-voice")
except PackageNotFoundError:
    # A source-only import has no installed distribution metadata. Run uv sync
    # to install the version declared in pyproject.toml.
    __version__ = "0+unknown"
