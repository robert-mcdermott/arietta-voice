"""Console API and frontend."""

from arietta_voice.web.app import create_app
from arietta_voice.web.service import AriettaApiService

__all__ = ["AriettaApiService", "create_app"]
