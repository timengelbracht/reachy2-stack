from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, List
from pathlib import Path

from reachy2_stack.core.client import ReachyClient


@dataclass
class AudioSkill:
    """Comprehensive audio skill for managing & playing sounds on Reachy.

    Features:
    - upload(local_path)
    - list_files()
    - play_once(filename)
    - replay(filename, n_times)
    - stop()
    - remove(filename)
    - download(filename, local_path)
    """

    client: ReachyClient

    def _reachy(self):
        """Convenience accessor for the underlying ReachySDK client."""
        return self.client.connect_reachy

    # ----------------------------------------------------------
    # File management
    # ----------------------------------------------------------

    def list_files(self) -> List[str]:
        """Return list of audio files currently stored on Reachy."""
        reachy = self._reachy()
        return reachy.audio.get_audio_files()

    def upload(self, local_path: str | Path) -> None:
        """Upload a local .wav/.mp3/.ogg file to Reachy."""
        reachy = self._reachy()
        local_path = Path(local_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Local audio file not found: {local_path}")

        reachy.audio.upload_audio_file(str(local_path))

    def remove(self, filename: str) -> None:
        """Delete an audio file from Reachy's temporary audio folder."""
        reachy = self._reachy()
        reachy.audio.remove_audio_file(filename)

    def download(self, filename: str, local_path: str | Path) -> None:
        """Download a file from Reachy to the local machine."""
        reachy = self._reachy()
        reachy.audio.download_audio_file(filename, str(local_path))

    # ----------------------------------------------------------
    # Playback
    # ----------------------------------------------------------

    def play_once(self, filename: str) -> None:
        """Play a single audio file by name."""
        reachy = self._reachy()
        reachy.audio.play_audio_file(filename)

    def replay(
        self,
        filename: str,
        n_times: int = 1,
        wait_secs: Optional[float] = None,
    ) -> None:
        """Replay an uploaded audio file multiple times.

        Args:
            filename: Name of the remote file (e.g. 'hello.wav')
            n_times: Number of repetitions
            wait_secs: Optional pause between repetitions
        """
        reachy = self._reachy()

        for i in range(n_times):
            reachy.audio.play_audio_file(filename)
            if wait_secs is not None and i < n_times - 1:
                time.sleep(wait_secs)

    def stop(self) -> None:
        """Stop any audio currently playing."""
        reachy = self._reachy()
        reachy.audio.stop_playing()
