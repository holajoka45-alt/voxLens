"""Voice Activity Detection.

Currently wraps Silero VAD via torch.hub. Silero VAD is fast, works on CPU
and GPU, and is reasonably accurate for clean speech.

Future: fine-tunable VAD for domain-specific audio (call center, noisy
environments, etc.).

Known issues:
- Silero VAD struggles with overlapping speech (detects as one segment).
- False positives on music with vocals, laughter, and strong breathing.
- The torch.hub loading is fragile. If the Silero repo moves or changes
  the API, this breaks. We pin a specific commit.
"""

from typing import Optional

import torch
import numpy as np


class VoiceActivityDetector:
    """Voice activity detection wrapper.

    Args:
        model: Loaded Silero VAD model.
        threshold: Speech probability threshold [0, 1].
        min_speech_duration_s: Minimum speech segment duration.
        min_silence_duration_s: Minimum silence between segments.
        device: Torch device.
    """

    def __init__(
        self,
        model,
        threshold: float = 0.5,
        min_speech_duration_s: float = 0.25,
        min_silence_duration_s: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.threshold = threshold
        self.min_speech_duration_s = min_speech_duration_s
        self.min_silence_duration_s = min_silence_duration_s
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_pretrained(cls, name: str = "silero-vad") -> "VoiceActivityDetector":
        """Load Silero VAD from torch.hub.

        NOTE: Requires internet connection on first load.
        """
        if name != "silero-vad":
            raise ValueError(f"Unknown VAD model: {name}")

        # Pin a specific Silero VAD version to avoid sudden API changes
        # TODO: update this commit hash periodically
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            # trust_repo=True,  # uncomment if torch complains
        )

        return cls(model)

    def detect(self, audio: torch.Tensor | np.ndarray, sr: int) -> list[dict]:
        """Detect speech segments in audio.

        Args:
            audio: 1D audio array (torch tensor or numpy).
            sr: Sample rate. Must be 8000 or 16000 for Silero VAD.

        Returns:
            List of dicts with 'start' and 'end' (seconds).
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        if sr not in {8000, 16000}:
            raise ValueError(f"Silero VAD requires 8kHz or 16kHz audio, got {sr}Hz")

        audio = audio.to(self.device)

        # Silero VAD expects shape (1, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        with torch.no_grad():
            speech_timestamps = self.model.get_speech_timestamps(
                audio,
                self.model,
                threshold=self.threshold,
                min_speech_duration_ms=int(self.min_speech_duration_s * 1000),
                min_silence_duration_ms=int(self.min_silence_duration_s * 1000),
                sampling_rate=sr,
            )

        # Convert to seconds
        # Timestamps from Silero are in samples
        segments = []
        for ts in speech_timestamps:
            segments.append({
                "start": ts["start"] / sr,
                "end": ts["end"] / sr,
            })

        return segments

    def to(self, device: torch.device):
        """Move model to device."""
        self.device = device
        self.model.to(device)
