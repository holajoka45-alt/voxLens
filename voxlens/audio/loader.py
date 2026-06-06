"""Audio loading utilities.

Supports WAV, MP3, FLAC, M4A, and anything else that librosa/soundfile
can handle.

Known limitations:
- M4A (AAC) loading requires ffmpeg on the system PATH.
- 32-bit float WAV files may be clipped if they exceed [-1, 1].
- Very long files (>4 hours) are loaded entirely into RAM.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch


def load_audio(
    path: Union[str, Path],
    target_sr: int = 16000,
    mono: bool = True,
    start: float = 0.0,
    duration: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """Load audio file, resample, convert to mono, return as tensor.

    Args:
        path: Audio file path.
        target_sr: Target sample rate.
        mono: If True, convert to mono by averaging channels.
        start: Start time in seconds.
        duration: Duration to load. None = entire file.

    Returns:
        (waveform, sample_rate): waveform is (samples,) float32 tensor.

    Raises:
        FileNotFoundError: If audio file doesn't exist.
        RuntimeError: If format is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Try soundfile first (WAV, FLAC), fall back to librosa (MP3, M4A)
    # TODO: add audioread for more formats
    try:
        import soundfile as sf

        info = sf.info(str(path))
        sr = info.samplerate

        # Calculate frame offset
        offset_frames = int(start * sr)
        frames_to_read = int(duration * sr) if duration else -1

        audio, _ = sf.read(
            str(path),
            start=offset_frames,
            frames=frames_to_read,
            dtype="float32",
        )
    except Exception:
        import librosa

        # librosa handles MP3, M4A, etc. but is slower
        audio, sr = librosa.load(
            str(path),
            sr=target_sr if target_sr else None,
            mono=mono,
            offset=start,
            duration=duration,
        )
        return torch.from_numpy(audio).float(), sr

    # Convert to mono
    if mono and audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return torch.from_numpy(audio.copy()).float(), sr


def get_audio_info(path: Union[str, Path]) -> dict:
    """Get audio file metadata without loading the full file.

    Args:
        path: Audio file path.

    Returns:
        Dict with 'sample_rate', 'duration_s', 'n_channels', 'format'.
    """
    path = Path(path)

    try:
        import soundfile as sf

        info = sf.info(str(path))
        return {
            "sample_rate": info.samplerate,
            "duration_s": info.duration,
            "n_channels": info.channels,
            "format": info.format,
        }
    except Exception:
        # Fallback: load with librosa (slow, but works for more formats)
        import librosa

        y, sr = librosa.load(str(path), sr=None, mono=False)
        duration = len(y) / sr if y.ndim == 1 else y.shape[1] / sr
        n_channels = 1 if y.ndim == 1 else y.shape[0]
        return {
            "sample_rate": sr,
            "duration_s": duration,
            "n_channels": n_channels,
            "format": str(path.suffix).lower(),
        }
