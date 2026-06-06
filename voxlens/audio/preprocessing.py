"""Audio preprocessing utilities for diarization.

Chunking, augmentation, and feature extraction helpers.
"""

from typing import List, Tuple

import numpy as np
import torch


def chunk_audio(
    waveform: torch.Tensor,
    sr: int,
    chunk_duration_s: float = 30.0,
    overlap_s: float = 2.0,
) -> Tuple[List[torch.Tensor], List[float]]:
    """Split audio into overlapping chunks for processing.

    Args:
        waveform: (samples,) float tensor.
        sr: Sample rate.
        chunk_duration_s: Length of each chunk.
        overlap_s: Overlap between consecutive chunks.

    Returns:
        (chunks, chunk_starts): List of waveform tensors and their
                                start times in seconds.
    """
    chunk_samples = int(chunk_duration_s * sr)
    overlap_samples = int(overlap_s * sr)
    stride = chunk_samples - overlap_samples

    if stride <= 0:
        raise ValueError("Overlap must be less than chunk duration")

    total_samples = len(waveform)
    chunks = []
    starts = []

    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[start:end]
        chunks.append(chunk)
        starts.append(start / sr)
        start += stride

    return chunks, starts


class AudioAugmentation:
    """Experimental audio augmentations for diarization training.

    ⚠️ EXPERIMENTAL — Whether these help diarization is unclear.
    Use with caution and always validate on your task.

    Augmentations include:
    - Additive noise (white, pink, or from provided noise file)
    - Room impulse response (RIR) convolution
    - Random gain
    - Time stretching (via resampling trick)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def add_noise(self, waveform: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
        """Add white Gaussian noise at specified SNR.

        Args:
            waveform: (samples,) float tensor.
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            Noisy waveform.
        """
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        return waveform + noise

    def random_gain(self, waveform: torch.Tensor, db_range: float = 6.0) -> torch.Tensor:
        """Apply random gain within db_range.

        Args:
            waveform: (samples,) float tensor.
            db_range: Max gain change in dB (will be in [-db_range, +db_range]).

        Returns:
            Gain-adjusted waveform.
        """
        gain_db = self.rng.uniform(-db_range, db_range)
        gain_linear = 10 ** (gain_db / 20)
        return waveform * gain_linear

    # TODO: add RIR convolution (needs scipy.signal.fftconvolve)
    # TODO: add time stretching (needs librosa.effects.time_stretch)
