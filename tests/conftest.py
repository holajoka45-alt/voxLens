"""Shared fixtures for VoxLens tests."""

import pytest
import numpy as np
import torch


@pytest.fixture
def sample_audio():
    """Generate a tiny synthetic audio file for testing."""
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz tone
    return torch.from_numpy(audio), sr


@pytest.fixture
def sample_embeddings():
    """Generate synthetic speaker embeddings."""
    np.random.seed(42)
    n_speakers = 3
    n_per_speaker = 10
    dim = 192

    embeddings = []
    for i in range(n_speakers):
        centroid = np.random.randn(dim).astype(np.float32)
        centroid = centroid / np.linalg.norm(centroid)
        for _ in range(n_per_speaker):
            emb = centroid + 0.05 * np.random.randn(dim).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

    return np.stack(embeddings)
