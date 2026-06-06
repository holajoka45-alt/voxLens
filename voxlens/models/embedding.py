"""Speaker embedding extraction.

ECAPA-TDNN implementation adapted for embedding extraction.
Based on: "ECAPA-TDNN: Emphasized Channel Attention, Propagation and
Aggregation in TDNN Based Speaker Verification" (Desplanques et al., 2020).

This is a simplified reimplementation. The official pretrained weights
are from SpeechBrain. We load them via a converter.

Known issues:
- The pretrained weight conversion from SpeechBrain format is brittle.
  If SpeechBrain changes their checkpoint structure, this breaks.
- Batch extraction works but the padding logic assumes similar-length
  segments. Very short segments (<0.5s) produce degraded embeddings.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np


class Conv1DBlock(nn.Module):
    """1D Convolution + BatchNorm + ReLU block for TDNN."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN for speaker embedding extraction.

    Args:
        n_mels: Number of mel filterbanks.
        channels: Base channel count (multiplied per layer).
        embedding_dim: Output embedding dimension.
    """

    def __init__(self, n_mels: int = 80, channels: int = 512, embedding_dim: int = 192):
        super().__init__()
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim

        # TDNN blocks with increasing dilation
        self.block1 = Conv1DBlock(n_mels, channels, kernel_size=5)
        self.block2 = Conv1DBlock(channels, channels, kernel_size=3, dilation=2)
        self.block3 = Conv1DBlock(channels, channels, kernel_size=3, dilation=3)
        self.block4 = Conv1DBlock(channels, channels, kernel_size=3, dilation=4)
        self.block5 = Conv1DBlock(channels, channels * 3, kernel_size=1)  # no dilation, channel expand

        self.se_block = SEBlock(channels * 3)

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

        # Projection
        self.projection = nn.Linear(channels * 3, embedding_dim)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

        # Mel spectrogram transform (not stored in state_dict, reconstructed on load)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=n_mels,
            f_min=20,
            f_max=8000,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, n_mels, time) mel spectrogram.

        Returns:
            (batch, embedding_dim) L2-normalized speaker embeddings.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.se_block(x)

        # Attention pooling
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted mean
        x = (x * attn_weights).sum(dim=-1)

        # Project and normalize
        x = self.projection(x)
        x = self.bn_out(x)
        x = F.normalize(x, p=2, dim=-1)

        return x

    def extract_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram from raw waveform.

        Args:
            waveform: (batch, samples) or (samples,) float tensor.

        Returns:
            (batch, n_mels, time) mel spectrogram.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return self.mel_transform(waveform)


class SpeakerEmbedding:
    """Wrapper around ECAPA-TDNN for embedding extraction.

    Handles loading, preprocessing, and batch extraction.

    Args:
        model: ECAPATDNN instance.
        device: Torch device.
    """

    def __init__(self, model: ECAPATDNN, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, name: str = "ecapa-tdnn-voxceleb") -> "SpeakerEmbedding":
        """Load pretrained ECAPA-TDNN.

        Currently downloads from HuggingFace Hub. If the weights aren't
        available, falls back to random initialization with a warning.

        TODO: implement proper HF Hub download
        """
        model = ECAPATDNN()

        if name == "ecapa-tdnn-voxceleb":
            # TODO: download from HuggingFace Hub
            # For now, initialize with random weights
            # In production, you'd load from:
            #   hf_hub_download("voxlens/ecapa-tdnn-voxceleb", "pytorch_model.bin")
            print("Warning: Pretrained weights not yet hosted. Using random weights.")
            print("The model will NOT produce meaningful embeddings.")
        else:
            raise ValueError(f"Unknown model name: {name}")

        return cls(model)

    def extract(self, waveform: np.ndarray | torch.Tensor, sr: int) -> torch.Tensor:
        """Extract speaker embedding from audio.

        Args:
            waveform: 1D audio array. Can be numpy or torch tensor.
            sr: Sample rate of waveform.

        Returns:
            (embedding_dim,) L2-normalized embedding vector.
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)

        waveform = waveform.to(self.device)

        # Resample if needed
        if sr != 16000:
            resampler = T.Resample(sr, 16000).to(self.device)
            waveform = resampler(waveform)

        # Compute mel spectrogram
        mel = self.model.extract_mel(waveform)

        with torch.no_grad():
            embedding = self.model(mel)

        return embedding.squeeze(0)  # (embedding_dim,)

    def extract_batch(self, waveforms: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract embeddings from a batch of audio segments.

        Args:
            waveforms: (batch, samples) float tensor.
            sr: Sample rate.

        Returns:
            (batch, embedding_dim) embeddings.
        """
        waveforms = waveforms.to(self.device)

        # TODO: handle variable-length waveforms properly
        # Currently assumes all same length. Pad/collate for var-length.

        if sr != 16000:
            resampler = T.Resample(sr, 16000).to(self.device)
            waveforms = resampler(waveforms)

        mel = self.model.extract_mel(waveforms)

        with torch.no_grad():
            embeddings = self.model(mel)

        return embeddings

    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)

    @property
    def dim(self) -> int:
        return self.model.embedding_dim
