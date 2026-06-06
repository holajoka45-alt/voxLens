"""End-to-end diarization pipeline.

This is the main entry point. It chains VAD → embedding extraction →
clustering into one call. Most users will only need this module.

The pipeline is designed to be modular — you can swap out the VAD model,
embedding model, or clustering algorithm independently.

Known limitations:
- Entire audio file is loaded into memory. For very long recordings (>4 hours
  at 16kHz), this can exceed 2 GB RAM. Use chunking or streaming (planned).
- Speaker count estimation is heuristic. It works well for 2-8 speakers in
  meeting-style audio, degrades for phone calls, and fails catastrophically
  for monologue detection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from voxlens.audio.loader import load_audio, get_audio_info
from voxlens.audio.preprocessing import chunk_audio
from voxlens.models.embedding import SpeakerEmbedding
from voxlens.models.vad import VoiceActivityDetector
from voxlens.core.clusterer import Clusterer, ClusterConfig
from voxlens.utils.rttm import RTTMWriter
from voxlens.utils.gpu import get_device


@dataclass
class DiarizationSegment:
    """A single speaker segment.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        speaker: Speaker label (e.g., "SPEAKER_0").
        confidence: Optional confidence score [0, 1].
    """
    start: float
    end: float
    speaker: str
    confidence: float = 1.0


@dataclass
class DiarizationResult:
    """Result of a diarization run.

    Attributes:
        segments: List of speaker segments.
        audio_duration_s: Duration of the input audio.
        processing_time_s: Wall-clock time for processing.
        n_speakers_estimated: Estimated number of speakers (may differ from actual).
        config: Snapshot of pipeline config used.
    """
    segments: list[DiarizationSegment]
    audio_duration_s: float
    processing_time_s: float
    n_speakers_estimated: int
    config: dict = field(default_factory=dict)

    def to_rttm(self, path: str | Path) -> None:
        """Write segments to RTTM file (standard diarization format).

        Args:
            path: Output .rttm file path.
        """
        RTTMWriter.write(path, self.segments, audio_duration=self.audio_duration_s)

    def summary(self) -> str:
        """Human-readable summary string."""
        speaker_times = {}
        for seg in self.segments:
            speaker_times[seg.speaker] = speaker_times.get(seg.speaker, 0) + (seg.end - seg.start)

        lines = [
            f"Diarization complete in {self.processing_time_s:.1f}s",
            f"Audio duration: {self.audio_duration_s:.1f}s",
            f"Estimated speakers: {self.n_speakers_estimated}",
            f"Segments: {len(self.segments)}",
            "",
            "Speaker breakdown:",
        ]
        for speaker, total in sorted(speaker_times.items()):
            pct = total / self.audio_duration_s * 100
            lines.append(f"  {speaker}: {total:.1f}s ({pct:.0f}%)")

        return "\n".join(lines)


class DiarizationPipeline:
    """End-to-end speaker diarization pipeline.

    Chains VAD → embedding extraction → clustering.

    Args:
        vad_model: Voice activity detection model.
        embedding_model: Speaker embedding extractor.
        clusterer: Clustering algorithm for grouping embeddings.
        device: Torch device to use.
        sample_rate: Target sample rate (Hz). All audio is resampled to this.
        chunk_duration_s: Duration of audio chunks for processing.
                          Shorter = less memory, more boundary artifacts.
        chunk_overlap_s: Overlap between chunks to reduce boundary errors.
    """

    def __init__(
        self,
        vad_model: VoiceActivityDetector,
        embedding_model: SpeakerEmbedding,
        clusterer: Optional[Clusterer] = None,
        device: Optional[torch.device] = None,
        sample_rate: int = 16000,
        chunk_duration_s: float = 30.0,
        chunk_overlap_s: float = 2.0,
    ):
        self.vad = vad_model
        self.embedding = embedding_model
        self.clusterer = clusterer or Clusterer(ClusterConfig(method="spectral"))
        self.device = device or get_device()
        self.sample_rate = sample_rate
        self.chunk_duration_s = chunk_duration_s
        self.chunk_overlap_s = chunk_overlap_s

        # Move models to device
        self.vad.to(self.device)
        self.embedding.to(self.device)

    @classmethod
    def from_pretrained(cls, name: str = "voxlens/base", **kwargs) -> "DiarizationPipeline":
        """Load a pre-configured pipeline.

        Currently only "voxlens/base" is supported — ECAPA-TDNN + Silero VAD
        + spectral clustering. Model weights are downloaded from HuggingFace Hub.

        Args:
            name: Pipeline preset name.
            **kwargs: Override default config values.

        NOTE: The pretrained weights are work-in-progress. Currently downloads
        from a HuggingFace repo that may not exist yet. Fallback initializes
        with random weights — which obviously won't work well.
        """
        # TODO: actually upload pretrained weights to HF Hub
        # TODO: add "voxlens/fast" preset (smaller model, CPU-friendly)

        vad = VoiceActivityDetector.from_pretrained("silero-vad")
        embedding = SpeakerEmbedding.from_pretrained("ecapa-tdnn-voxceleb")

        # Default config
        cfg = dict(
            sample_rate=16000,
            chunk_duration_s=30.0,
            chunk_overlap_s=2.0,
        )
        cfg.update(kwargs)

        return cls(vad_model=vad, embedding_model=embedding, **cfg)

    def diarize(self, audio_path: str | Path) -> DiarizationResult:
        """Run diarization on an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, M4A, etc.).

        Returns:
            DiarizationResult with segments and metadata.
        """
        t_start = time.perf_counter()

        # Load audio
        waveform, sr = load_audio(audio_path, target_sr=self.sample_rate)
        audio_duration = len(waveform) / sr

        # Get chunks
        chunks, chunk_starts = chunk_audio(
            waveform,
            sr=sr,
            chunk_duration_s=self.chunk_duration_s,
            overlap_s=self.chunk_overlap_s,
        )

        # Process each chunk: VAD → embedding
        all_embeddings = []
        all_segment_boundaries = []

        for i, (chunk, chunk_start) in enumerate(zip(chunks, chunk_starts)):
            # VAD
            speech_timestamps = self.vad.detect(chunk, sr)

            for ts in speech_timestamps:
                # Extract speech segment
                start_sample = int(ts["start"] * sr)
                end_sample = int(ts["end"] * sr)
                speech = chunk[start_sample:end_sample]

                if len(speech) < sr * 0.5:  # skip segments < 0.5s
                    continue

                # Extract embedding
                embedding = self.embedding.extract(speech, sr)
                all_embeddings.append(embedding.cpu().numpy())

                # Store absolute time boundaries
                abs_start = chunk_start + ts["start"]
                abs_end = chunk_start + ts["end"]
                all_segment_boundaries.append((abs_start, abs_end))

        if not all_embeddings:
            print("Warning: No speech detected in audio.")
            return DiarizationResult(
                segments=[],
                audio_duration_s=audio_duration,
                processing_time_s=time.perf_counter() - t_start,
                n_speakers_estimated=0,
            )

        # Cluster embeddings
        embeddings_array = np.stack(all_embeddings)
        labels, n_speakers = self.clusterer.cluster(embeddings_array)

        # Build segments
        segments = []
        for (start, end), label in zip(all_segment_boundaries, labels):
            segments.append(DiarizationSegment(
                start=round(start, 3),
                end=round(end, 3),
                speaker=f"SPEAKER_{label}",
            ))

        # Merge adjacent segments from same speaker
        segments = self._merge_adjacent(segments)

        processing_time = time.perf_counter() - t_start

        return DiarizationResult(
            segments=segments,
            audio_duration_s=audio_duration,
            processing_time_s=processing_time,
            n_speakers_estimated=n_speakers,
            config={
                "sample_rate": self.sample_rate,
                "chunk_duration_s": self.chunk_duration_s,
                "chunk_overlap_s": self.chunk_overlap_s,
                "cluster_method": self.clusterer.config.method,
            },
        )

    @staticmethod
    def _merge_adjacent(segments: list[DiarizationSegment], gap_tolerance: float = 0.5) -> list[DiarizationSegment]:
        """Merge segments from the same speaker if they're close together.

        Args:
            segments: Sorted list of segments.
            gap_tolerance: Max gap (seconds) to merge across.

        Returns:
            Merged segment list.
        """
        if not segments:
            return segments

        # Sort by start time
        segments = sorted(segments, key=lambda s: s.start)

        merged = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            if seg.speaker == last.speaker and (seg.start - last.end) <= gap_tolerance:
                # Merge
                last.end = max(last.end, seg.end)
                last.confidence = (last.confidence + seg.confidence) / 2
            else:
                merged.append(seg)

        return merged
