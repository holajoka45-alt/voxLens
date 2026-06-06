"""Tests for the diarization pipeline."""

import pytest
import numpy as np
import torch

from voxlens.core.pipeline import DiarizationPipeline, DiarizationResult
from voxlens.utils.rttm import RTTMWriter, RTTMReader


class TestDiarizationResult:
    def test_to_rttm_roundtrip(self, tmp_path):
        from voxlens.core.pipeline import DiarizationSegment

        segments = [
            DiarizationSegment(0.0, 2.5, "SPEAKER_0"),
            DiarizationSegment(3.0, 5.0, "SPEAKER_1"),
            DiarizationSegment(5.0, 7.2, "SPEAKER_0"),
        ]

        result = DiarizationResult(
            segments=segments,
            audio_duration_s=7.2,
            processing_time_s=1.5,
            n_speakers_estimated=2,
        )

        rttm_path = tmp_path / "test.rttm"
        result.to_rttm(rttm_path)

        # Read back
        read_segments = RTTMReader.read(rttm_path)
        assert len(read_segments) == 3

        # Check values (allow small float differences)
        assert read_segments[0].speaker == "SPEAKER_0"
        assert abs(read_segments[0].start - 0.0) < 0.01
        assert abs(read_segments[0].end - 2.5) < 0.01

    def test_summary_string(self):
        from voxlens.core.pipeline import DiarizationSegment

        segments = [
            DiarizationSegment(0.0, 5.0, "SPEAKER_0"),
            DiarizationSegment(5.0, 10.0, "SPEAKER_1"),
        ]

        result = DiarizationResult(
            segments=segments,
            audio_duration_s=10.0,
            processing_time_s=2.0,
            n_speakers_estimated=2,
        )

        summary = result.summary()
        assert "10.0s" in summary
        assert "SPEAKER_0" in summary
        assert "50%" in summary


class TestMergeAdjacent:
    def test_merge_same_speaker(self):
        from voxlens.core.pipeline import DiarizationPipeline, DiarizationSegment

        segments = [
            DiarizationSegment(0.0, 2.0, "SPEAKER_0"),
            DiarizationSegment(2.1, 4.0, "SPEAKER_0"),  # 0.1s gap
            DiarizationSegment(5.0, 7.0, "SPEAKER_1"),
        ]

        merged = DiarizationPipeline._merge_adjacent(segments, gap_tolerance=0.5)
        assert len(merged) == 2
        assert merged[0].start == 0.0
        assert merged[0].end == 4.0
        assert merged[0].speaker == "SPEAKER_0"

    def test_no_merge_different_speaker(self):
        from voxlens.core.pipeline import DiarizationPipeline, DiarizationSegment

        segments = [
            DiarizationSegment(0.0, 2.0, "SPEAKER_0"),
            DiarizationSegment(2.0, 4.0, "SPEAKER_1"),
        ]

        merged = DiarizationPipeline._merge_adjacent(segments)
        assert len(merged) == 2
