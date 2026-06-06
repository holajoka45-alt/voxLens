"""RTTM file I/O.

RTTM (Rich Transcription Time Marked) is the standard format for
diarization output. Every diarization tool reads/writes RTTM.

Format: SPEAKER <file> <ch> <start> <duration> <type> <speaker> <confidence>
"""

from pathlib import Path
from typing import List, Optional

from voxlens.core.pipeline import DiarizationSegment


class RTTMWriter:
    """Write diarization results to RTTM format."""

    @staticmethod
    def write(
        path: str | Path,
        segments: List[DiarizationSegment],
        audio_duration: float = 0.0,
        file_id: str = "recording",
        channel: int = 1,
    ):
        """Write segments to RTTM file.

        Args:
            path: Output file path.
            segments: List of DiarizationSegment objects.
            audio_duration: Total audio duration (for reference).
            file_id: Recording ID (used in RTTM header).
            channel: Audio channel number.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for seg in segments:
                duration = seg.end - seg.start
                f.write(
                    f"SPEAKER {file_id} {channel} {seg.start:.3f} {duration:.3f} "
                    f"<NA> <NA> {seg.speaker} {seg.confidence:.3f}\n"
                )


class RTTMReader:
    """Read RTTM files into DiarizationSegment list.

    Useful for loading ground truth annotations or third-party outputs.
    """

    @staticmethod
    def read(path: str | Path) -> List[DiarizationSegment]:
        """Parse an RTTM file.

        Args:
            path: RTTM file path.

        Returns:
            List of DiarizationSegment objects.
        """
        segments = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if parts[0] != "SPEAKER" or len(parts) < 8:
                    continue

                # SPEAKER <file> <ch> <start> <duration> <type> <subtype> <speaker> [conf]
                start = float(parts[2])
                duration = float(parts[3])
                speaker = parts[7]
                confidence = float(parts[8]) if len(parts) > 8 else 1.0

                segments.append(DiarizationSegment(
                    start=start,
                    end=start + duration,
                    speaker=speaker,
                    confidence=confidence,
                ))

        return sorted(segments, key=lambda s: s.start)
