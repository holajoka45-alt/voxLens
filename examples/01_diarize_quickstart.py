"""Quick start example: Diarize an audio file.

Usage:
    python examples/01_diarize_quickstart.py meeting.wav
"""

import sys
from voxlens.core import DiarizationPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python 01_diarize_quickstart.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Load pipeline (downloads models on first run)
    print("Loading pipeline...")
    pipeline = DiarizationPipeline.from_pretrained("voxlens/base")

    # Run diarization
    print(f"Diarizing: {audio_path}")
    result = pipeline.diarize(audio_path)

    # Print results
    print()
    print(result.summary())
    print()
    print("Segments:")
    for seg in result.segments:
        print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker}")

    # Save RTTM
    output_path = audio_path.rsplit(".", 1)[0] + ".rttm"
    result.to_rttm(output_path)
    print(f"\nSaved RTTM to: {output_path}")


if __name__ == "__main__":
    main()
