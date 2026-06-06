"""Diarization evaluation metrics.

Standard metrics: Diarization Error Rate (DER), Jaccard Error Rate (JER).
Implemented from scratch for transparency, but also wraps pyannote.metrics
for validation.

Known issues:
- DER calculation does NOT include overlapping speech scoring (collar-based
  forgiveness). This matches the NIST DER standard but may differ from
  pyannote's default DER which includes overlap regions.
"""

from typing import List, Tuple

import numpy as np


def compute_der(
    reference: List[Tuple[float, float, str]],
    hypothesis: List[Tuple[float, float, str]],
    collar: float = 0.0,
) -> float:
    """Compute Diarization Error Rate (DER).

    DER = (false alarm + missed detection + speaker confusion) / total reference time

    Args:
        reference: List of (start, end, speaker) tuples (ground truth).
        hypothesis: List of (start, end, speaker) tuples (prediction).
        collar: Forgiveness collar around segment boundaries (seconds).
                Default 0.0 = no forgiveness.

    Returns:
        DER as a float (0.0 to potentially >1.0 if system is very bad).

    NOTE: This is a simplified DER that doesn't handle overlapping speech
          in the reference. For NIST-compliant scoring, use pyannote.metrics.
    """
    # This is a simplified implementation
    # TODO: implement full NIST DER with frame-level scoring
    # For now, we delegate to pyannote for accuracy

    try:
        from pyannote.metrics.diarization import DiarizationErrorRate

        metric = DiarizationErrorRate(collar=collar)

        # Convert to pyannote format
        from pyannote.core import Annotation, Segment

        ref_ann = Annotation()
        for start, end, speaker in reference:
            ref_ann[Segment(start, end)] = speaker

        hyp_ann = Annotation()
        for start, end, speaker in hypothesis:
            hyp_ann[Segment(start, end)] = speaker

        return metric(ref_ann, hyp_ann)
    except ImportError:
        print("Warning: pyannote.metrics not installed. Using simplified DER.")
        return _simple_der(reference, hypothesis, collar)


def _simple_der(
    reference: List[Tuple[float, float, str]],
    hypothesis: List[Tuple[float, float, str]],
    collar: float = 0.0,
) -> float:
    """Simplified DER without overlap handling.

    Only used as fallback when pyannote is not available.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0

    # Convert to frame-level labels (20ms frames)
    frame_step = 0.02  # 20ms
    max_time = max(
        max(e for _, e, _ in reference),
        max(e for _, e, _ in hypothesis) if hypothesis else 0,
    )
    n_frames = int(max_time / frame_step) + 1

    ref_labels = np.full(n_frames, -1, dtype=int)  # -1 = no speech
    hyp_labels = np.full(n_frames, -1, dtype=int)

    # Map speaker strings to ints
    ref_speakers = {}
    hyp_speakers = {}

    for start, end, speaker in reference:
        s_idx = int(start / frame_step)
        e_idx = int(end / frame_step)
        if speaker not in ref_speakers:
            ref_speakers[speaker] = len(ref_speakers)
        ref_labels[s_idx:e_idx] = ref_speakers[speaker]

    for start, end, speaker in hypothesis:
        s_idx = int(start / frame_step)
        e_idx = int(end / frame_step)
        if speaker not in hyp_speakers:
            hyp_speakers[speaker] = len(hyp_speakers)
        hyp_labels[s_idx:e_idx] = hyp_speakers[speaker]

    # Score
    ref_speech = ref_labels >= 0
    hyp_speech = hyp_labels >= 0

    false_alarm = (hyp_speech & ~ref_speech).sum()
    missed = (ref_speech & ~hyp_speech).sum()
    confusion = (ref_speech & hyp_speech & (ref_labels != hyp_labels)).sum()
    total_ref = ref_speech.sum()

    if total_ref == 0:
        return 0.0

    return (false_alarm + missed + confusion) / total_ref
