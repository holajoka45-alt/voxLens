#!/bin/bash
# Benchmark GPU diarization performance
# Usage: bash scripts/benchmark_gpu_diarization.sh [audio_file]

set -euo pipefail

AUDIO="${1:-test_meeting.wav}"

echo "=== VoxLens GPU Benchmark ==="
echo "Audio: $AUDIO"
echo ""

python3 -c "
import time
import torch
from voxlens.core import DiarizationPipeline

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

print()
print('Loading pipeline...')
pipeline = DiarizationPipeline.from_pretrained('voxlens/base')

print('Running diarization...')
times = []
for i in range(3):
    t0 = time.perf_counter()
    result = pipeline.diarize('$AUDIO')
    t1 = time.perf_counter()
    times.append(t1 - t0)
    print(f'  Run {i+1}: {t1-t0:.2f}s ({result.n_speakers_estimated} speakers, {len(result.segments)} segments)')

print()
print(f'Average: {sum(times)/len(times):.2f}s')
print(f'Real-time factor: {result.audio_duration_s / (sum(times)/len(times)):.1f}x')
"
