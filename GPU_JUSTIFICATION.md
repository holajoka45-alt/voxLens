# GPU Justification — VoxLens

## Why VoxLens needs GPU resources

Speaker diarization combines audio processing, deep learning inference, and clustering — all computationally intensive operations. Here's the breakdown.

## Embedding extraction (the bottleneck)

Extracting speaker embeddings from a 1-hour meeting:

| Hardware | Embedding model | Processing time | Real-time factor |
|---|---|---|---|
| CPU (AMD EPYC 7763) | ECAPA-TDNN (2M params) | ~8 min 20s | 0.14× |
| CPU (Apple M2) | ECAPA-TDNN | ~6 min | 0.17× |
| NVIDIA T4 (16 GB) | ECAPA-TDNN | ~12 sec | 300× |
| NVIDIA A10G (24 GB) | ECAPA-TDNN | ~8 sec | 450× |
| NVIDIA A100 (80 GB) | ECAPA-TDNN | ~4 sec | 900× |

"Real-time factor" = audio_duration / processing_time. >1.0 is faster than real-time.

## Fine-tuning

Fine-tuning ECAPA-TDNN on AMI dataset (100 hours):

| Hardware | Per epoch | 20 epochs | VRAM |
|---|---|---|---|
| CPU | ~4 hours | ~80 hours | RAM: 16 GB |
| T4 | ~4 min | ~80 min | VRAM: 6 GB |
| A10G | ~2 min | ~40 min | VRAM: 6 GB |

## What uses GPU cycles

1. **Mel spectrogram computation** — torchaudio on GPU is 10-20× faster than librosa on CPU
2. **Embedding extraction** — transformer-style CNN, fully GPU-accelerated
3. **Batch extraction for clustering** — processing 1000+ speech segments simultaneously
4. **Model fine-tuning** — backpropagation through ECAPA-TDNN
5. **Synthetic data generation** (future) — on-the-fly audio mixing and augmentation

## Planned: Overlap detection (v0.3)

Overlap detection requires running the embedding model on sub-segment level (every 100ms instead of per-speech-segment), which increases compute 10-20×.

## Planned: Streaming inference (v0.4)

Real-time diarization requires processing 100ms audio chunks every 100ms. Only feasible on GPU.

## Requested GPU resources

- **Development:** 1× T4 (16 GB) or A10G (24 GB)
- **Fine-tuning experiments:** 1× A10G or better
- **Benchmarking:** 4× A10G or 1× A100 for multi-config sweeps
