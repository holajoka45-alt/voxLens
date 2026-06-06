# VoxLens — GPU Cloud Application Pitch

## What

VoxLens is an open-source speaker diarization toolkit. Given an audio file, it answers "who spoke when?" — using voice activity detection, speaker embedding extraction, and clustering. It's designed for practitioners who need to fine-tune diarization on domain-specific data.

## Why it matters

Speaker diarization powers meeting transcription, call center analytics, podcast indexing, court reporting, and media archiving. Every organization with recorded conversations needs diarization. But most solutions are either black-box commercial SaaS (expensive, privacy-invasive) or research code that's impossible to adapt to new domains.

VoxLens makes diarization hackable — you can swap models, fine-tune on your data, and benchmark on your metrics.

## Current state

- Early-stage (v0.2), open source (Apache 2.0)
- Working end-to-end pipeline (VAD + ECAPA-TDNN + spectral clustering)
- RTTM export, DER/JER metrics
- Experimental fine-tuning support
- Solo indie developer, open to contributors

## Why GPU resources are essential

A 1-hour meeting takes ~8 minutes to diarize on CPU vs ~8 seconds on GPU. Fine-tuning a model on 100 hours of meeting data takes 80 hours on CPU vs 40 minutes on GPU. Without GPU, the toolkit is impractical for anything beyond toy examples.

## What we'd use GPU resources for

1. **Fine-tuning pipeline development** — testing loss functions, data augmentation, and training recipes
2. **Benchmarking** — standardized evaluation across AMI, VoxConverse, CallHome datasets
3. **Synthetic data research** — generating diverse training data for domain adaptation
4. **Overlap detection research** — running sub-segment level models for overlapping speech detection

## Request

- Cloud GPU credits — $500-2000 equivalent
- Or access to 1× A10G/A100 instance for 3-6 months

## Links

- GitHub: github.com/voxlens/voxlens
- License: Apache 2.0
- Maintainer: solo indie developer
