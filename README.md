# VoxLens

> Speaker diarization fine-tuning and evaluation toolkit. Figure out who spoke when — on your own data, with your own models.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/holajoka45-alt/voxLens/actions)

**Status: alpha. Under active development. Expect sharp edges. Not production-ready.**

## What is this?

VoxLens is a toolkit that takes an audio file and answers: **"who spoke when?"**

It wraps the speaker diarization stack — voice activity detection, speaker embedding extraction, and clustering — into a single pipeline that you can configure, fine-tune, benchmark, and hack on.

Think of it as "the diarization equivalent of HuggingFace's `pipeline` + `Trainer`" but specialized, opinionated, and rough around the corners.

## What this is NOT

- A production speech-to-text system (use WhisperX for that)
- A real-time diarization service
- A commercial speaker recognition product
- A GUI tool for labeling speakers

## Quick Start

### Install
