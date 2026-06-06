"""GPU utilities for VoxLens."""

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available torch device.

    Args:
        prefer_gpu: If True, use GPU when available.

    Returns:
        torch.device.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_gpu_memory_info() -> dict:
    """Get GPU memory info for diagnostics.

    Returns:
        Dict with memory stats, or empty dict if no GPU.
    """
    if not torch.cuda.is_available():
        return {}

    total = torch.cuda.get_device_properties(0).total_mem
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)

    return {
        "total_gb": total / 1e9,
        "allocated_gb": allocated / 1e9,
        "reserved_gb": reserved / 1e9,
        "free_gb": (total - reserved) / 1e9,
    }


def estimate_batch_size(
    model: torch.nn.Module,
    max_duration_s: float = 10.0,
    sample_rate: int = 16000,
    safety_factor: float = 0.6,
) -> int:
    """Estimate max batch size for embedding extraction.

    Runs forward passes with increasing batch sizes until close to OOM.

    Args:
        model: SpeakerEmbedding model.
        max_duration_s: Max segment duration to simulate.
        sample_rate: Audio sample rate.
        safety_factor: Fraction of VRAM to use.

    Returns:
        Estimated safe batch size.

    NOTE: This is slow. Run once per model, not per audio file.
    """
    if not torch.cuda.is_available():
        return 4  # conservative CPU batch

    total_vram = torch.cuda.get_device_properties(0).total_mem
    target_bytes = int(total_vram * safety_factor)

    samples = int(max_duration_s * sample_rate)

    batch_size = 1
    while True:
        try:
            x = torch.randn(batch_size, 80, 100, device="cuda")
            model.to("cuda")
            _ = model(x)
            torch.cuda.empty_cache()

            if torch.cuda.max_memory_allocated() > target_bytes:
                return max(1, batch_size - 1)

            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return max(1, batch_size // 2)
            raise
