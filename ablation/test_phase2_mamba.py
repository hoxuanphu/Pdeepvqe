"""Smoke tests for Phase 2 Mamba sequence modeling.

Run:
    python ablation/test_phase2_mamba.py
"""

import argparse
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ablation.deepvqe_ablation import (
    DeepVQE_Ablation,
    MambaBottleneck_Ablation,
    StreamDeepVQE_Ablation,
    StreamMambaBottleneck_Ablation,
    convert_ablation_to_stream,
    get_ablation_config,
    stream_sequence,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def test_mamba_bottleneck_parity(atol):
    offline = MambaBottleneck_Ablation(
        input_size=24,
        num_blocks=2,
        hidden_size=32,
        d_state=8,
        d_conv=4,
        expand=2,
    ).eval()
    stream = StreamMambaBottleneck_Ablation(
        input_size=24,
        num_blocks=2,
        hidden_size=32,
        d_state=8,
        d_conv=4,
        expand=2,
    ).eval()
    stream.load_state_dict(offline.state_dict(), strict=True)

    x = torch.randn(2, 3, 7, 8)
    y_offline = offline(x)
    cache = stream.init_cache(x.shape[0], x.device, x.dtype)
    y_stream = []
    for frame_idx in range(x.shape[2]):
        y, cache, _ = stream(x[:, :, frame_idx : frame_idx + 1, :], cache)
        y_stream.append(y)
    y_stream = torch.cat(y_stream, dim=2)

    max_abs_error = (y_offline - y_stream).abs().max().item()
    if y_offline.shape != y_stream.shape:
        raise AssertionError(f"Mamba bottleneck shape mismatch: {y_offline.shape} != {y_stream.shape}")
    if len(cache) != len(stream.cache_names()):
        raise AssertionError(f"Mamba cache length mismatch: {len(cache)} != {len(stream.cache_names())}")
    if max_abs_error > atol:
        raise AssertionError(f"Mamba bottleneck streaming parity failed: {max_abs_error} > {atol}")
    print(f"Mamba bottleneck parity passed max_abs_error={max_abs_error:.6g}")


@torch.no_grad()
def test_deepvqe_mamba_integration(frames, atol):
    cfg = get_ablation_config("Mamba_b2_h384")
    model = DeepVQE_Ablation(**cfg).eval()
    stream = StreamDeepVQE_Ablation(**cfg).eval()
    convert_ablation_to_stream(stream, model, strict=True)

    x = torch.randn(1, 257, frames, 2)
    y_offline = model(x)
    y_stream, cache = stream_sequence(stream, x)

    cache_names = stream.get_cache_names()
    expected_mamba_names = (
        "mamba1_conv_cache",
        "mamba1_ssm_state",
        "mamba2_conv_cache",
        "mamba2_ssm_state",
    )
    for name in expected_mamba_names:
        if name not in cache_names:
            raise AssertionError(f"Missing stream cache name: {name}")
    if len(cache) != len(cache_names):
        raise AssertionError(f"DeepVQE Mamba cache length mismatch: {len(cache)} != {len(cache_names)}")
    if y_offline.shape != y_stream.shape:
        raise AssertionError(f"DeepVQE Mamba shape mismatch: {y_offline.shape} != {y_stream.shape}")

    max_abs_error = (y_offline - y_stream).abs().max().item()
    if max_abs_error > atol:
        raise AssertionError(f"DeepVQE Mamba streaming parity failed: {max_abs_error} > {atol}")
    print(f"DeepVQE Mamba integration passed max_abs_error={max_abs_error:.6g}")


def main():
    parser = argparse.ArgumentParser(description="Verify Phase 2 Mamba sequence modeling")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    seed_everything(args.seed)
    test_mamba_bottleneck_parity(args.atol)
    test_deepvqe_mamba_integration(args.frames, args.atol)


if __name__ == "__main__":
    main()

