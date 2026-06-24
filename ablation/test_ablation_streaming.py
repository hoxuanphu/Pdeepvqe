"""Streaming parity checks for ``StreamDeepVQE_Ablation``.

Run:
    python ablation/test_ablation_streaming.py
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
    ABLATION_CONFIGS,
    DeepVQE_Ablation,
    StreamDeepVQE_Ablation,
    convert_ablation_to_stream,
    stream_sequence,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Verify DeepVQE ablation streaming parity")
    parser.add_argument("--configs", nargs="*", default=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--freq-bins", type=int, default=257)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    seed_everything(args.seed)
    for config_id in args.configs:
        offline = DeepVQE_Ablation.from_config_id(config_id).eval()
        stream = StreamDeepVQE_Ablation.from_config_id(config_id).eval()
        convert_ablation_to_stream(stream, offline, strict=True)

        x = torch.randn(1, args.freq_bins, args.frames, 2)
        y_offline = offline(x)
        y_stream, cache = stream_sequence(stream, x)

        if len(cache) != len(stream.get_cache_names()):
            raise AssertionError(f"{config_id}: unexpected cache length {len(cache)}")
        if y_offline.shape != y_stream.shape:
            raise AssertionError(f"{config_id}: shape mismatch {tuple(y_offline.shape)} != {tuple(y_stream.shape)}")

        max_abs_error = (y_offline - y_stream).abs().max().item()
        if max_abs_error > args.atol:
            raise AssertionError(f"{config_id}: streaming parity failed: {max_abs_error} > {args.atol}")

        print(f"{config_id}: streaming parity passed max_abs_error={max_abs_error:.6g}")


if __name__ == "__main__":
    main()
