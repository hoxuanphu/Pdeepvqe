"""Baseline parity checks for ``DeepVQE_Ablation``.

Run:
    python ablation/test_ablation_baseline.py
"""

import argparse
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from deepvqe import DeepVQE
from ablation.deepvqe_ablation import DeepVQE_Ablation, count_parameters


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Verify DeepVQE_Ablation baseline compatibility")
    parser.add_argument("--freq-bins", type=int, default=257)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    seed_everything(args.seed)

    baseline = DeepVQE().eval()
    ablation = DeepVQE_Ablation.from_config_id("Baseline").eval()

    baseline_keys = list(baseline.state_dict().keys())
    ablation_keys = list(ablation.state_dict().keys())
    if baseline_keys != ablation_keys:
        baseline_only = sorted(set(baseline_keys) - set(ablation_keys))
        ablation_only = sorted(set(ablation_keys) - set(baseline_keys))
        raise AssertionError(
            "State-dict keys differ.\n"
            f"Only in DeepVQE: {baseline_only}\n"
            f"Only in DeepVQE_Ablation: {ablation_only}"
        )

    baseline_params = count_parameters(baseline)
    ablation_params = count_parameters(ablation)
    if baseline_params != ablation_params:
        raise AssertionError(f"Parameter count mismatch: {baseline_params} != {ablation_params}")

    ablation.load_state_dict(baseline.state_dict(), strict=True)

    x = torch.randn(2, args.freq_bins, args.frames, 2)
    y_baseline = baseline(x)
    y_ablation = ablation(x)

    if y_baseline.shape != y_ablation.shape:
        raise AssertionError(f"Output shape mismatch: {tuple(y_baseline.shape)} != {tuple(y_ablation.shape)}")

    max_abs_error = (y_baseline - y_ablation).abs().max().item()
    if max_abs_error > args.atol:
        raise AssertionError(f"Numerical parity failed: max_abs_error={max_abs_error} > {args.atol}")

    print("DeepVQE_Ablation baseline parity passed")
    print(f"params={ablation_params}")
    print(f"output_shape={tuple(y_ablation.shape)}")
    print(f"max_abs_error={max_abs_error:.6g}")


if __name__ == "__main__":
    main()
