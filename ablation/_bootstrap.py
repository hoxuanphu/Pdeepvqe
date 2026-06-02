"""Helpers for running ablation scripts directly from the repo root."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_repo_root():
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT
