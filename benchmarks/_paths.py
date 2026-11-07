"""Shared path setup for the benchmark scripts."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("kernels/triton", "kernels/thunderkittens"):
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)

RESULTS_DIR = os.environ.get("FLASHFFN_RESULTS_DIR", os.path.join(ROOT, "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)
