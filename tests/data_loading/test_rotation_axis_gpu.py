"""GPU tests for rotation axis selection using the real DALI library."""

import json
import pathlib
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).with_name("rotation_axis_check.py")
ROOT = pathlib.Path(__file__).resolve().parents[2]


def _run(*args):
    """Run the check script with real DALI; skip if DALI/GPU is unavailable."""
    probe = subprocess.run(
        [sys.executable, "-I", "-c",
         "import nvidia.dali, torch; assert torch.cuda.is_available()"],
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("Requires real DALI and a GPU.")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True, text=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT)},
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("array_axis", [0, 1, 2])
def test_fixed_axis_leaves_array_axis_invariant(array_axis):
    counts = _run("fixed", array_axis)
    assert counts[str(array_axis)] == 1


def test_random_axis_is_uniform_and_never_degenerate():
    counts = _run("random")
    assert counts["-1"] == 0
    assert all(counts[k] > 60 for k in ("0", "1", "2"))


@pytest.mark.parametrize("label_dtype", ["float32", "uint8"])
def test_permute_axes_is_uniform_aligned_and_keeps_dtype(label_dtype):
    result = _run("permute", label_dtype)
    counts = result["counts"]
    assert len(counts) == 6
    assert all(c > 60 for c in counts.values())
    assert result["misaligned"] == 0
    assert result["label_dtype"] == label_dtype
