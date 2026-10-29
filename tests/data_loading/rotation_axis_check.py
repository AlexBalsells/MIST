"""Run DALI's rotation_fn on coordinate ramps and report invariant axes.

This script is run in a subprocess by test_rotation_axis_gpu.py because the
test suite replaces nvidia.dali with mocks (see tests/conftest.py). Prints a
JSON dict with the invariant-axis counts (keys "0", "1", "2", "-1").
"""
import json
import sys
import types

import numpy as np
from nvidia.dali import fn
from nvidia.dali.pipeline import Pipeline

from mist.data_loading.data_loading_constants import DataLoadingConstants as K
from mist.data_loading.dali_loader import TrainPipeline

SIZE = 32


def rotate(random_axis, fixed_axis, batch_size):
    """Return the ramp volume and the rotated outputs for a batch."""
    coords = np.meshgrid(*[np.arange(SIZE, dtype=np.float32)] * 3,
                         indexing="ij")
    vol = np.stack(coords, axis=-1)
    label = (coords[0] > 8).astype(np.float32)[..., None]
    stub = types.SimpleNamespace(
        roi_size=(SIZE, SIZE, SIZE),
        random_rotation_axis=random_axis,
        fixed_rotation_axis=fixed_axis,
        has_dtms=False,
    )
    pipe = Pipeline(batch_size=batch_size, num_threads=1, device_id=0, seed=1)
    with pipe:
        img = fn.external_source(
            source=lambda: [vol] * batch_size, batch=True, device="gpu",
            layout="DHWC",
        )
        lbl = fn.external_source(
            source=lambda: [label] * batch_size, batch=True, device="gpu",
            layout="DHWC",
        )
        pipe.set_outputs(*TrainPipeline.rotation_fn(stub, img, lbl))
    pipe.build()
    out, _ = pipe.run()
    return vol, [np.array(out.as_cpu()[i]) for i in range(batch_size)]


def invariant_axis(vol, out):
    """Index of the array axis the rotation leaves unchanged, else -1."""
    inner = slice(8, -8)
    change = [
        np.abs(out[inner, inner, inner, k] - vol[inner, inner, inner, k]).mean()
        for k in range(3)
    ]
    axis = int(np.argmin(change))
    return axis if change[axis] < 0.3 else -1


def main():
    K.ROTATION_FN_PROBABILITY = 1.0
    K.ROTATION_FN_RANGE_MIN = 9.99
    K.ROTATION_FN_RANGE_MAX = 10.0
    mode = sys.argv[1]
    if mode == "fixed":
        axis = int(sys.argv[2])
        vol, outs = rotate(False, K.ROTATION_AXIS_BY_ARRAY_AXIS[axis], 1)
    else:
        vol, outs = rotate(True, None, 300)
    counts = {"-1": 0, "0": 0, "1": 0, "2": 0}
    for o in outs:
        counts[str(invariant_axis(vol, o))] += 1
    print(json.dumps(counts))


if __name__ == "__main__" and sys.argv[1] != "permute":
    main()


def permute(batch_size, label_dtype):
    """Run permute_axes_fn on ramps; return counts per permutation."""
    from mist.data_loading.data_loading_utils import permute_axes_fn

    coords = np.meshgrid(*[np.arange(SIZE, dtype=np.float32)] * 3,
                         indexing="ij")
    vol = np.stack(coords, axis=-1)
    label = (coords[0] > 8).astype(label_dtype)[..., None]
    pipe = Pipeline(batch_size=batch_size, num_threads=1, device_id=0, seed=3)
    with pipe:
        img = fn.external_source(
            source=lambda: [vol] * batch_size, batch=True, device="gpu",
            layout="DHWC",
        )
        lbl = fn.external_source(
            source=lambda: [label] * batch_size, batch=True, device="gpu",
            layout="DHWC",
        )
        pipe.set_outputs(*permute_axes_fn(img, lbl))
    pipe.build()
    out_img, out_lbl = pipe.run()
    counts, misaligned = {}, 0
    for i in range(batch_size):
        o_img = np.array(out_img.as_cpu()[i])
        o_lbl = np.array(out_lbl.as_cpu()[i])
        # For each original axis k, the output axis along which channel k
        # varies. This tuple identifies the applied permutation.
        along = tuple(
            int(np.argmax([np.ptp(o_img[..., k], axis=a).max()
                           for a in range(3)]))
            for k in range(3)
        )
        counts[str(along)] = counts.get(str(along), 0) + 1
        misaligned += int(
            (o_lbl[..., 0].astype(bool) != (o_img[..., 0] > 8)).any()
        )
    return {"counts": counts, "misaligned": misaligned,
            "label_dtype": str(o_lbl.dtype)}


if __name__ == "__main__" and sys.argv[1] == "permute":
    print(json.dumps(permute(600, np.dtype(sys.argv[2]).type)))
