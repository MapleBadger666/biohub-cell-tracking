
# ============================================================
# Notebook 14 / 14.15C-2
# Build frozen_v9 policies from the two verified Kaggle
# prediction sets.
#
# Uses test Zarr metadata for voxel scale.
# Does not read train GT or invoke the evaluator.
# Does not modify or save any GEFF.
# ============================================================

import json
import sys
from pathlib import Path

assert sys.platform == "linux"

ROOT = Path(
    "/kaggle/input/datasets/markdd520/"
    "biohub-frozen-v9-warm2-bundle/bundle"
)

DATA_DIR = Path(
    "/kaggle/input/competitions/"
    "biohub-cell-tracking-during-development/test"
)

WORK_ROOT = Path("/kaggle/working/biohub_1415")

SPLITS = WORK_ROOT / "test_splits_1415.json"

PRED_995 = (
    WORK_ROOT / "predictions/kaggle_1415/"
    "warm2_det0995_v1/split_0"
)

PRED_9975 = (
    WORK_ROOT / "predictions/kaggle_1415/"
    "warm2_det09975_v1/split_0"
)

assert ROOT.is_dir()
assert DATA_DIR.is_dir()
assert SPLITS.is_file()
assert PRED_995.is_dir()
assert PRED_9975.is_dir()

sys.dont_write_bytecode = True

paths = [
    ROOT / "external/official/scripts",
    ROOT / "external/official/src",
    ROOT / "src",
    ROOT / "scripts",
]

for path in reversed(paths):
    assert path.is_dir(), path
    value = str(path)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from biohub_cell_tracking import frozen_v9 as frozen

assert Path(frozen.__file__).resolve().is_relative_to(
    ROOT.resolve()
)

dataset_names = json.loads(
    SPLITS.read_text(encoding="utf-8")
)[0]["test"]

actual_names = sorted(
    p.name
    for p in DATA_DIR.iterdir()
    if p.is_dir() and p.name.endswith(".zarr")
)

assert dataset_names == actual_names, (
    "当前测试目录与已冻结的动态测试列表不一致"
)

assert dataset_names
assert len(dataset_names) == len(set(dataset_names))

for name in dataset_names:
    assert (DATA_DIR / name).is_dir(), name
    assert (PRED_995 / f"{name}.geff").is_dir(), name
    assert (PRED_9975 / f"{name}.geff").is_dir(), name

# Explicitly preserve the source-defined prefix behavior.
prefix_map = {
    name: name.split("_")[0]
    for name in dataset_names
}

cfg = frozen.FROZEN_V9_CONFIG

assert cfg.det_threshold == 0.995
assert cfg.high_threshold == 0.9975
assert cfg.confidence_component_limit == 9
assert tuple(cfg.model_downsample_zyx) == (1.0, 4.0, 4.0)

print(
    "========== 14.15C-2 / FROZEN POLICY INPUTS ==========",
    flush=True,
)

print("DATA_DIR =", DATA_DIR, flush=True)
print("PRED_995 =", PRED_995, flush=True)
print("PRED_9975 =", PRED_9975, flush=True)
print("DATASET_COUNT =", len(dataset_names), flush=True)
print("PREFIX_MAP =", prefix_map, flush=True)
print("CONFIG =", cfg, flush=True)
print("GT_DIR_PROVIDED = NO", flush=True)

# Keep the returned object in notebook memory for 14.15C-3.
policies_1415 = frozen.build_frozen_v9_policies(
    dataset_names=dataset_names,
    pred_995_dir=PRED_995,
    pred_9975_dir=PRED_9975,
    train_dir=DATA_DIR,
    prefix_map=prefix_map,
    config=cfg,
    progress=True,
)

print(
    "\n========== 14.15C-2 / POLICY SUMMARY ==========",
    flush=True,
)

tables = {
    "gap_candidates": policies_1415.gap_candidates,
    "single_policy": policies_1415.single_policy,
    "isolated_candidates": policies_1415.isolated_candidates,
    "isolated_scope": policies_1415.isolated_scope,
    "strict_two_policy": policies_1415.strict_two_policy,
}

for name, frame in tables.items():
    print(
        name,
        "ROWS =", len(frame),
        "COLUMNS =", len(frame.columns),
        flush=True,
    )

    if "dataset" in frame.columns:
        print(
            "  PER_DATASET =",
            frame.groupby("dataset").size().to_dict(),
            flush=True,
        )

assert policies_1415.config == cfg

print("\n14.15C2_FROZEN_POLICIES_BUILT")
print("TRACKED_GEFFS_SAVED=0")
print("INPUT_GEFFS_MODIFIED=NO")
print("GT_READ=NO")
print("EVALUATOR_RUN=NO")