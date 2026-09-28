
# ============================================================
# Notebook 14 / 14.15B-6
# Full Kaggle CUDA inference: Warm2 / det_threshold=0.9975
#
# Preserves existing det=0.995 predictions.
# No GT, no evaluator, no ILP, no frame limit.
# ============================================================

import sys
import json
import hashlib
from pathlib import Path

import torch

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

USERNAME = "kaggle_1415"
METHOD = "warm2_det09975_v1"

PREDICTIONS_ROOT = WORK_ROOT / "predictions"

OUTPUT_DIR = (
    PREDICTIONS_ROOT
    / USERNAME
    / METHOD
    / "split_0"
)

PREVIOUS_OUTPUT_DIR = (
    PREDICTIONS_ROOT
    / USERNAME
    / "warm2_det0995_v1"
    / "split_0"
)

WEIGHTS = (
    ROOT
    / "models/official_baseline/"
      "official_baseline_fold0_warm2_lr1e5/"
      "split_0/edge_predictor_best.pth"
)

EXPECTED_WEIGHTS_SHA = (
    "31d209bfde1fd5bbf3b2a661cdb4a25"
    "bb72922b832d315ee5a75cc32f44656c1"
)

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()

# ------------------------------------------------------------
# 1. Source, data, checkpoint and output safety
# ------------------------------------------------------------

assert sys.platform == "linux"
assert Path("/kaggle/working").is_dir()

assert ROOT.is_dir()
assert DATA_DIR.is_dir()
assert SPLITS.is_file()
assert WEIGHTS.is_file()

assert sha256_file(WEIGHTS) == EXPECTED_WEIGHTS_SHA, (
    "Checkpoint SHA 不匹配，停止"
)

assert OUTPUT_DIR.resolve().is_relative_to(
    Path("/kaggle/working").resolve()
)

assert OUTPUT_DIR != PREVIOUS_OUTPUT_DIR

# predict() removes existing *.geff in its selected directory.
# Never rerun into an existing output directory.
assert not OUTPUT_DIR.exists(), (
    f"新输出目录已存在，禁止覆盖：{OUTPUT_DIR}"
)

assert PREVIOUS_OUTPUT_DIR.is_dir(), (
    "此前的 0.995 输出目录不存在"
)

test_names = json.loads(
    SPLITS.read_text(encoding="utf-8")
)[0]["test"]

actual_names = sorted(
    p.name
    for p in DATA_DIR.iterdir()
    if p.is_dir() and p.name.endswith(".zarr")
)

assert test_names == actual_names, (
    "动态测试列表与当前数据不一致"
)

expected_geffs = sorted(
    f"{name}.geff" for name in test_names
)

previous_geffs = sorted(
    p.name
    for p in PREVIOUS_OUTPUT_DIR.glob("*.geff")
)

assert previous_geffs == expected_geffs, (
    "此前的 0.995 GEFF 列表不完整"
)

assert torch.cuda.is_available(), (
    "CUDA 不可用，禁止静默回退 CPU"
)

# ------------------------------------------------------------
# 2. Frozen module import
# ------------------------------------------------------------

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

import predict_unet_transformer_mps as predictor

assert Path(predictor.__file__).resolve().is_relative_to(
    ROOT.resolve()
)

cfg = predictor.PredictConfig(
    det_threshold=0.9975,
    pool_kernel_um=3.0,
    use_ilp=False,
)

assert cfg.det_threshold == 0.9975
assert cfg.pool_kernel_um == 3.0
assert cfg.det_tta is True
assert cfg.edge_activation == "softmax"
assert cfg.use_ilp is False
assert cfg.max_parents_per_node == 1
assert cfg.max_children_per_node == 2

# ------------------------------------------------------------
# 3. CUDA inference
# ------------------------------------------------------------

print(
    "========== 14.15B-6 / PRE-INFERENCE ==========",
    flush=True,
)

print("CUDA_DEVICE =", torch.cuda.get_device_name(0))
print("WEIGHTS_SHA256 =", sha256_file(WEIGHTS))
print("TEST_COUNT =", len(test_names))
print("DET_THRESHOLD =", cfg.det_threshold)
print("POOL_KERNEL_UM =", cfg.pool_kernel_um)
print("USE_ILP =", cfg.use_ilp)
print("OUTPUT_DIR =", OUTPUT_DIR)
print("PREVIOUS_OUTPUT_PRESERVED =", PREVIOUS_OUTPUT_DIR)
print("EVALUATE = False")
print("MAX_FRAMES = None")

original_predictions_path = predictor.PREDICTIONS_PATH
original_username = predictor.USERNAME

try:
    predictor.PREDICTIONS_PATH = PREDICTIONS_ROOT
    predictor.USERNAME = USERNAME

    assert (
        predictor.PREDICTIONS_PATH
        / predictor.USERNAME
        / METHOD
        / "split_0"
    ) == OUTPUT_DIR

    predictor.predict(
        data_dir=DATA_DIR,
        fold=0,
        splits_file=SPLITS,
        weights_path=WEIGHTS,
        cfg=cfg,
        method=METHOD,
        debug_video=None,
        unet_batch_size=4,
        video_slice=None,
        evaluate=False,
        max_frames=None,
    )

finally:
    predictor.PREDICTIONS_PATH = original_predictions_path
    predictor.USERNAME = original_username

# ------------------------------------------------------------
# 4. Output inventory
# ------------------------------------------------------------

print("\n========== OUTPUT CHECK ==========", flush=True)

assert OUTPUT_DIR.is_dir(), "0.9975 输出目录不存在"

actual_geffs = sorted(
    p.name
    for p in OUTPUT_DIR.glob("*.geff")
)

previous_geffs_after = sorted(
    p.name
    for p in PREVIOUS_OUTPUT_DIR.glob("*.geff")
)

print("EXPECTED_GEFFS =", expected_geffs)
print("ACTUAL_GEFFS =", actual_geffs)

assert actual_geffs == expected_geffs, (
    "0.9975 GEFF 列表不完整"
)

assert previous_geffs_after == previous_geffs, (
    "此前的 0.995 GEFF 列表发生变化"
)

print("\n14.15B6_CUDA_INFERENCE_OUTPUTS_PRESENT")
print("PREVIOUS_GEFF_LIST_PRESERVED=YES")
print("GEFF_SEMANTICS_VERIFIED=NO")
print("GT_READ=NO")
print("EVALUATOR_RUN=NO")
print("FROZEN_SOURCE_MODIFIED=NO")