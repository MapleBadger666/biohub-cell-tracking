
# Notebook 14 / 14.15B-3
# Full Kaggle CUDA inference: Warm2 / det_threshold=0.995
# No ground truth and no evaluator.

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
METHOD = "warm2_det0995_v1"

PREDICTIONS_ROOT = WORK_ROOT / "predictions"

OUTPUT_DIR = (
    PREDICTIONS_ROOT / USERNAME / METHOD / "split_0"
)

WEIGHTS = (
    ROOT
    / "models/official_baseline/"
      "official_baseline_fold0_warm2_lr1e5/"
      "split_0/edge_predictor_best.pth"
)

assert ROOT.is_dir()
assert DATA_DIR.is_dir()
assert SPLITS.is_file()
assert WEIGHTS.is_file()
assert OUTPUT_DIR.resolve().is_relative_to(
    Path("/kaggle/working").resolve()
)

assert not OUTPUT_DIR.exists(), (
    "输出目录已存在，禁止重跑或覆盖。"
)

assert torch.cuda.is_available(), (
    "CUDA 不可用：停止，不允许静默回退到 CPU。"
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
    "测试列表与当前挂载的测试数据不一致。"
)

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
    det_threshold=0.995,
    pool_kernel_um=3.0,
    use_ilp=False,
)

assert cfg.det_threshold == 0.995
assert cfg.pool_kernel_um == 3.0
assert cfg.det_tta is True
assert cfg.edge_activation == "softmax"
assert cfg.use_ilp is False
assert cfg.max_parents_per_node == 1
assert cfg.max_children_per_node == 2

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

print("========== 14.15B-3 / PRE-INFERENCE ==========", flush=True)
print("CUDA_DEVICE =", torch.cuda.get_device_name(0), flush=True)
print("WEIGHTS_SHA256 =", sha256_file(WEIGHTS), flush=True)
print("TEST_COUNT =", len(test_names), flush=True)
print("DET_THRESHOLD =", cfg.det_threshold, flush=True)
print("POOL_KERNEL_UM =", cfg.pool_kernel_um, flush=True)
print("USE_ILP =", cfg.use_ilp, flush=True)
print("OUTPUT_DIR =", OUTPUT_DIR, flush=True)
print("EVALUATE = False", flush=True)
print("MAX_FRAMES = None", flush=True)

# Redirect only runtime output-related module globals.
# The source file and bundled weights remain unchanged.
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

print("\n========== OUTPUT CHECK ==========", flush=True)

assert OUTPUT_DIR.is_dir(), "预测输出目录不存在"

actual_geffs = sorted(
    p.name for p in OUTPUT_DIR.glob("*.geff")
)

expected_geffs = sorted(
    f"{name}.geff" for name in test_names
)

print("EXPECTED_GEFFS =", expected_geffs)
print("ACTUAL_GEFFS =", actual_geffs)

assert actual_geffs == expected_geffs, (
    "输出 GEFF 列表与测试序列不一致"
)

print("\n14.15B3_CUDA_INFERENCE_OUTPUTS_PRESENT")
print("GT_READ=NO")
print("EVALUATOR_RUN=NO")
print("FROZEN_SOURCE_MODIFIED=NO")