"""Unit tests for scripts/stage11_multiframe_utils.py.

Synthetic, unit-scale only: no checkpoint is loaded, no model runs, and no
competition data is read. The long extraction lives in
scripts/run_stage11_multiframe_extract.py.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stage11_multiframe_utils import (
    EXPECTED_FEATURE_DIMS,
    MATCH_RADIUS_UM,
    OUTPUT_COLUMNS,
    SCALE_UM,
    DatasetSelection,
    candidate_geometry,
    canonicalize_det_logits,
    division_frame_times,
    eligible_label,
    gt_child_degrees,
    iter_window_batches,
    match_gt_to_pred,
    score_features,
    select_datasets,
    sha256_file,
    top2_by_raw_logit,
    validate_schema,
    verify_checkpoint,
)

DOWNSAMPLE = np.array([1.0, 4.0, 4.0], dtype=np.float32)
UNIT_SCALE = np.array([1.0, 1.0, 1.0], dtype=np.float32)


# ============================================================
# Constants and schema
# ============================================================


def test_expected_feature_dimensions_match_the_contract() -> None:
    assert EXPECTED_FEATURE_DIMS["parent_q"] == 128
    assert EXPECTED_FEATURE_DIMS["k_mean"] == 128
    assert EXPECTED_FEATURE_DIMS["k_absdiff"] == 128
    assert EXPECTED_FEATURE_DIMS["score_features"] == 6
    assert EXPECTED_FEATURE_DIMS["geometry"] == 5


def test_output_columns_are_exactly_the_contracted_schema() -> None:
    assert OUTPUT_COLUMNS == (
        "dataset",
        "fold",
        "t",
        "node_id",
        "label",
        "division_frame",
        "match_distance_um",
        "parent_q",
        "k_mean",
        "k_absdiff",
        "score_features",
        "geometry",
        "top1_xyz",
        "top2_xyz",
    )


def test_stage11_voxel_scale_is_the_frozen_constant() -> None:
    assert SCALE_UM.tolist() == [1.625, 0.40625, 0.40625]
    assert MATCH_RADIUS_UM == 7.0


def valid_row(**overrides) -> dict:
    row = {
        "dataset": "6bba_x",
        "fold": 0,
        "t": 5,
        "node_id": 42,
        "label": 1,
        "division_frame": True,
        "match_distance_um": 1.25,
        "parent_q": np.zeros(128, dtype=np.float32),
        "k_mean": np.zeros(128, dtype=np.float32),
        "k_absdiff": np.zeros(128, dtype=np.float32),
        "score_features": np.zeros(6, dtype=np.float32),
        "geometry": np.zeros(5, dtype=np.float32),
        "top1_xyz": np.zeros(3, dtype=np.float32),
        "top2_xyz": np.zeros(3, dtype=np.float32),
    }
    row.update(overrides)
    return row


def test_schema_validation_accepts_a_well_formed_frame() -> None:
    validate_schema(pd.DataFrame([valid_row()], columns=list(OUTPUT_COLUMNS)))


def test_schema_validation_accepts_an_empty_frame() -> None:
    validate_schema(pd.DataFrame(columns=list(OUTPUT_COLUMNS)))


def test_schema_validation_rejects_a_wrong_feature_width() -> None:
    frame = pd.DataFrame([valid_row(parent_q=np.zeros(64, dtype=np.float32))], columns=list(OUTPUT_COLUMNS))

    with pytest.raises(ValueError, match="parent_q must have width 128"):
        validate_schema(frame)


def test_schema_validation_rejects_a_wrong_dtype() -> None:
    frame = pd.DataFrame([valid_row(geometry=np.zeros(5, dtype=np.float64))], columns=list(OUTPUT_COLUMNS))

    with pytest.raises(TypeError, match="geometry must be float32"):
        validate_schema(frame)


def test_schema_validation_rejects_a_reordered_frame() -> None:
    columns = list(OUTPUT_COLUMNS)
    columns[0], columns[1] = columns[1], columns[0]
    frame = pd.DataFrame([valid_row()])[columns]

    with pytest.raises(ValueError, match="schema mismatch"):
        validate_schema(frame)


def test_schema_validation_rejects_a_non_binary_label() -> None:
    frame = pd.DataFrame([valid_row(label=2)], columns=list(OUTPUT_COLUMNS))

    with pytest.raises(ValueError, match="label must be 0 or 1"):
        validate_schema(frame)


# ============================================================
# Top-2 candidate ordering (raw logits, never GT)
# ============================================================


def test_top2_follows_descending_raw_logits() -> None:
    assert top2_by_raw_logit(np.array([0.1, 3.0, -2.0, 2.5])) == (1, 3)


def test_top2_ignores_probability_ordering() -> None:
    # A column softmax would reorder these; the contract says raw logits decide.
    raw = np.array([-5.0, -4.0, -9.0])

    assert top2_by_raw_logit(raw) == (1, 0)


def test_top2_breaks_ties_by_ascending_target_index() -> None:
    assert top2_by_raw_logit(np.array([2.0, 2.0, 2.0])) == (0, 1)


def test_top2_is_deterministic_across_repeated_calls() -> None:
    raw = np.array([1.0, 1.0, 0.5, 1.0])

    assert {top2_by_raw_logit(raw) for _ in range(20)} == {(0, 1)}


def test_top2_requires_at_least_two_candidates() -> None:
    with pytest.raises(ValueError, match="at least two candidate targets"):
        top2_by_raw_logit(np.array([1.0]))


# ============================================================
# score_features ordering
# ============================================================


def test_score_features_preserve_the_11_09a_ordering() -> None:
    features = score_features(raw1=3.0, raw2=1.0, prob1=0.7, prob2=0.2)

    assert features.tolist() == pytest.approx([3.0, 1.0, 2.0, 0.7, 0.2, 0.5])
    assert features.dtype == np.float32
    assert len(features) == EXPECTED_FEATURE_DIMS["score_features"]


def test_score_feature_differences_can_be_negative() -> None:
    features = score_features(raw1=-1.0, raw2=2.0, prob1=0.1, prob2=0.6)

    assert features[2] == pytest.approx(-3.0)
    assert features[5] == pytest.approx(-0.5)


# ============================================================
# Geometry
# ============================================================


def test_geometry_ordering_is_d1_d2_sep_angle_balance() -> None:
    # Daughters opposite the parent along y on the downsampled grid.
    geometry = candidate_geometry(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        DOWNSAMPLE,
        UNIT_SCALE,
    )

    assert len(geometry) == EXPECTED_FEATURE_DIMS["geometry"]
    assert geometry.dtype == np.float32
    assert geometry[0] == pytest.approx(4.0)
    assert geometry[1] == pytest.approx(4.0)
    assert geometry[2] == pytest.approx(8.0)
    assert geometry[3] == pytest.approx(180.0)
    assert geometry[4] == pytest.approx(0.0, abs=1e-6)


def test_geometry_measures_a_right_angle_split() -> None:
    geometry = candidate_geometry(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        DOWNSAMPLE,
        UNIT_SCALE,
    )

    assert geometry[3] == pytest.approx(90.0)
    assert geometry[2] == pytest.approx(4.0 * np.sqrt(2.0), rel=1e-5)


def test_geometry_applies_the_downsample_then_the_physical_scale() -> None:
    # One downsampled y step is 4 voxels, and one z step is 1 voxel; with the
    # Stage 11 anisotropic scale the z arm is 1.625 um and the y arm 1.625 um.
    along_z = candidate_geometry(
        np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), DOWNSAMPLE, SCALE_UM
    )
    along_y = candidate_geometry(
        np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]), DOWNSAMPLE, SCALE_UM
    )

    assert along_z[0] == pytest.approx(1.625)
    assert along_y[0] == pytest.approx(4 * 0.40625)
    assert along_z[0] == pytest.approx(along_y[0])


def test_geometry_balance_is_normalised_arm_asymmetry() -> None:
    geometry = candidate_geometry(
        np.zeros(3),
        np.array([0.0, 7.0, 0.0]),
        np.array([0.0, -2.0, 0.0]),
        DOWNSAMPLE,
        UNIT_SCALE,
    )

    assert geometry[0] == pytest.approx(28.0)
    assert geometry[1] == pytest.approx(8.0)
    assert geometry[4] == pytest.approx(20.0 / 36.0, rel=1e-5)


def test_geometry_degenerate_zero_arm_reports_zero_angle() -> None:
    geometry = candidate_geometry(np.zeros(3), np.zeros(3), np.array([0.0, 1.0, 0.0]), DOWNSAMPLE, UNIT_SCALE)

    assert geometry[0] == pytest.approx(0.0)
    assert geometry[3] == pytest.approx(0.0)


# ============================================================
# Label eligibility
# ============================================================


@pytest.mark.parametrize(
    ("degree", "expected"),
    [(0, None), (1, 0), (2, 1), (3, None), (4, None)],
)
def test_label_eligibility_rules(degree, expected) -> None:
    assert eligible_label(degree) == expected


def test_outdegree_zero_is_excluded_rather_than_a_negative() -> None:
    # An annotation that simply ends is not evidence of continuation.
    assert eligible_label(0) is None
    assert eligible_label(1) == 0


# ============================================================
# GT child degrees / division-frame metadata
# ============================================================


def gt_frames():
    """Parent 1 divides at t=0; parent 2 continues; parent 3 ends at t=1."""
    nodes = pd.DataFrame(
        {
            "node_id": [1, 2, 10, 11, 20, 3],
            "t": [0, 0, 1, 1, 1, 1],
            "z": [0.0, 5.0, 0.0, 0.0, 5.0, 9.0],
            "y": [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            "x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    edges = pd.DataFrame(
        {
            "source_id": [1, 1, 2],
            "target_id": [10, 11, 20],
        }
    )
    return nodes, edges


def test_gt_child_degrees_counts_consecutive_children() -> None:
    degrees = gt_child_degrees(*gt_frames())

    assert int(degrees.loc[1]) == 2
    assert int(degrees.loc[2]) == 1
    assert 3 not in degrees.index


def test_gt_child_degrees_ignores_non_consecutive_edges() -> None:
    nodes = pd.DataFrame({"node_id": [1, 2], "t": [0, 5], "z": [0.0, 0.0], "y": [0.0, 0.0], "x": [0.0, 0.0]})
    edges = pd.DataFrame({"source_id": [1], "target_id": [2]})

    assert len(gt_child_degrees(nodes, edges)) == 0


def test_gt_child_degrees_on_an_edgeless_graph_is_empty() -> None:
    nodes, _ = gt_frames()

    assert len(gt_child_degrees(nodes, pd.DataFrame(columns=["source_id", "target_id"]))) == 0


def test_division_frame_times_mark_only_frames_with_an_eligible_division() -> None:
    nodes, edges = gt_frames()

    assert division_frame_times(nodes, gt_child_degrees(nodes, edges)) == {0}


def test_division_frame_metadata_does_not_filter_ordinary_frames() -> None:
    """A frame with only continuations is still extractable; it is just flagged False."""
    nodes, edges = gt_frames()
    degrees = gt_child_degrees(nodes, edges)
    division_times = division_frame_times(nodes, degrees)

    ordinary_parent_frames = {
        int(nodes.loc[nodes["node_id"] == node_id, "t"].iloc[0])
        for node_id, degree in degrees.items()
        if eligible_label(int(degree)) == 0
    }

    # Ordinary parents exist at t=0, which happens to also carry a division; the
    # metadata flag is per frame and never removes those ordinary rows.
    assert ordinary_parent_frames == {0}
    assert division_times == {0}
    assert eligible_label(int(degrees.loc[2])) == 0


def test_division_frame_times_is_empty_without_divisions() -> None:
    nodes = pd.DataFrame({"node_id": [1, 2], "t": [0, 1], "z": [0.0, 0.0], "y": [0.0, 0.0], "x": [0.0, 0.0]})
    edges = pd.DataFrame({"source_id": [1], "target_id": [2]})

    assert division_frame_times(nodes, gt_child_degrees(nodes, edges)) == set()


# ============================================================
# GT -> detection matching
# ============================================================


def test_matching_pairs_each_gt_node_with_its_nearest_detection() -> None:
    gt = np.array([[0.0, 0.0, 0.0], [0.0, 40.0, 0.0]], dtype=np.float32)
    pred = np.array([[0.0, 40.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    matches = match_gt_to_pred(gt, pred, SCALE_UM)

    assert sorted((gi, pj) for gi, pj, _ in matches) == [(0, 1), (1, 0)]
    assert all(dist == pytest.approx(0.0) for _, _, dist in matches)


def test_matching_drops_partners_beyond_the_seven_micron_radius() -> None:
    gt = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    far = np.array([[0.0, 100.0, 0.0]], dtype=np.float32)

    assert match_gt_to_pred(gt, far, SCALE_UM) == []


def test_matching_is_one_to_one() -> None:
    gt = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    pred = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    matches = match_gt_to_pred(gt, pred, SCALE_UM)

    assert len(matches) == 1
    assert len({pj for _, pj, _ in matches}) == 1


def test_matching_handles_empty_inputs() -> None:
    empty = np.empty((0, 3), dtype=np.float32)
    some = np.zeros((2, 3), dtype=np.float32)

    assert match_gt_to_pred(empty, some) == []
    assert match_gt_to_pred(some, empty) == []


# ============================================================
# Manifest-driven dataset selection
# ============================================================


def manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["6bba_b", "6bba_a", "6bba_c", "44b6_a", "6bba_d"],
            "prefix": ["6bba", "6bba", "6bba", "44b6", "6bba"],
            "fold": [0, 0, 1, 0, 2],
        }
    )


def test_fold_filtering_uses_the_manifest_exactly() -> None:
    selection = select_datasets(manifest(), prefix="6bba", folds=[0])

    assert selection.names == ("6bba_a", "6bba_b")
    assert selection.folds == (0,)
    assert len(selection) == 2


def test_selection_excludes_other_prefixes() -> None:
    selection = select_datasets(manifest(), prefix="6bba", folds=[0, 1, 2])

    assert "44b6_a" not in selection.names
    assert len(selection) == 4


def test_selection_is_sorted_and_deterministic() -> None:
    first = select_datasets(manifest(), prefix="6bba", folds=[0, 1, 2])
    second = select_datasets(manifest().iloc[::-1].reset_index(drop=True), prefix="6bba", folds=[0, 1, 2])

    assert first.names == second.names == tuple(sorted(first.names))


def test_selection_rejects_an_unknown_fold() -> None:
    with pytest.raises(ValueError, match="fold\\(s\\) not present"):
        select_datasets(manifest(), prefix="6bba", folds=[9])


def test_selection_rejects_an_empty_match() -> None:
    with pytest.raises(ValueError, match="no datasets matched"):
        select_datasets(manifest(), prefix="nope", folds=[0])


def test_selection_rejects_a_manifest_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing column"):
        select_datasets(pd.DataFrame({"dataset": ["a"]}), prefix="6bba", folds=[0])


def test_selection_never_walks_the_data_tree(tmp_path: Path) -> None:
    """Datasets come only from manifest rows, never from directory discovery."""
    train = tmp_path / "train"
    train.mkdir()

    for name in ("6bba_a", "6bba_b"):
        (train / f"{name}.zarr").mkdir()
        (train / f"{name}.geff").mkdir()

    # An on-disk dataset absent from the manifest must never be picked up.
    (train / "6bba_ghost.zarr").mkdir()
    (train / "6bba_ghost.geff").mkdir()

    selection = select_datasets(manifest(), prefix="6bba", folds=[0], data_root=train)

    assert selection.names == ("6bba_a", "6bba_b")
    assert "6bba_ghost" not in selection.names


def test_selection_fails_loudly_on_a_missing_dataset(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    (train / "6bba_a").mkdir()
    (train / "6bba_a.geff").mkdir()

    with pytest.raises(FileNotFoundError, match="missing under"):
        select_datasets(manifest(), prefix="6bba", folds=[0], data_root=train)


def test_dataset_selection_is_a_plain_immutable_record() -> None:
    selection = select_datasets(manifest(), prefix="6bba", folds=[0])

    assert isinstance(selection, DatasetSelection)
    with pytest.raises(dataclasses.FrozenInstanceError):
        selection.prefix = "44b6"


# ============================================================
# Checkpoint verification
# ============================================================


def test_checkpoint_hash_verification_accepts_the_expected_digest(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pth"
    path.write_bytes(b"frozen weights")
    expected = hashlib.sha256(b"frozen weights").hexdigest()

    assert verify_checkpoint(path, expected) == expected
    assert sha256_file(path) == expected


def test_checkpoint_hash_verification_rejects_a_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pth"
    path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_checkpoint(path, "0" * 64)


def test_checkpoint_verification_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        verify_checkpoint(tmp_path / "absent.pth", "0" * 64)


# ============================================================
# Window batching (encode-once caching contract)
# ============================================================


def test_window_batches_are_contiguous_ascending_and_complete() -> None:
    batches = iter_window_batches(9, 4)

    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    assert [t for batch in batches for t in batch] == list(range(9))


def test_window_batching_handles_no_windows() -> None:
    assert iter_window_batches(0, 4) == []


def test_window_batching_rejects_a_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        iter_window_batches(4, 0)


# ============================================================
# Detection-logit canonicalisation
# ============================================================


def test_canonicalize_accepts_a_per_frame_list() -> None:
    import torch

    frames = [torch.zeros(2, 1, 3, 4, 5), torch.ones(2, 1, 3, 4, 5)]

    out = canonicalize_det_logits(frames, 2, 2)

    assert tuple(out.shape) == (2, 2, 1, 3, 4, 5)
    assert float(out[0, 1].mean()) == pytest.approx(1.0)


def test_canonicalize_transposes_frame_first_tensors() -> None:
    import torch

    out = canonicalize_det_logits(torch.zeros(2, 5, 1, 3, 4, 5), 5, 2)

    assert tuple(out.shape) == (5, 2, 1, 3, 4, 5)


def test_canonicalize_rejects_an_unexpected_shape() -> None:
    import torch

    with pytest.raises(RuntimeError, match="Unexpected detection logits shape"):
        canonicalize_det_logits(torch.zeros(3, 3, 3), 2, 2)


# ---------------------------------------------------------------------------
# Regression: real competition train layout
#
# A selected dataset exists as:
#
#     <dataset>.zarr/
#     <dataset>.geff/
#
# There is intentionally NO bare <dataset>/ directory.
# ---------------------------------------------------------------------------

def test_select_datasets_accepts_real_zarr_geff_layout(tmp_path):
    import pandas as pd

    manifest = pd.DataFrame(
        {
            "dataset": ["6bba_demo"],
            "prefix": ["6bba"],
            "fold": [0],
        }
    )

    (tmp_path / "6bba_demo.zarr").mkdir()
    (tmp_path / "6bba_demo.geff").mkdir()

    result = select_datasets(
        manifest,
        prefix="6bba",
        folds=[0],
        data_root=tmp_path,
    )

    assert result.frame["dataset"].tolist() == [
        "6bba_demo"
    ]


def test_select_datasets_requires_real_zarr_layout(tmp_path):
    import pandas as pd
    import pytest

    manifest = pd.DataFrame(
        {
            "dataset": ["6bba_demo"],
            "prefix": ["6bba"],
            "fold": [0],
        }
    )

    # Reproduce the exact layout that triggered the bug:
    # GEFF exists, but Zarr is absent.
    (tmp_path / "6bba_demo.geff").mkdir()

    with pytest.raises(
        FileNotFoundError,
        match="6bba_demo",
    ):
        select_datasets(
            manifest,
            prefix="6bba",
            folds=[0],
            data_root=tmp_path,
        )


def test_select_datasets_requires_geff_with_zarr(tmp_path):
    import pandas as pd
    import pytest

    manifest = pd.DataFrame(
        {
            "dataset": ["6bba_demo"],
            "prefix": ["6bba"],
            "fold": [0],
        }
    )

    # Zarr alone is also insufficient because this extractor
    # requires GT tracks for matching / labels.
    (tmp_path / "6bba_demo.zarr").mkdir()

    with pytest.raises(
        FileNotFoundError,
        match="6bba_demo",
    ):
        select_datasets(
            manifest,
            prefix="6bba",
            folds=[0],
            data_root=tmp_path,
        )

