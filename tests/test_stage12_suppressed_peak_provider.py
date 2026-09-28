\
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import biohub_cell_tracking.stage12_suppressed_peak_provider as sp


def test_frozen_suppressed_peak_source_contract():
    assert (
        sp.SUPPRESSED_PEAK_PROVIDER_ID
        ==
        "SUPPRESSED_PEAK_SOURCE_DERIVED_PROVIDER_V1"
    )

    assert (
        sp.SUPPRESSED_PEAK_CANONICAL_WRITER_SHA256
        ==
        "35843437ccc783cb07751efbb9d6d241b411a3c22cc6bb294928e2c62c22f3fc"
    )

    assert (
        sp.SUPPRESSED_PEAK_EXTRACT_PEAKS_SHA256
        ==
        "56cc697bc41121124e0ae8c634860f9205c7591d5e69a4edd22e0c41727fd47a"
    )


def test_source_derived_transform_removes_historical_p995_runtime_boundary():
    audit = (
        sp.source_derived_transform_audit()
    )

    assert (
        audit[
            "P995_graph_replacements"
        ]
        ==
        1
    )

    assert (
        audit[
            "remaining_P995_runtime_loads"
        ]
        ==
        0
    )

    assert (
        audit[
            "graph_attributes"
        ]
        ==
        [
            "node_attrs"
        ]
    )

    assert (
        audit[
            "universe_line"
        ]
        ==
        810
    )

    removed = set(
        audit[
            "removed_controlled_assignments"
        ]
    )

    assert {
        "PROJECT",
        "TRAIN_ROOT",
        "P995",
        "CKPT",
        "OUT_DIR",
        "CV_PATH",
        "cv",
        "corp",
        "names",
        "fold_map",
    } <= removed


def test_fresh_p995_node_graph_exposes_only_node_attribute_table():
    array = np.array(
        [
            [
                0,
                1,
                2,
                3,
            ],
            [
                1,
                4,
                5,
                6,
            ],
        ],
        dtype=np.int64,
    )

    graph = (
        sp._FreshP995NodeGraph(
            array
        )
    )

    observed = (
        graph
        .node_attrs(
            attr_keys=[
                "t",
                "z",
                "y",
                "x",
            ]
        )
        .to_pandas()
    )

    expected = pd.DataFrame(
        array,
        columns=[
            "t",
            "z",
            "y",
            "x",
        ],
    )

    pd.testing.assert_frame_equal(
        observed,
        expected,
    )

    assert not hasattr(
        graph,
        "edges"
    )


def test_provider_rejects_empty_dataset_before_runtime(monkeypatch):
    class Dummy:
        pass

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        sp.suppressed_peak_provider(
            "   ",
            context=Dummy(),
        )


def test_provider_dataset_boundary_wrapper(monkeypatch, tmp_path):
    data_root = (
        tmp_path
        /
        "data"
    )

    data_root.mkdir()

    (
        data_root
        /
        "demo.zarr"
    ).mkdir()

    weights = (
        tmp_path
        /
        "model.pth"
    )

    weights.write_bytes(
        b"x"
    )

    class P995Context:
        def __init__(self):
            self.data_root = data_root
            self.weights_path = weights

    context = (
        sp.SuppressedPeakProviderContext(
            data_root=data_root,
            weights_path=weights,
            p995_context=P995Context(),
            source_fold_label=0,
        )
    )

    called = {}

    def fake_execute(
        dataset_id,
        inner_context,
    ):
        called[
            "dataset_id"
        ] = dataset_id

        assert (
            inner_context
            is
            context
        )

        return pd.DataFrame(
            {
                "dataset":
                    [
                        dataset_id
                    ],

                "detector_prob":
                    [
                        0.9
                    ],
            }
        )

    monkeypatch.setattr(
        sp,
        "_execute_source_derived",
        fake_execute,
    )

    result = (
        sp.suppressed_peak_provider(
            "demo",
            context=context,
        )
    )

    assert (
        called[
            "dataset_id"
        ]
        ==
        "demo"
    )

    assert (
        result[
            "dataset"
        ].tolist()
        ==
        [
            "demo"
        ]
    )

# === R3R8_R2R1_ENUMERATION_BOUNDARY_TEST ===

def test_r3r8_r2r1_exact_historical_enumeration_boundary_removed():
    audit = (
        sp.source_derived_transform_audit()
    )

    assert (
        audit[
            "historical_enumeration_assertions_removed"
        ]
        ==
        1
    )

    assert (
        audit[
            "historical_enumeration_assertion_line"
        ]
        ==
        169
    )

    assert (
        audit[
            "historical_enumeration_assertion_sha256"
        ]
        ==
        'f7e90f6be28ace0e5b33ed34e0afc176db5ac21c3ee044e2078511c96f2a6257'
    )

    assert (
        audit[
            "P995_graph_replacements"
        ]
        ==
        1
    )

    assert (
        audit[
            "remaining_P995_runtime_loads"
        ]
        ==
        0
    )


# === NB14_14_9C_MISSING_DATASET_COLUMN_BOUNDARY_TEST ===

def _controlled_provider_context(tmp_path):
    data_root = (
        tmp_path
        /
        "data"
    )

    data_root.mkdir()

    weights = (
        tmp_path
        /
        "model.pth"
    )

    weights.write_bytes(
        b"x"
    )

    class P995Context:
        def __init__(self):
            self.data_root = data_root
            self.weights_path = weights

    return sp.SuppressedPeakProviderContext(
        data_root=data_root,
        weights_path=weights,
        p995_context=P995Context(),
        source_fold_label=0,
    )


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(
            {
                "detector_prob":
                    [
                        0.9
                    ],
            }
        ),
        pd.DataFrame(
            {
                "detector_prob":
                    pd.Series(
                        [],
                        dtype=float,
                    ),
            }
        ),
    ],
    ids=[
        "nonempty_missing_dataset",
        "empty_missing_dataset",
    ],
)
def test_provider_rejects_output_missing_dataset_column(
    monkeypatch,
    tmp_path,
    frame,
):
    context = (
        _controlled_provider_context(
            tmp_path
        )
    )

    def fake_execute(
        dataset_id,
        inner_context,
    ):
        return frame

    monkeypatch.setattr(
        sp,
        "_execute_source_derived",
        fake_execute,
    )

    with pytest.raises(
        RuntimeError,
        match="missing required dataset column",
    ):
        sp.suppressed_peak_provider(
            "demo",
            context=context,
        )


# === NB14_14_9D_CROSS_DATASET_BOUNDARY_TEST ===

@pytest.mark.parametrize(
    "datasets",
    [
        [
            "other",
        ],
        [
            "demo",
            "other",
        ],
    ],
    ids=[
        "foreign_only",
        "mixed_target_and_foreign",
    ],
)
def test_provider_rejects_output_from_another_dataset(
    monkeypatch,
    tmp_path,
    datasets,
):
    context = (
        _controlled_provider_context(
            tmp_path
        )
    )

    frame = pd.DataFrame(
        {
            "dataset":
                datasets,

            "detector_prob":
                [
                    0.9
                ]
                *
                len(
                    datasets
                ),
        }
    )

    def fake_execute(
        dataset_id,
        inner_context,
    ):
        return frame

    monkeypatch.setattr(
        sp,
        "_execute_source_derived",
        fake_execute,
    )

    with pytest.raises(
        RuntimeError,
        match="leaked rows from another dataset",
    ):
        sp.suppressed_peak_provider(
            "demo",
            context=context,
        )
