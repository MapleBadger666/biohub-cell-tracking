from __future__ import annotations

from pathlib import Path

import inspect

import numpy as np
import pytest

import biohub_cell_tracking.stage12_portable_current_best as portable


VALID_SHA = "a" * 64


def _context(provider=None):

    if provider is None:

        def provider(dataset_id, nodes, t, *, context):
            return (
                [1],
                [2],
                np.asarray([[0.9]], dtype=float),
            )

    return portable.PortableCurrentBestContext(
        b256_pred_995_root=Path("/tmp/b256_995"),
        b256_pred_9975_root=Path("/tmp/b256_9975"),
        s9995_pred_995_root=Path("/tmp/s9995_995"),
        s9995_pred_9975_root=Path("/tmp/s9995_9975"),
        dataset_root=Path("/tmp/dataset"),
        physical_scale_zyx_um=(1.625, 0.40625, 0.40625),
        frozen_v9_policy_bundle=object(),
        association_provider=provider,
        association_provider_identity_sha256=VALID_SHA,
        strict=True,
    )


def test_frozen_v1_threshold_is_exact():

    assert portable.MUTUAL_CONT_V1_PROB_MIN == 0.25


def test_context_requires_strict_true():

    with pytest.raises(
        portable.PortableRuntimeContractError
    ):

        portable.PortableCurrentBestContext(
            b256_pred_995_root="/tmp/a",
            b256_pred_9975_root="/tmp/b",
            s9995_pred_995_root="/tmp/c",
            s9995_pred_9975_root="/tmp/d",
            dataset_root="/tmp/e",
            physical_scale_zyx_um=(1.0, 1.0, 1.0),
            frozen_v9_policy_bundle=object(),
            association_provider=lambda *a, **k: None,
            association_provider_identity_sha256=VALID_SHA,
            strict=False,
        )


def test_context_rejects_bad_provider_identity():

    with pytest.raises(
        portable.PortableRuntimeContractError
    ):

        portable.PortableCurrentBestContext(
            b256_pred_995_root="/tmp/a",
            b256_pred_9975_root="/tmp/b",
            s9995_pred_995_root="/tmp/c",
            s9995_pred_9975_root="/tmp/d",
            dataset_root="/tmp/e",
            physical_scale_zyx_um=(1.0, 1.0, 1.0),
            frozen_v9_policy_bundle=object(),
            association_provider=lambda *a, **k: None,
            association_provider_identity_sha256="not-a-sha",
            strict=True,
        )


def test_context_rejects_invalid_scale():

    with pytest.raises(
        portable.PortableRuntimeContractError
    ):

        portable.PortableCurrentBestContext(
            b256_pred_995_root="/tmp/a",
            b256_pred_9975_root="/tmp/b",
            s9995_pred_995_root="/tmp/c",
            s9995_pred_9975_root="/tmp/d",
            dataset_root="/tmp/e",
            physical_scale_zyx_um=(1.0, 0.0, 1.0),
            frozen_v9_policy_bundle=object(),
            association_provider=lambda *a, **k: None,
            association_provider_identity_sha256=VALID_SHA,
            strict=True,
        )


def test_association_provider_contract():

    def provider(dataset_id, nodes, t, *, context):

        assert dataset_id == "dataset-x"
        assert t == 4

        return (
            [10, 11],
            [20, 21],
            np.asarray(
                [
                    [0.9, 0.1],
                    [0.2, 0.8],
                ],
                dtype=float,
            ),
        )

    context = _context(
        provider
    )

    source_ids, target_ids, probs = (
        portable.call_association_provider(
            "dataset-x",
            nodes=None,
            t=4,
            context=context,
        )
    )

    np.testing.assert_array_equal(
        source_ids,
        [10, 11],
    )

    np.testing.assert_array_equal(
        target_ids,
        [20, 21],
    )

    assert probs.shape == (2, 2)


def test_v1_two_reciprocal_actions():

    actions = (
        portable.reconstruct_mutual_cont_v1_actions(
            "d",
            [10, 11],
            [20, 21],
            np.asarray(
                [
                    [0.9, 0.1],
                    [0.2, 0.8],
                ]
            ),
            source_outdegree={},
            target_indegree={},
        )
    )

    assert [
        (
            a.source_node_id,
            a.target_node_id,
        )
        for a in actions
    ] == [
        (10, 20),
        (11, 21),
    ]


def test_v1_requires_reciprocal_top1():

    actions = (
        portable.reconstruct_mutual_cont_v1_actions(
            "d",
            [10, 11],
            [20, 21],
            np.asarray(
                [
                    [0.90, 0.80],
                    [0.70, 0.60],
                ]
            ),
            source_outdegree={},
            target_indegree={},
        )
    )

    assert [
        (
            a.source_node_id,
            a.target_node_id,
        )
        for a in actions
    ] == [
        (10, 20),
    ]


def test_v1_threshold_is_inclusive():

    actions = (
        portable.reconstruct_mutual_cont_v1_actions(
            "d",
            [10],
            [20],
            np.asarray(
                [
                    [0.25],
                ]
            ),
            source_outdegree={},
            target_indegree={},
        )
    )

    assert len(actions) == 1

    assert actions[0].candidate_prob == 0.25


def test_v1_below_threshold_rejected():

    actions = (
        portable.reconstruct_mutual_cont_v1_actions(
            "d",
            [10],
            [20],
            np.asarray(
                [
                    [0.249999999],
                ]
            ),
            source_outdegree={},
            target_indegree={},
        )
    )

    assert actions == tuple()


def test_v1_requires_source_outdegree_zero():

    actions = (
        portable.reconstruct_mutual_cont_v1_actions(
            "d",
            [10],
            [20],
            np.asarray(
                [
                    [0.9],
                ]
            ),
            source_outdegree={
                10: 1,
            },
            target_indegree={},
        )
    )

    assert actions == tuple()


def test_v1_requires_target_indegree_zero():

    actions = (
        portable.reconstruct_mutual_cont_v1_actions(
            "d",
            [10],
            [20],
            np.asarray(
                [
                    [0.9],
                ]
            ),
            source_outdegree={},
            target_indegree={
                20: 1,
            },
        )
    )

    assert actions == tuple()


def test_v1_tie_behavior_is_deterministic_first_occurrence():

    actions = (
        portable.reconstruct_mutual_cont_v1_actions(
            "d",
            [10, 11],
            [20, 21],
            np.asarray(
                [
                    [0.5, 0.5],
                    [0.5, 0.5],
                ]
            ),
            source_outdegree={},
            target_indegree={},
        )
    )

    assert [
        (
            a.source_node_id,
            a.target_node_id,
        )
        for a in actions
    ] == [
        (10, 20),
    ]


def test_empty_association_matrix_returns_no_actions():

    actions = (
        portable.reconstruct_mutual_cont_v1_actions(
            "d",
            [],
            [],
            np.empty(
                (0, 0),
                dtype=float,
            ),
            source_outdegree={},
            target_indegree={},
        )
    )

    assert actions == tuple()


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray([[np.nan]]),
        np.asarray([[np.inf]]),
        np.asarray([[-0.1]]),
        np.asarray([[1.1]]),
    ],
)
def test_invalid_probability_matrix_fails_closed(matrix):

    with pytest.raises(
        portable.PortableRuntimeContractError
    ):

        portable.reconstruct_mutual_cont_v1_actions(
            "d",
            [10],
            [20],
            matrix,
            source_outdegree={},
            target_indegree={},
        )


def test_probability_matrix_shape_must_match_ids():

    with pytest.raises(
        portable.PortableRuntimeContractError
    ):

        portable.reconstruct_mutual_cont_v1_actions(
            "d",
            [10, 11],
            [20],
            np.asarray(
                [
                    [0.9, 0.8],
                ]
            ),
            source_outdegree={},
            target_indegree={},
        )


def test_duplicate_source_ids_fail_closed():

    with pytest.raises(
        portable.PortableRuntimeContractError
    ):

        portable.reconstruct_mutual_cont_v1_actions(
            "d",
            [10, 10],
            [20],
            np.asarray(
                [
                    [0.9],
                    [0.8],
                ]
            ),
            source_outdegree={},
            target_indegree={},
        )


def test_duplicate_target_ids_fail_closed():

    with pytest.raises(
        portable.PortableRuntimeContractError
    ):

        portable.reconstruct_mutual_cont_v1_actions(
            "d",
            [10],
            [20, 20],
            np.asarray(
                [
                    [0.9, 0.8],
                ]
            ),
            source_outdegree={},
            target_indegree={},
        )


def test_public_entrypoint_fails_closed_while_partial():

    context = _context()

    with pytest.raises(
        portable.PortableImplementationIncompleteError
    ):

        portable.build_portable_pre_g3h3_current_best(
            "dataset-x",
            context=context,
        )


def test_no_historical_action_fallback_in_module_source():

    source = inspect.getsource(
        portable
    ).lower()

    forbidden = [
        "stage12_fold0_mutual_cont_v1_actions_locked.pkl",
        "stage12_fold0_mutual_cont_v2_actions_locked.pkl",
        "stage11_fold0_morphdiv_v2_actions_locked.pkl",
        "stage12_fold0_r1p50_confirmation_actions.pkl",
        "cv_manifest",
        "fold0_membership",
        "historical_dataset_allowlist",
    ]

    for token in forbidden:

        assert token not in source


def test_v1_reconstruction_has_no_runtime_threshold_argument():

    signature = inspect.signature(
        portable.reconstruct_mutual_cont_v1_actions
    )

    assert (
        "threshold"
        not in
        signature.parameters
    )


def test_portable_frozen_v9_wiring_uses_explicit_context(monkeypatch):

    captured = {}

    def fake_apply_frozen_v9(
        dataset_name,
        pred_995_dir,
        pred_9975_dir,
        policies,
        train_dir,
    ):

        captured["dataset_name"] = dataset_name
        captured["pred_995_dir"] = pred_995_dir
        captured["pred_9975_dir"] = pred_9975_dir
        captured["policies"] = policies
        captured["train_dir"] = train_dir

        return "frozen-v9-graph"

    monkeypatch.setattr(
        portable._frozen_v9,
        "apply_frozen_v9",
        fake_apply_frozen_v9,
    )

    context = _context()

    result = portable._build_portable_frozen_v9(
        "dataset-x",
        context=context,
    )

    assert result == "frozen-v9-graph"

    assert captured["dataset_name"] == "dataset-x"

    assert captured["pred_995_dir"] == Path(
        "/tmp/b256_995"
    )

    assert captured["pred_9975_dir"] == Path(
        "/tmp/b256_9975"
    )

    assert captured["train_dir"] == Path(
        "/tmp/dataset"
    )

    assert (
        captured["policies"]
        is
        context.frozen_v9_policy_bundle
    )


# BEGIN NOTEBOOK13 13.3D3 MORPH PROVIDER TESTS

import copy

import numpy as np
import pandas as pd
import pytest

from biohub_cell_tracking.stage12_portable_current_best import (
    MORPH_DIV_V2_CANDIDATE_COLUMNS,
    MORPH_DIV_V2_CONFIG_ID,
    MorphProviderContractError,
    apply_portable_morph_div_v2,
    select_morph_div_v2_candidates,
    validate_morph_candidate_provider,
    validate_morph_candidate_table,
)


class _D3Table:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df.copy(
            deep=True
        )


class _D3FakeGraph:
    def __init__(
        self,
        node_ids,
        edges,
    ):
        self._nodes = pd.DataFrame(
            {
                "node_id":
                    list(
                        node_ids
                    )
            }
        )

        self._edges = pd.DataFrame(
            list(
                edges
            ),
            columns=[
                "source",
                "target",
                "edge_prob",
            ],
        )

    def node_attrs(
        self,
        attr_keys=None,
    ):
        if attr_keys is None:
            df = self._nodes
        else:
            df = self._nodes[
                list(
                    attr_keys
                )
            ]

        return _D3Table(
            df
        )

    def edge_attrs(
        self,
        attr_keys=None,
    ):
        if attr_keys is None:
            df = self._edges
        else:
            df = self._edges[
                list(
                    attr_keys
                )
            ]

        return _D3Table(
            df
        )

    def add_edge(
        self,
        *,
        source,
        target,
        edge_prob,
    ):
        row = pd.DataFrame(
            [
                {
                    "source":
                        int(
                            source
                        ),

                    "target":
                        int(
                            target
                        ),

                    "edge_prob":
                        float(
                            edge_prob
                        ),
                }
            ]
        )

        self._edges = pd.concat(
            [
                self._edges,
                row,
            ],
            ignore_index=True,
        )


def _d3_candidates(
    dataset="ds",
):
    return pd.DataFrame(
        [
            {
                "dataset":
                    dataset,

                "morph_rank":
                    3,

                "t":
                    1,

                "source_pred_node_id":
                    1,

                "target_pred_node_id":
                    3,

                "candidate_rank":
                    3,

                "candidate_prob":
                    0.10,
            },

            {
                "dataset":
                    dataset,

                "morph_rank":
                    4,

                "t":
                    1,

                "source_pred_node_id":
                    1,

                "target_pred_node_id":
                    4,

                "candidate_rank":
                    1,

                "candidate_prob":
                    0.90,
            },

            {
                "dataset":
                    dataset,

                "morph_rank":
                    1,

                "t":
                    1,

                "source_pred_node_id":
                    2,

                "target_pred_node_id":
                    4,

                "candidate_rank":
                    4,

                "candidate_prob":
                    0.90,
            },

            {
                "dataset":
                    dataset,

                "morph_rank":
                    1,

                "t":
                    1,

                "source_pred_node_id":
                    2,

                "target_pred_node_id":
                    5,

                "candidate_rank":
                    1,

                "candidate_prob":
                    0.09,
            },
        ],
        columns=list(
            MORPH_DIV_V2_CANDIDATE_COLUMNS
        ),
    )


class _D3Provider:
    provider_id = "test_provider"
    provider_version = "1"
    provider_source_sha256 = "a" * 64
    provider_authority_status = "ENGINEERING_UNVERIFIED"

    def __init__(
        self,
        rows,
    ):
        self.rows = rows.copy(
            deep=True
        )

    def build_candidates(
        self,
        dataset_id,
        post_frozen_v9_graph,
        *,
        context,
    ):
        return self.rows.copy(
            deep=True
        )


class _D3MutatingProvider(_D3Provider):
    def build_candidates(
        self,
        dataset_id,
        post_frozen_v9_graph,
        *,
        context,
    ):
        post_frozen_v9_graph.add_edge(
            source=2,
            target=5,
            edge_prob=0.5,
        )

        return self.rows.copy(
            deep=True
        )


def test_d3_morph_config_identity_is_frozen():
    assert (
        MORPH_DIV_V2_CONFIG_ID
        ==
        "K3_R3_P100"
    )


def test_d3_provider_identity_required():
    with pytest.raises(
        MorphProviderContractError
    ):
        validate_morph_candidate_provider(
            None
        )


def test_d3_provider_requires_source_sha():
    provider = _D3Provider(
        _d3_candidates()
    )

    provider.provider_source_sha256 = "bad"

    with pytest.raises(
        MorphProviderContractError
    ):
        validate_morph_candidate_provider(
            provider
        )


def test_d3_provider_rejects_unknown_authority_status():
    provider = _D3Provider(
        _d3_candidates()
    )

    provider.provider_authority_status = "TRUST_ME"

    with pytest.raises(
        MorphProviderContractError
    ):
        validate_morph_candidate_provider(
            provider
        )


def test_d3_exact_schema_is_required():
    rows = _d3_candidates()

    rows["extra"] = 1

    with pytest.raises(
        MorphProviderContractError
    ):
        validate_morph_candidate_table(
            rows,
            dataset_id="ds",
        )


def test_d3_schema_order_is_exact():
    rows = _d3_candidates()

    rows = rows[
        list(
            reversed(
                rows.columns
            )
        )
    ]

    with pytest.raises(
        MorphProviderContractError
    ):
        validate_morph_candidate_table(
            rows,
            dataset_id="ds",
        )


def test_d3_empty_schema_valid_table_allowed():
    rows = pd.DataFrame(
        columns=list(
            MORPH_DIV_V2_CANDIDATE_COLUMNS
        )
    )

    out = validate_morph_candidate_table(
        rows,
        dataset_id="ds",
    )

    assert out.empty


def test_d3_dataset_identity_must_match():
    rows = _d3_candidates(
        dataset="wrong"
    )

    with pytest.raises(
        MorphProviderContractError
    ):
        validate_morph_candidate_table(
            rows,
            dataset_id="ds",
        )


def test_d3_nulls_fail_closed():
    rows = _d3_candidates()

    rows.loc[
        0,
        "candidate_prob",
    ] = np.nan

    with pytest.raises(
        MorphProviderContractError
    ):
        validate_morph_candidate_table(
            rows,
            dataset_id="ds",
        )


def test_d3_non_integer_rank_fails():
    rows = _d3_candidates()

    # The fixture intentionally needs to represent a
    # non-integer-like numeric rank.  Newer pandas versions
    # reject assigning 1.5 directly into an int64 column before
    # the runtime validator is reached, so widen the TEST
    # column first.
    rows["morph_rank"] = (
        rows["morph_rank"]
        .astype(float)
    )

    rows.loc[
        0,
        "morph_rank",
    ] = 1.5

    with pytest.raises(
        MorphProviderContractError
    ):
        validate_morph_candidate_table(
            rows,
            dataset_id="ds",
        )


def test_d3_probability_range_fails_closed():
    rows = _d3_candidates()

    rows.loc[
        0,
        "candidate_prob",
    ] = 1.01

    with pytest.raises(
        MorphProviderContractError
    ):
        validate_morph_candidate_table(
            rows,
            dataset_id="ds",
        )


def test_d3_selection_uses_exact_k3_r3_p100_gates():
    rows = _d3_candidates()

    selected = select_morph_div_v2_candidates(
        rows,
        dataset_id="ds",
    )

    assert len(
        selected
    ) == 1

    row = selected.iloc[
        0
    ]

    assert int(
        row[
            "morph_rank"
        ]
    ) == 3

    assert int(
        row[
            "candidate_rank"
        ]
    ) == 3

    assert float(
        row[
            "candidate_prob"
        ]
    ) == pytest.approx(
        0.10
    )


def test_d3_selector_has_no_threshold_arguments():
    import inspect

    signature = inspect.signature(
        select_morph_div_v2_candidates
    )

    assert (
        "threshold"
        not in
        signature.parameters
    )

    assert (
        "prob_threshold"
        not in
        signature.parameters
    )


def test_d3_selection_order_is_deterministic():
    rows = pd.DataFrame(
        [
            {
                "dataset":
                    "ds",

                "morph_rank":
                    2,

                "t":
                    4,

                "source_pred_node_id":
                    9,

                "target_pred_node_id":
                    11,

                "candidate_rank":
                    1,

                "candidate_prob":
                    0.9,
            },

            {
                "dataset":
                    "ds",

                "morph_rank":
                    1,

                "t":
                    8,

                "source_pred_node_id":
                    7,

                "target_pred_node_id":
                    10,

                "candidate_rank":
                    1,

                "candidate_prob":
                    0.9,
            },
        ],
        columns=list(
            MORPH_DIV_V2_CANDIDATE_COLUMNS
        ),
    )

    selected = select_morph_div_v2_candidates(
        rows,
        dataset_id="ds",
    )

    assert (
        selected[
            "morph_rank"
        ].tolist()
        ==
        [
            1,
            2,
        ]
    )


def test_d3_provider_may_not_mutate_graph():
    graph = _D3FakeGraph(
        node_ids=[
            1,
            2,
            3,
            4,
            5,
        ],
        edges=[
            (
                1,
                2,
                0.8,
            )
        ],
    )

    provider = _D3MutatingProvider(
        _d3_candidates()
    )

    with pytest.raises(
        MorphProviderContractError
    ):
        apply_portable_morph_div_v2(
            "ds",
            graph,
            context=object(),
            provider=provider,
        )


def test_d3_missing_provider_fails_closed():
    graph = _D3FakeGraph(
        node_ids=[
            1,
            2,
            3,
        ],
        edges=[
            (
                1,
                2,
                0.8,
            )
        ],
    )

    with pytest.raises(
        MorphProviderContractError
    ):
        apply_portable_morph_div_v2(
            "ds",
            graph,
            context=object(),
            provider=None,
        )


def test_d3_morph_application_adds_second_source_edge():
    graph = _D3FakeGraph(
        node_ids=[
            1,
            2,
            3,
            4,
            5,
        ],
        edges=[
            (
                1,
                2,
                0.8,
            )
        ],
    )

    rows = pd.DataFrame(
        [
            {
                "dataset":
                    "ds",

                "morph_rank":
                    1,

                "t":
                    1,

                "source_pred_node_id":
                    1,

                "target_pred_node_id":
                    3,

                "candidate_rank":
                    1,

                "candidate_prob":
                    0.72,
            }
        ],
        columns=list(
            MORPH_DIV_V2_CANDIDATE_COLUMNS
        ),
    )

    provider = _D3Provider(
        rows
    )

    result = apply_portable_morph_div_v2(
        "ds",
        graph,
        context=object(),
        provider=provider,
        return_audit=True,
    )

    assert (
        result[
            "candidate_count"
        ]
        ==
        1
    )

    assert (
        result[
            "selected_count"
        ]
        ==
        1
    )

    assert (
        result[
            "accepted_count"
        ]
        ==
        1
    )

    edges = result[
        "graph"
    ].edge_attrs(
        attr_keys=[
            "source",
            "target",
            "edge_prob",
        ]
    ).to_pandas()

    added = edges[
        (
            edges[
                "source"
            ]
            ==
            1
        )
        &
        (
            edges[
                "target"
            ]
            ==
            3
        )
    ]

    assert len(
        added
    ) == 1

    assert float(
        added.iloc[
            0
        ][
            "edge_prob"
        ]
    ) == pytest.approx(
        0.72
    )

    # Provider is engineering-unverified.
    assert (
        result[
            "historical_structural_fidelity_verified"
        ]
        is False
    )


def test_d3_application_skips_source_not_outdegree_one():
    graph = _D3FakeGraph(
        node_ids=[
            1,
            2,
            3,
            4,
        ],
        edges=[
            (
                1,
                2,
                0.8,
            ),
            (
                1,
                4,
                0.7,
            ),
        ],
    )

    rows = pd.DataFrame(
        [
            {
                "dataset":
                    "ds",

                "morph_rank":
                    1,

                "t":
                    1,

                "source_pred_node_id":
                    1,

                "target_pred_node_id":
                    3,

                "candidate_rank":
                    1,

                "candidate_prob":
                    0.72,
            }
        ],
        columns=list(
            MORPH_DIV_V2_CANDIDATE_COLUMNS
        ),
    )

    result = apply_portable_morph_div_v2(
        "ds",
        graph,
        context=object(),
        provider=_D3Provider(
            rows
        ),
        return_audit=True,
    )

    assert (
        result[
            "accepted_count"
        ]
        ==
        0
    )


def test_d3_application_skips_target_not_indegree_zero():
    graph = _D3FakeGraph(
        node_ids=[
            1,
            2,
            3,
            4,
        ],
        edges=[
            (
                1,
                2,
                0.8,
            ),
            (
                4,
                3,
                0.6,
            ),
        ],
    )

    rows = pd.DataFrame(
        [
            {
                "dataset":
                    "ds",

                "morph_rank":
                    1,

                "t":
                    1,

                "source_pred_node_id":
                    1,

                "target_pred_node_id":
                    3,

                "candidate_rank":
                    1,

                "candidate_prob":
                    0.72,
            }
        ],
        columns=list(
            MORPH_DIV_V2_CANDIDATE_COLUMNS
        ),
    )

    result = apply_portable_morph_div_v2(
        "ds",
        graph,
        context=object(),
        provider=_D3Provider(
            rows
        ),
        return_audit=True,
    )

    assert (
        result[
            "accepted_count"
        ]
        ==
        0
    )


def test_d3_provider_referencing_missing_node_fails_closed():
    graph = _D3FakeGraph(
        node_ids=[
            1,
            2,
            3,
        ],
        edges=[
            (
                1,
                2,
                0.8,
            )
        ],
    )

    rows = pd.DataFrame(
        [
            {
                "dataset":
                    "ds",

                "morph_rank":
                    1,

                "t":
                    1,

                "source_pred_node_id":
                    1,

                "target_pred_node_id":
                    999,

                "candidate_rank":
                    1,

                "candidate_prob":
                    0.72,
            }
        ],
        columns=list(
            MORPH_DIV_V2_CANDIDATE_COLUMNS
        ),
    )

    with pytest.raises(
        MorphProviderContractError
    ):
        apply_portable_morph_div_v2(
            "ds",
            graph,
            context=object(),
            provider=_D3Provider(
                rows
            ),
        )


def test_d3_runtime_does_not_mutate_input_graph():
    graph = _D3FakeGraph(
        node_ids=[
            1,
            2,
            3,
        ],
        edges=[
            (
                1,
                2,
                0.8,
            )
        ],
    )

    before = copy.deepcopy(
        graph._edges
    )

    rows = pd.DataFrame(
        [
            {
                "dataset":
                    "ds",

                "morph_rank":
                    1,

                "t":
                    1,

                "source_pred_node_id":
                    1,

                "target_pred_node_id":
                    3,

                "candidate_rank":
                    1,

                "candidate_prob":
                    0.72,
            }
        ],
        columns=list(
            MORPH_DIV_V2_CANDIDATE_COLUMNS
        ),
    )

    apply_portable_morph_div_v2(
        "ds",
        graph,
        context=object(),
        provider=_D3Provider(
            rows
        ),
    )

    pd.testing.assert_frame_equal(
        graph._edges,
        before,
    )


def test_d3_empty_provider_output_is_valid():
    graph = _D3FakeGraph(
        node_ids=[
            1,
            2,
        ],
        edges=[
            (
                1,
                2,
                0.8,
            )
        ],
    )

    rows = pd.DataFrame(
        columns=list(
            MORPH_DIV_V2_CANDIDATE_COLUMNS
        )
    )

    result = apply_portable_morph_div_v2(
        "ds",
        graph,
        context=object(),
        provider=_D3Provider(
            rows
        ),
        return_audit=True,
    )

    assert (
        result[
            "candidate_count"
        ]
        ==
        0
    )

    assert (
        result[
            "selected_count"
        ]
        ==
        0
    )

    assert (
        result[
            "accepted_count"
        ]
        ==
        0
    )


def test_d3_engineering_provider_is_not_fidelity_verified():
    provider = _D3Provider(
        _d3_candidates()
    )

    identity = validate_morph_candidate_provider(
        provider
    )

    assert (
        identity[
            "provider_authority_status"
        ]
        ==
        "ENGINEERING_UNVERIFIED"
    )


# END NOTEBOOK13 13.3D3 MORPH PROVIDER TESTS


# BEGIN NOTEBOOK13 13.3D4-R2R3 TESTS

from types import SimpleNamespace
import inspect

import numpy as np
import pandas as pd
import pytest

from biohub_cell_tracking.stage12_portable_current_best import (
    MUTUAL_CONT_V2_CANDIDATE_PROB_MIN,
    MutualContV2ContractError,
    _portable_r1p50_source_physical_distance,
    apply_portable_det_bridge_r1p50_core,
    apply_portable_mutual_cont_v2,
    build_portable_mutual_cont_v2_actions,
)


class _D4R2R3Graph:

    def __init__(
        self,
        nodes,
        edges=None,
    ):

        self._nodes = pd.DataFrame(
            nodes,
            columns=[
                "node_id",
                "t",
                "z",
                "y",
                "x",
            ],
        )

        self._edges = pd.DataFrame(
            (
                []
                if edges is None
                else edges
            ),
            columns=[
                "source",
                "target",
                "edge_prob",
            ],
        )


    def node_attrs(
        self,
        attr_keys=None,
        *args,
        **kwargs,
    ):

        frame = self._nodes.copy(
            deep=True
        )

        if attr_keys is not None:

            frame = frame[
                list(
                    attr_keys
                )
            ]

        return frame


    def edge_attrs(
        self,
        attr_keys=None,
        *args,
        **kwargs,
    ):

        frame = self._edges.copy(
            deep=True
        )

        if attr_keys is not None:

            existing = [
                column
                for column
                in attr_keys
                if column in frame.columns
            ]

            frame = frame[
                existing
            ]

        return frame


    def add_edge(
        self,
        *args,
        **kwargs,
    ):

        if kwargs:

            source = kwargs.get(
                "source"
            )

            target = kwargs.get(
                "target"
            )

            edge_prob = kwargs.get(
                "edge_prob",
                kwargs.get(
                    "prob",
                    kwargs.get(
                        "weight",
                        1.0,
                    ),
                ),
            )

        else:

            source = args[
                0
            ]

            target = args[
                1
            ]

            edge_prob = (
                args[
                    2
                ]
                if len(args) >= 3
                else 1.0
            )

        row = pd.DataFrame(
            [
                {
                    "source":
                        int(
                            source
                        ),

                    "target":
                        int(
                            target
                        ),

                    "edge_prob":
                        float(
                            edge_prob
                        ),
                }
            ]
        )

        self._edges = pd.concat(
            [
                self._edges,
                row,
            ],
            ignore_index=True,
        )


class _D4R2R3Provider:

    def __init__(
        self,
        matrix,
    ):

        self.matrix = np.asarray(
            matrix,
            dtype=float,
        )


    def __call__(
        self,
        dataset_id,
        nodes,
        t,
        *,
        context,
    ):

        if int(t) != 0:

            return (
                np.asarray(
                    [],
                    dtype=int,
                ),
                np.asarray(
                    [],
                    dtype=int,
                ),
                None,
            )

        return (
            np.asarray(
                [
                    1,
                    2,
                ],
                dtype=int,
            ),
            np.asarray(
                [
                    3,
                    4,
                ],
                dtype=int,
            ),
            self.matrix.copy(),
        )


def _d4r2r3_graph(
    edges=None,
):

    return _D4R2R3Graph(
        nodes=[
            (
                1,
                0,
                0.0,
                0.0,
                0.0,
            ),
            (
                2,
                0,
                0.0,
                10.0,
                0.0,
            ),
            (
                3,
                1,
                0.0,
                0.0,
                1.0,
            ),
            (
                4,
                1,
                0.0,
                10.0,
                1.0,
            ),
        ],
        edges=edges,
    )


def _d4r2r3_context(
    matrix,
):

    return SimpleNamespace(
        association_provider=
            _D4R2R3Provider(
                matrix
            ),

        association_provider_identity_sha256=
            "c" * 64,
    )


def test_d4r2r3_v2_frozen_threshold():

    assert (
        MUTUAL_CONT_V2_CANDIDATE_PROB_MIN
        ==
        pytest.approx(
            0.25
        )
    )


def test_d4r2r3_v2_threshold_boundary():

    actions = (
        build_portable_mutual_cont_v2_actions(
            "ds",
            _d4r2r3_graph(),
            context=_d4r2r3_context(
                [
                    [
                        0.25,
                        0.01,
                    ],
                    [
                        0.01,
                        0.24,
                    ],
                ]
            ),
        )
    )

    assert len(
        actions
    ) == 1

    row = actions.iloc[
        0
    ]

    assert int(
        row[
            "source_pred_node_id"
        ]
    ) == 1

    assert int(
        row[
            "target_pred_node_id"
        ]
    ) == 3

    assert float(
        row[
            "candidate_prob"
        ]
    ) == pytest.approx(
        0.25
    )


def test_d4r2r3_v2_reciprocal_top1():

    actions = (
        build_portable_mutual_cont_v2_actions(
            "ds",
            _d4r2r3_graph(),
            context=_d4r2r3_context(
                [
                    [
                        0.90,
                        0.10,
                    ],
                    [
                        0.10,
                        0.80,
                    ],
                ]
            ),
        )
    )

    pairs = {
        (
            int(
                row.source_pred_node_id
            ),
            int(
                row.target_pred_node_id
            ),
        )

        for row
        in actions.itertuples(
            index=False
        )
    }

    assert pairs == {
        (
            1,
            3,
        ),
        (
            2,
            4,
        ),
    }


def test_d4r2r3_v2_post_v1_free_endpoints():

    graph = _d4r2r3_graph(
        edges=[
            (
                1,
                4,
                0.7,
            ),
            (
                2,
                3,
                0.8,
            ),
        ]
    )

    actions = (
        build_portable_mutual_cont_v2_actions(
            "ds",
            graph,
            context=_d4r2r3_context(
                [
                    [
                        0.95,
                        0.10,
                    ],
                    [
                        0.10,
                        0.90,
                    ],
                ]
            ),
        )
    )

    assert (
        1
        not in
        actions[
            "source_pred_node_id"
        ].tolist()
    )

    assert (
        3
        not in
        actions[
            "target_pred_node_id"
        ].tolist()
    )


def test_d4r2r3_v2_provider_identity_fail_closed():

    context = SimpleNamespace(
        association_provider=
            _D4R2R3Provider(
                [
                    [
                        0.9,
                        0.1,
                    ],
                    [
                        0.1,
                        0.8,
                    ],
                ]
            ),

        association_provider_identity_sha256=
            "not-a-sha",
    )

    with pytest.raises(
        MutualContV2ContractError
    ):

        build_portable_mutual_cont_v2_actions(
            "ds",
            _d4r2r3_graph(),
            context=context,
        )


def test_d4r2r3_v2_input_graph_not_mutated():

    graph = _d4r2r3_graph()

    before_nodes = graph._nodes.copy(
        deep=True
    )

    before_edges = graph._edges.copy(
        deep=True
    )

    result = (
        apply_portable_mutual_cont_v2(
            "ds",
            graph,
            context=_d4r2r3_context(
                [
                    [
                        0.90,
                        0.10,
                    ],
                    [
                        0.10,
                        0.80,
                    ],
                ]
            ),
            return_audit=True,
        )
    )

    pd.testing.assert_frame_equal(
        graph._nodes,
        before_nodes,
    )

    pd.testing.assert_frame_equal(
        graph._edges,
        before_edges,
    )

    assert (
        result[
            "generated_action_count"
        ]
        ==
        2
    )

    assert (
        result[
            "accepted_action_count"
        ]
        ==
        2
    )

    assert (
        result[
            "structural_fidelity_verified"
        ]
        is False
    )


def test_d4r2r3_r1p50_root_explicit_inputs():

    signature = inspect.signature(
        apply_portable_det_bridge_r1p50_core
    )

    assert (
        "eligible"
        in
        signature.parameters
    )

    assert (
        "scale"
        in
        signature.parameters
    )

    eligible = signature.parameters[
        "eligible"
    ]

    scale = signature.parameters[
        "scale"
    ]

    assert (
        eligible.kind
        ==
        inspect.Parameter.KEYWORD_ONLY
    )

    assert (
        scale.kind
        ==
        inspect.Parameter.KEYWORD_ONLY
    )

    assert (
        eligible.default
        is
        inspect.Parameter.empty
    )

    assert (
        scale.default
        is
        inspect.Parameter.empty
    )


def test_d4r2r3_physical_distance_explicit_scale():

    signature = inspect.signature(
        _portable_r1p50_source_physical_distance
    )

    assert (
        "scale"
        in
        signature.parameters
    )

    parameter = signature.parameters[
        "scale"
    ]

    assert (
        parameter.kind
        ==
        inspect.Parameter.KEYWORD_ONLY
    )

    assert (
        parameter.default
        is
        inspect.Parameter.empty
    )

    a = {
        "z":
            0.0,
        "y":
            0.0,
        "x":
            0.0,
    }

    b = {
        "z":
            1.0,
        "y":
            2.0,
        "x":
            3.0,
    }

    scale = np.asarray(
        [
            1.625,
            0.40625,
            0.40625,
        ],
        dtype=float,
    )

    expected = float(
        np.linalg.norm(
            np.asarray(
                [
                    -1.0,
                    -2.0,
                    -3.0,
                ]
            )
            *
            scale
        )
    )

    observed = (
        _portable_r1p50_source_physical_distance(
            a,
            b,
            scale=scale,
        )
    )

    assert observed == pytest.approx(
        expected
    )


def test_d4r2r3_r1p50_missing_explicit_inputs_fail_python_signature():

    signature = inspect.signature(
        apply_portable_det_bridge_r1p50_core
    )

    assert (
        signature.parameters[
            "eligible"
        ].default
        is
        inspect.Parameter.empty
    )

    assert (
        signature.parameters[
            "scale"
        ].default
        is
        inspect.Parameter.empty
    )


# END NOTEBOOK13 13.3D4-R2R3 TESTS


# === TEST_P995_BASELINE_NODE_STATE_PROVIDER_V1 ===

from contextlib import contextmanager as _p995_test_contextmanager
from dataclasses import fields as _p995_dataclass_fields
from pathlib import Path as _p995_test_Path
from types import SimpleNamespace as _p995_SimpleNamespace

import numpy as _p995_test_np

import biohub_cell_tracking.stage12_portable_current_best as _p995_portable


def _p995_test_cfg(*, use_ilp=False):
    return _p995_SimpleNamespace(
        det_threshold=0.5,
        det_tta=False,
        edge_activation="sigmoid",
        ilp_appearance_weight=1.25,
        ilp_disappearance_weight=2.25,
        ilp_division_weight=3.25,
        ilp_edge_weight=4.25,
        max_children_per_node=2,
        max_parents_per_node=1,
        pool_kernel_um=(1.0, 1.0, 1.0),
        threshold=0.5,
        use_ilp=use_ilp,
    )


class _P995FakeTable:
    def __init__(self, values):
        self.values = _p995_test_np.asarray(values)

    def __getitem__(self, columns):
        assert columns == ["t", "z", "y", "x"]
        return self

    def to_numpy(self):
        return self.values


class _P995FakeNodeAttrs:
    def __init__(self, values):
        self.values = values

    def to_pandas(self):
        return _P995FakeTable(
            self.values
        )


class _P995FakeGraph:
    def __init__(
        self,
        values,
        *,
        edge_count,
    ):
        self.values = values
        self.edge_count = edge_count
        self.requested_attrs = None

    def num_edges(self):
        return self.edge_count

    def node_attrs(
        self,
        *,
        attr_keys,
    ):
        self.requested_attrs = list(
            attr_keys
        )

        assert (
            self.requested_attrs
            ==
            ["t", "z", "y", "x"]
        )

        return _P995FakeNodeAttrs(
            self.values
        )


def _p995_fake_authority(
    *,
    edge_count=0,
):
    calls = {}

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    class _MPS:
        @staticmethod
        def is_available():
            return False

    def _device(name):
        calls.setdefault(
            "device",
            [],
        ).append(name)

        return f"device:{name}"

    torch = _p995_SimpleNamespace(
        cuda=_Cuda(),
        backends=_p995_SimpleNamespace(
            mps=_MPS(),
        ),
        device=_device,
    )

    class _EdgeAttr:
        def __init__(self, name):
            self.name = name

        def __rmul__(self, value):
            return (
                float(value),
                self.name,
            )

    class _Solver:
        def __init__(
            self,
            **kwargs,
        ):
            calls[
                "solver_init"
            ] = kwargs

        def solve(
            self,
            graph,
        ):
            calls[
                "solver_solve"
            ] = graph

            return graph

    td = _p995_SimpleNamespace(
        EdgeAttr=_EdgeAttr,
        solvers=_p995_SimpleNamespace(
            ILPSolver=_Solver,
        ),
    )

    @_p995_test_contextmanager
    def suppress_output():
        calls[
            "suppress_output"
        ] = (
            calls.get(
                "suppress_output",
                0,
            )
            +
            1
        )

        yield

    def load_model(
        weights_path,
        device,
    ):
        calls[
            "load_model"
        ] = {
            "weights_path":
                weights_path,

            "device":
                device,
        }

        return (
            "MODEL",
            2,
            (1, 4, 4),
        )

    def predict_video(
        model,
        dataset_path,
        device,
        **kwargs,
    ):
        calls[
            "predict_video"
        ] = {
            "model":
                model,

            "dataset_path":
                dataset_path,

            "device":
                device,

            "kwargs":
                dict(kwargs),
        }

        return (
            "COORDS",
            "EDGES",
        )

    graph = _P995FakeGraph(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
        edge_count=edge_count,
    )

    def build_graph(
        coords,
        edges,
    ):
        calls[
            "build_graph"
        ] = {
            "coords":
                coords,

            "edges":
                edges,
        }

        return graph

    module = _p995_SimpleNamespace(
        torch=torch,
        td=td,
        suppress_output=suppress_output,
        load_model=load_model,
        predict_video=predict_video,
        build_graph=build_graph,
    )

    return (
        module,
        calls,
        graph,
    )


def test_p995_provider_frozen_public_contract():
    assert (
        _p995_portable.P995_BASELINE_NODE_STATE_PROVIDER_ID
        ==
        "P995_BASELINE_NODE_STATE_PROVIDER_V1"
    )

    assert (
        _p995_portable.P995_BASELINE_PREDICT_AUTHORITY_MODULE
        ==
        "scripts.predict_unet_transformer_mps"
    )

    assert (
        _p995_portable.P995_BASELINE_PREDICT_SOURCE_SHA256
        ==
        "23b76f7c90ffe698e955efee2b4333277a07f5b58b7eff6e580f2bb2ff49bc55"
    )

    assert (
        _p995_portable.P995_BASELINE_LOAD_MODEL_SOURCE_SHA256
        ==
        "bbabeb0d3116689002ca9f9c955fba716c145282f08ed104e67ab0764e616e1b"
    )

    assert (
        _p995_portable.P995_BASELINE_PROVIDER_AUTHORITY_STATUS
        ==
        "SOURCE_IMPLEMENTED_STRUCTURAL_FIDELITY_PENDING"
    )

    assert (
        tuple(
            _p995_portable.P995_BASELINE_NODE_COLUMNS
        )
        ==
        ("t", "z", "y", "x")
    )

    assert [
        field.name
        for field
        in _p995_dataclass_fields(
            _p995_portable.P995BaselineNodeStateProviderContext
        )
    ] == [
        "data_root",
        "weights_path",
        "cfg",
        "max_frames",
        "unet_batch_size",
    ]


def test_p995_provider_gt_blind_wiring_without_ilp(
    monkeypatch,
):
    authority, calls, graph = (
        _p995_fake_authority(
            edge_count=0,
        )
    )

    monkeypatch.setattr(
        _p995_portable,
        "_load_p995_predict_authority_module",
        lambda: authority,
    )

    cfg = _p995_test_cfg(
        use_ilp=False,
    )

    context = (
        _p995_portable.P995BaselineNodeStateProviderContext(
            data_root=_p995_test_Path(
                "/fresh/data"
            ),
            weights_path=_p995_test_Path(
                "/frozen/model/edge_predictor_best.pth"
            ),
            cfg=cfg,
            max_frames=None,
            unet_batch_size=4,
        )
    )

    result = (
        _p995_portable.baseline_node_state_provider(
            "dataset_A",
            context=context,
        )
    )

    assert (
        result.dtype
        ==
        _p995_test_np.int64
    )

    assert (
        result.shape
        ==
        (2, 4)
    )

    assert (
        result.tolist()
        ==
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
    )

    assert (
        calls[
            "device"
        ]
        ==
        ["cpu"]
    )

    assert (
        calls[
            "load_model"
        ][
            "weights_path"
        ]
        ==
        _p995_test_Path(
            "/frozen/model/edge_predictor_best.pth"
        )
    )

    predict_call = (
        calls[
            "predict_video"
        ]
    )

    assert (
        predict_call[
            "dataset_path"
        ]
        ==
        _p995_test_Path(
            "/fresh/data/dataset_A"
        )
    )

    assert (
        predict_call[
            "model"
        ]
        ==
        "MODEL"
    )

    assert (
        predict_call[
            "device"
        ]
        ==
        "device:cpu"
    )

    assert (
        predict_call[
            "kwargs"
        ][
            "cfg"
        ]
        is
        cfg
    )

    assert (
        predict_call[
            "kwargs"
        ][
            "window_size"
        ]
        ==
        2
    )

    assert (
        predict_call[
            "kwargs"
        ][
            "downsample"
        ]
        ==
        (1, 4, 4)
    )

    assert (
        predict_call[
            "kwargs"
        ][
            "max_frames"
        ]
        is None
    )

    assert (
        predict_call[
            "kwargs"
        ][
            "unet_batch_size"
        ]
        ==
        4
    )

    assert (
        calls[
            "build_graph"
        ]
        ==
        {
            "coords":
                "COORDS",

            "edges":
                "EDGES",
        }
    )

    assert (
        "solver_init"
        not in
        calls
    )

    assert (
        graph.requested_attrs
        ==
        ["t", "z", "y", "x"]
    )

    assert not hasattr(
        authority,
        "save_graph",
    )


def test_p995_provider_applies_frozen_ilp_branch(
    monkeypatch,
):
    authority, calls, _ = (
        _p995_fake_authority(
            edge_count=1,
        )
    )

    monkeypatch.setattr(
        _p995_portable,
        "_load_p995_predict_authority_module",
        lambda: authority,
    )

    cfg = _p995_test_cfg(
        use_ilp=True,
    )

    context = (
        _p995_portable.P995BaselineNodeStateProviderContext(
            data_root=_p995_test_Path(
                "/fresh/data"
            ),
            weights_path=_p995_test_Path(
                "/frozen/model/edge_predictor_best.pth"
            ),
            cfg=cfg,
            max_frames=17,
            unet_batch_size=8,
        )
    )

    result = (
        _p995_portable.baseline_node_state_provider(
            "dataset_B",
            context=context,
        )
    )

    assert (
        result.shape
        ==
        (2, 4)
    )

    assert (
        result.dtype
        ==
        _p995_test_np.int64
    )

    assert (
        calls[
            "predict_video"
        ][
            "kwargs"
        ][
            "max_frames"
        ]
        ==
        17
    )

    assert (
        calls[
            "predict_video"
        ][
            "kwargs"
        ][
            "unet_batch_size"
        ]
        ==
        8
    )

    assert (
        calls[
            "solver_init"
        ][
            "edge_weight"
        ]
        ==
        (
            4.25,
            "edge_prob",
        )
    )

    assert (
        calls[
            "solver_init"
        ][
            "appearance_weight"
        ]
        ==
        1.25
    )

    assert (
        calls[
            "solver_init"
        ][
            "disappearance_weight"
        ]
        ==
        2.25
    )

    assert (
        calls[
            "solver_init"
        ][
            "division_weight"
        ]
        ==
        3.25
    )

    assert (
        calls.get(
            "suppress_output"
        )
        ==
        1
    )

    assert (
        "solver_solve"
        in
        calls
    )


def test_p995_provider_rejects_incomplete_cfg_before_authority_load(
    monkeypatch,
):
    authority_loaded = {
        "value":
            False,
    }

    def _should_not_load():
        authority_loaded[
            "value"
        ] = True

        raise AssertionError(
            "authority must not load before cfg validation"
        )

    monkeypatch.setattr(
        _p995_portable,
        "_load_p995_predict_authority_module",
        _should_not_load,
    )

    context = (
        _p995_portable.P995BaselineNodeStateProviderContext(
            data_root=_p995_test_Path(
                "/fresh/data"
            ),
            weights_path=_p995_test_Path(
                "/frozen/model/edge_predictor_best.pth"
            ),
            cfg=_p995_SimpleNamespace(
                use_ilp=False,
            ),
        )
    )

    try:
        _p995_portable.baseline_node_state_provider(
            "dataset_C",
            context=context,
        )

    except TypeError as exc:
        assert (
            "missing frozen required fields"
            in
            str(exc)
        )

    else:
        raise AssertionError(
            "incomplete cfg should have been rejected"
        )

    assert (
        authority_loaded[
            "value"
        ]
        is False
    )


# === TEST_P995_REPO_LOCAL_IMPORT_CLOSURE_V3 ===

def test_p995_repo_local_import_roots_are_frozen_v3():
    assert (
        tuple(
            _p995_portable.P995_BASELINE_IMPORT_ROOT_RELATIVE_PATHS
        )
        ==
        ('.', 'scripts', 'external/official/scripts', 'external/official/src')
    )

    assert (
        _p995_portable.P995_BASELINE_IMPORT_ROOT_RELATIVE_PATHS[0]
        ==
        "."
    )

    assert (
        "scripts"
        in
        _p995_portable.P995_BASELINE_IMPORT_ROOT_RELATIVE_PATHS
    )

    assert (
        'external/official/scripts'
        in
        _p995_portable.P995_BASELINE_IMPORT_ROOT_RELATIVE_PATHS
    )

    assert (
        'external/official/src'
        in
        _p995_portable.P995_BASELINE_IMPORT_ROOT_RELATIVE_PATHS
    )

    assert (
        len(
            _p995_portable.P995_BASELINE_IMPORT_ROOT_RELATIVE_PATHS
        )
        ==
        len(
            set(
                _p995_portable.P995_BASELINE_IMPORT_ROOT_RELATIVE_PATHS
            )
        )
    )
