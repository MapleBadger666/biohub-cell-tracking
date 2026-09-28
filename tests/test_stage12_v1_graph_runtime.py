from __future__ import annotations

import inspect

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import biohub_cell_tracking.stage12_portable_current_best as portable


class _Table:
    def __init__(self, frame):
        self._frame = frame.copy(deep=True)

    def to_pandas(self):
        return self._frame.copy(deep=True)


class _Graph:
    def __init__(self, nodes, edges=None):
        self._nodes = pd.DataFrame(nodes).copy(deep=True)

        if edges is None:
            edges = []

        self._edges = pd.DataFrame(
            edges,
            columns=[
                "edge_id",
                "source_id",
                "target_id",
                "edge_dist",
                "edge_prob",
            ],
        ).copy(deep=True)

    def node_attrs(self, attr_keys=None, unpack=True):
        frame = self._nodes.copy(deep=True)

        if attr_keys is not None:
            frame = frame[
                list(attr_keys)
            ].copy(deep=True)

        return _Table(frame)

    def edge_attrs(self, attr_keys=None, unpack=True):
        frame = self._edges.copy(deep=True)

        if attr_keys is not None:
            requested = list(attr_keys)

            base = [
                column
                for column in (
                    "edge_id",
                    "source_id",
                    "target_id",
                )
                if column in frame.columns
            ]

            requested = base + [
                column
                for column in requested
                if column not in base
            ]

            frame = frame[
                requested
            ].copy(deep=True)

        return _Table(frame)

    def add_edge(
        self,
        source,
        target,
        attrs,
        validate_keys=False,
    ):
        assert validate_keys is True

        record = {
            "edge_id":
                int(len(self._edges)),

            "source_id":
                int(source),

            "target_id":
                int(target),

            "edge_dist":
                float(attrs["edge_dist"]),

            "edge_prob":
                float(attrs["edge_prob"]),
        }

        self._edges = pd.concat(
            [
                self._edges,
                pd.DataFrame([record]),
            ],
            ignore_index=True,
        )

        return int(
            record[
                "edge_id"
            ]
        )


def _graph():
    return _Graph(
        [
            {
                "node_id": 1,
                "t": 0,
                "z": 0.0,
                "y": 0.0,
                "x": 0.0,
            },
            {
                "node_id": 2,
                "t": 1,
                "z": 0.0,
                "y": 3.0,
                "x": 4.0,
            },
        ]
    )


def _context(provider, identity=None):
    if identity is None:
        identity = (
            portable.
            REFERENCE_ASSOCIATION_PROVIDER_SHA256
        )

    return portable.PortableCurrentBestContext(
        b256_pred_995_root=Path("."),
        b256_pred_9975_root=Path("."),
        s9995_pred_995_root=Path("."),
        s9995_pred_9975_root=Path("."),
        dataset_root=Path("."),
        physical_scale_zyx_um=(1.0, 1.0, 1.0),
        frozen_v9_policy_bundle=object(),
        association_provider=provider,
        association_provider_identity_sha256=identity,
        strict=True,
    )


def _single_pair_provider(probability):
    calls = []

    def provider(
        dataset_id,
        nodes,
        t,
        *,
        context,
    ):
        calls.append(
            (
                dataset_id,
                int(t),
                tuple(nodes["node_id"].astype(int).tolist()),
            )
        )

        if int(t) == 0:
            return (
                np.asarray(
                    [1],
                    dtype=int,
                ),
                np.asarray(
                    [2],
                    dtype=int,
                ),
                np.asarray(
                    [[float(probability)]],
                    dtype=float,
                ),
            )

        return (
            np.asarray(
                [],
                dtype=int,
            ),
            np.asarray(
                [],
                dtype=int,
            ),
            np.empty(
                (0, 0),
                dtype=float,
            ),
        )

    provider.calls = calls

    return provider


def test_v1_graph_runtime_adds_frozen_continuation_edge(
    monkeypatch,
):
    graph = _graph()

    provider = _single_pair_provider(
        0.80
    )

    monkeypatch.setattr(
        portable._frozen_v9,
        "recovered_edge_dist",
        lambda nodes, source, target, config: 5.0,
    )

    result = (
        portable.
        _reconstruct_portable_mutual_cont_v1(
            graph,
            "ds",
            context=_context(
                provider
            ),
        )
    )

    # Runtime works on a clone.
    assert result is not graph

    assert len(
        graph._edges
    ) == 0

    assert len(
        result._edges
    ) == 1

    row = result._edges.iloc[0]

    assert int(
        row["source_id"]
    ) == 1

    assert int(
        row["target_id"]
    ) == 2

    assert float(
        row["edge_dist"]
    ) == pytest.approx(
        5.0
    )

    assert float(
        row["edge_prob"]
    ) == pytest.approx(
        0.80
    )

    assert provider.calls[0][0] == "ds"

    assert provider.calls[0][1] == 0


def test_v1_graph_runtime_keeps_exact_threshold_boundary(
    monkeypatch,
):
    monkeypatch.setattr(
        portable._frozen_v9,
        "recovered_edge_dist",
        lambda nodes, source, target, config: 1.0,
    )

    at_threshold = (
        portable.
        _reconstruct_portable_mutual_cont_v1(
            _graph(),
            "ds",
            context=_context(
                _single_pair_provider(
                    0.25
                )
            ),
        )
    )

    below_threshold = (
        portable.
        _reconstruct_portable_mutual_cont_v1(
            _graph(),
            "ds",
            context=_context(
                _single_pair_provider(
                    0.249999
                )
            ),
        )
    )

    assert len(
        at_threshold._edges
    ) == 1

    assert len(
        below_threshold._edges
    ) == 0


def test_v1_graph_runtime_respects_source_outdegree_zero_gate(
    monkeypatch,
):
    graph = _Graph(
        [
            {
                "node_id": 1,
                "t": 0,
                "z": 0.0,
                "y": 0.0,
                "x": 0.0,
            },
            {
                "node_id": 2,
                "t": 1,
                "z": 0.0,
                "y": 3.0,
                "x": 4.0,
            },
            {
                "node_id": 3,
                "t": 1,
                "z": 0.0,
                "y": 1.0,
                "x": 1.0,
            },
        ],
        [
            {
                "edge_id": 0,
                "source_id": 1,
                "target_id": 3,
                "edge_dist": 1.0,
                "edge_prob": 0.9,
            },
        ],
    )

    monkeypatch.setattr(
        portable._frozen_v9,
        "recovered_edge_dist",
        lambda nodes, source, target, config: 5.0,
    )

    result = (
        portable.
        _reconstruct_portable_mutual_cont_v1(
            graph,
            "ds",
            context=_context(
                _single_pair_provider(
                    0.95
                )
            ),
        )
    )

    # Existing edge only; source 1 is not a free endpoint.
    assert len(
        result._edges
    ) == 1

    row = result._edges.iloc[0]

    assert (
        int(
            row["source_id"]
        ),
        int(
            row["target_id"]
        ),
    ) == (
        1,
        3,
    )


def test_v1_graph_runtime_fails_closed_on_provider_identity_drift():
    provider = _single_pair_provider(
        0.90
    )

    with pytest.raises(
        portable.PortableRuntimeContractError,
        match="association-provider identity drift",
    ):
        (
            portable.
            _reconstruct_portable_mutual_cont_v1(
                _graph(),
                "ds",
                context=_context(
                    provider,
                    identity="f" * 64,
                ),
            )
        )


def test_v1_graph_runtime_fails_closed_on_unknown_provider_nodes():
    def provider(
        dataset_id,
        nodes,
        t,
        *,
        context,
    ):
        if int(t) == 0:
            return (
                np.asarray(
                    [999],
                    dtype=int,
                ),
                np.asarray(
                    [2],
                    dtype=int,
                ),
                np.asarray(
                    [[0.90]],
                    dtype=float,
                ),
            )

        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.empty((0, 0), dtype=float),
        )

    with pytest.raises(
        portable.PortableRuntimeContractError,
        match="outside the current post-Morph graph",
    ):
        (
            portable.
            _reconstruct_portable_mutual_cont_v1(
                _graph(),
                "ds",
                context=_context(
                    provider
                ),
            )
        )


def test_v1_graph_runtime_contains_no_historical_action_lookup():
    source = inspect.getsource(
        portable.
        _reconstruct_portable_mutual_cont_v1
    )

    assert "read_pickle" not in source

    assert (
        "stage12_fold0_mutual_cont_v1_actions_locked.pkl"
        not in source
    )

    assert "evaluate_graph" not in source
