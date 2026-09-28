from __future__ import annotations

import inspect

from pathlib import Path

import pytest

from biohub_cell_tracking import stage12_portable_current_best as portable
from biohub_cell_tracking import stage12_post_v2_graph_provider as post_v2


def _context(morph_provider=None):
    def association_provider(*args, **kwargs):
        raise AssertionError(
            "association provider should be monkeypatched away in "
            "orchestration tests."
        )

    return portable.PortableCurrentBestContext(
        b256_pred_995_root=Path("."),
        b256_pred_9975_root=Path("."),
        s9995_pred_995_root=Path("."),
        s9995_pred_9975_root=Path("."),
        dataset_root=Path("."),
        physical_scale_zyx_um=(1.0, 1.0, 1.0),
        frozen_v9_policy_bundle=object(),
        association_provider=association_provider,
        association_provider_identity_sha256=(
            portable.REFERENCE_ASSOCIATION_PROVIDER_SHA256
        ),
        morph_candidate_provider=morph_provider,
        strict=True,
    )


def test_context_exposes_explicit_morph_provider_seam():
    context = _context()

    assert hasattr(
        context,
        "morph_candidate_provider",
    )

    assert (
        context.morph_candidate_provider
        is None
    )


def test_post_v2_provider_fails_closed_when_morph_provider_missing(
    monkeypatch,
):
    def forbidden_frozen(*args, **kwargs):
        raise AssertionError(
            "FrozenV9 must not execute before missing Morph provider "
            "is rejected."
        )

    monkeypatch.setattr(
        post_v2.portable,
        "_build_portable_frozen_v9",
        forbidden_frozen,
    )

    with pytest.raises(
        portable.PortableRuntimeContractError,
        match="morph_candidate_provider",
    ):
        post_v2.post_v2_graph_provider(
            "ds",
            context=_context(
                None
            ),
        )


def test_post_v2_provider_exact_stage_order_and_arguments(
    monkeypatch,
):
    events = []

    morph_provider = object()

    g0 = object()
    g1 = object()
    g2 = object()
    g3 = object()

    context = _context(
        morph_provider
    )

    def frozen(
        dataset_id,
        *,
        context,
    ):
        events.append(
            (
                "FROZEN_V9",
                dataset_id,
            )
        )

        return (
            g0,
            (1.0, 1.0, 1.0),
            {
                "ok": True,
            },
        )

    def morph(
        dataset_id,
        post_frozen_v9_graph,
        *,
        context,
        provider,
        return_audit=False,
    ):
        events.append(
            (
                "MORPH",
                dataset_id,
                post_frozen_v9_graph,
                provider,
                return_audit,
            )
        )

        assert (
            post_frozen_v9_graph
            is g0
        )

        assert (
            provider
            is morph_provider
        )

        assert (
            return_audit
            is False
        )

        return g1

    def v1(
        graph,
        dataset_id,
        *,
        context,
    ):
        events.append(
            (
                "V1",
                dataset_id,
                graph,
            )
        )

        assert (
            graph
            is g1
        )

        return g2

    def v2(
        dataset_id,
        post_v1_graph,
        *,
        context,
        return_audit=False,
    ):
        events.append(
            (
                "V2",
                dataset_id,
                post_v1_graph,
                return_audit,
            )
        )

        assert (
            post_v1_graph
            is g2
        )

        assert (
            return_audit
            is False
        )

        return g3

    monkeypatch.setattr(
        post_v2.portable,
        "_build_portable_frozen_v9",
        frozen,
    )

    monkeypatch.setattr(
        post_v2.portable,
        "apply_portable_morph_div_v2",
        morph,
    )

    monkeypatch.setattr(
        post_v2.portable,
        "_reconstruct_portable_mutual_cont_v1",
        v1,
    )

    monkeypatch.setattr(
        post_v2.portable,
        "apply_portable_mutual_cont_v2",
        v2,
    )

    result = (
        post_v2.post_v2_graph_provider(
            "ds",
            context=context,
        )
    )

    assert result is g3

    assert [
        event[0]
        for event in events
    ] == [
        "FROZEN_V9",
        "MORPH",
        "V1",
        "V2",
    ]


def test_post_v2_provider_uses_frozen_v9_graph_element_zero(
    monkeypatch,
):
    morph_provider = object()

    graph = object()
    scale = object()
    stats = object()

    captured = {}

    monkeypatch.setattr(
        post_v2.portable,
        "_build_portable_frozen_v9",
        lambda dataset_id, *, context: (
            graph,
            scale,
            stats,
        ),
    )

    def morph(
        dataset_id,
        post_frozen_v9_graph,
        *,
        context,
        provider,
        return_audit=False,
    ):
        captured[
            "graph"
        ] = post_frozen_v9_graph

        return "morph"

    monkeypatch.setattr(
        post_v2.portable,
        "apply_portable_morph_div_v2",
        morph,
    )

    monkeypatch.setattr(
        post_v2.portable,
        "_reconstruct_portable_mutual_cont_v1",
        lambda graph, dataset_id, *, context: "v1",
    )

    monkeypatch.setattr(
        post_v2.portable,
        "apply_portable_mutual_cont_v2",
        lambda dataset_id, graph, *, context, return_audit=False: "v2",
    )

    result = (
        post_v2.post_v2_graph_provider(
            "ds",
            context=_context(
                morph_provider
            ),
        )
    )

    assert result == "v2"

    assert (
        captured[
            "graph"
        ]
        is graph
    )


def test_post_v2_provider_rejects_invalid_frozen_v9_return_shape(
    monkeypatch,
):
    morph_provider = object()

    monkeypatch.setattr(
        post_v2.portable,
        "_build_portable_frozen_v9",
        lambda dataset_id, *, context: object(),
    )

    with pytest.raises(
        portable.PortableRuntimeContractError,
        match="graph, scale, stats",
    ):
        post_v2.post_v2_graph_provider(
            "ds",
            context=_context(
                morph_provider
            ),
        )


def test_post_v2_provider_contains_no_historical_runtime_lookup_or_r1p50():
    source = inspect.getsource(
        post_v2.post_v2_graph_provider
    ).lower()

    assert "read_pickle" not in source

    assert (
        "stage11_fold0_morphdiv_v2_actions_locked.pkl"
        not in source
    )

    assert (
        "stage12_fold0_mutual_cont_v1_actions_locked.pkl"
        not in source
    )

    assert "r1p50" not in source

    assert "evaluate_graph" not in source

    assert "ground_truth" not in source

    assert "gt_graph" not in source
