"""Fresh post-V2 graph orchestration for Stage12 portability.

Engineering boundary only.

Exact stage order:

    FrozenV9 -> Morph-DIV-V2 -> MUTUAL-CONT-V1 -> MUTUAL-CONT-V2

R1P50 is intentionally excluded.

The Morph candidate provider is an explicit dependency carried by
PortableCurrentBestContext. Historical Morph/V1 action artifacts are never
runtime inputs here.
"""

from __future__ import annotations

from typing import Any

from . import stage12_portable_current_best as portable


def post_v2_graph_provider(
    dataset_id: str,
    *,
    context: portable.PortableCurrentBestContext,
) -> Any:
    """Build the fresh graph at the historical post-V2 boundary."""

    dataset_id = portable._validate_dataset_id(
        dataset_id
    )

    if not isinstance(
        context,
        portable.PortableCurrentBestContext,
    ):
        raise portable.PortableRuntimeContractError(
            "context must be PortableCurrentBestContext."
        )

    morph_provider = (
        context.morph_candidate_provider
    )

    if (
        morph_provider
        is None
    ):
        raise portable.PortableRuntimeContractError(
            "post_v2_graph_provider requires "
            "context.morph_candidate_provider; "
            "no historical-action or implicit-provider fallback is allowed."
        )

    frozen_result = (
        portable._build_portable_frozen_v9(
            dataset_id,
            context=context,
        )
    )

    if (
        not isinstance(
            frozen_result,
            tuple,
        )
        or
        len(
            frozen_result
        )
        !=
        3
    ):
        raise portable.PortableRuntimeContractError(
            "_build_portable_frozen_v9 must return "
            "(graph, scale, stats)."
        )

    post_frozen_v9_graph = (
        frozen_result[
            0
        ]
    )

    post_morph_graph = (
        portable.apply_portable_morph_div_v2(
            dataset_id,
            post_frozen_v9_graph,
            context=context,
            provider=morph_provider,
            return_audit=False,
        )
    )

    post_v1_graph = (
        portable._reconstruct_portable_mutual_cont_v1(
            post_morph_graph,
            dataset_id,
            context=context,
        )
    )

    post_v2_graph = (
        portable.apply_portable_mutual_cont_v2(
            dataset_id,
            post_v1_graph,
            context=context,
            return_audit=False,
        )
    )

    return post_v2_graph
