"""Portable Stage12 pre-G3H3 CURRENT_BEST construction.

This module is intentionally fail-closed while source-derived wiring for
all historical stages is completed.

Implemented here:

* explicit runtime context
* association-provider validation
* MUTUAL-CONT-V1 specification-derived reciprocal-Top1 reconstruction

Not yet claimed here:

* complete portable pre-G3H3 equivalence
* complete historical structural fidelity
* portable CURRENT_BEST equivalence

Scientific thresholds are frozen and must not be tuned in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import math
import re

import numpy as np

from . import frozen_v9 as _frozen_v9


MUTUAL_CONT_V1_PROB_MIN: float = 0.25

REFERENCE_ASSOCIATION_PROVIDER_SHA256: str = (
    "7aae212c3ae52e1070925386fb2775feec0c964337ec48441f1da02d7287a1b1"
)


class PortableCurrentBestError(RuntimeError):
    """Base error for fail-closed portable runtime construction."""


class PortableImplementationIncompleteError(
    PortableCurrentBestError
):
    """Raised when a not-yet-wired frozen stage is requested."""


class PortableRuntimeContractError(
    PortableCurrentBestError
):
    """Raised when explicit runtime inputs violate the frozen contract."""


@dataclass(frozen=True)
class MutualContV1Action:
    """One frozen-spec MUTUAL-CONT-V1 continuation action."""

    dataset_id: str
    source_node_id: int
    target_node_id: int
    candidate_prob: float


@dataclass(frozen=True)
class PortableCurrentBestContext:
    """Explicit runtime dependencies for portable pre-G3H3 construction.

    No historical fold membership, dataset allowlist, or precomputed
    action-table lookup is represented in this context.
    """

    b256_pred_995_root: Path | str
    b256_pred_9975_root: Path | str
    s9995_pred_995_root: Path | str
    s9995_pred_9975_root: Path | str

    dataset_root: Path | str

    physical_scale_zyx_um: tuple[float, float, float]

    frozen_v9_policy_bundle: Any

    association_provider: Callable[..., Any]
    association_provider_identity_sha256: str

    morph_candidate_provider: MorphCandidateProvider | None = None
    strict: bool = True

    def __post_init__(self) -> None:

        if self.strict is not True:

            raise PortableRuntimeContractError(
                "Portable runtime is fail-closed and requires strict=True."
            )

        if not callable(
            self.association_provider
        ):

            raise PortableRuntimeContractError(
                "association_provider must be callable."
            )

        _validate_sha256(
            self.association_provider_identity_sha256,
            field_name="association_provider_identity_sha256",
        )

        scale = tuple(
            float(v)
            for v in self.physical_scale_zyx_um
        )

        if len(scale) != 3:

            raise PortableRuntimeContractError(
                "physical_scale_zyx_um must have exactly three values."
            )

        if not all(
            math.isfinite(v)
            and
            v > 0.0
            for v in scale
        ):

            raise PortableRuntimeContractError(
                "physical_scale_zyx_um must contain finite positive values."
            )

        root_fields = (
            "b256_pred_995_root",
            "b256_pred_9975_root",
            "s9995_pred_995_root",
            "s9995_pred_9975_root",
            "dataset_root",
        )

        for field_name in root_fields:

            value = getattr(
                self,
                field_name,
            )

            if not isinstance(
                value,
                (
                    str,
                    Path,
                ),
            ):

                raise PortableRuntimeContractError(
                    f"{field_name} must be a path-like string or Path."
                )

            if not str(value).strip():

                raise PortableRuntimeContractError(
                    f"{field_name} may not be empty."
                )


def _validate_sha256(
    value: str,
    *,
    field_name: str,
) -> None:

    if not isinstance(
        value,
        str,
    ):

        raise PortableRuntimeContractError(
            f"{field_name} must be a hexadecimal SHA256 string."
        )

    if re.fullmatch(
        r"[0-9a-f]{64}",
        value,
    ) is None:

        raise PortableRuntimeContractError(
            f"{field_name} must be exactly 64 lowercase hex characters."
        )


def _validate_dataset_id(
    dataset_id: str,
) -> str:

    if not isinstance(
        dataset_id,
        str,
    ):

        raise PortableRuntimeContractError(
            "dataset_id must be a string."
        )

    dataset_id = dataset_id.strip()

    if not dataset_id:

        raise PortableRuntimeContractError(
            "dataset_id may not be empty."
        )

    return dataset_id


def _validate_node_ids(
    values: Sequence[int],
    *,
    field_name: str,
) -> np.ndarray:

    array = np.asarray(
        values,
    )

    if array.ndim != 1:

        raise PortableRuntimeContractError(
            f"{field_name} must be one-dimensional."
        )

    try:
        array = array.astype(
            np.int64,
            copy=False,
        )

    except Exception as exc:

        raise PortableRuntimeContractError(
            f"{field_name} must contain integer-compatible node IDs."
        ) from exc

    if len(
        np.unique(
            array
        )
    ) != len(array):

        raise PortableRuntimeContractError(
            f"{field_name} must contain unique node IDs."
        )

    return array


def _validate_association_matrix(
    source_node_ids: Sequence[int],
    target_node_ids: Sequence[int],
    candidate_prob_matrix: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    source_ids = _validate_node_ids(
        source_node_ids,
        field_name="source_node_ids",
    )

    target_ids = _validate_node_ids(
        target_node_ids,
        field_name="target_node_ids",
    )

    probabilities = np.asarray(
        candidate_prob_matrix,
        dtype=np.float64,
    )

    expected_shape = (
        len(source_ids),
        len(target_ids),
    )

    if probabilities.ndim != 2:

        raise PortableRuntimeContractError(
            "candidate_prob_matrix must be two-dimensional."
        )

    if probabilities.shape != expected_shape:

        raise PortableRuntimeContractError(
            "candidate_prob_matrix shape mismatch: "
            f"expected {expected_shape}, got {probabilities.shape}."
        )

    if probabilities.size == 0:

        return (
            source_ids,
            target_ids,
            probabilities,
        )

    if not np.all(
        np.isfinite(
            probabilities
        )
    ):

        raise PortableRuntimeContractError(
            "candidate_prob_matrix contains non-finite probabilities."
        )

    if np.any(
        probabilities < 0.0
    ) or np.any(
        probabilities > 1.0
    ):

        raise PortableRuntimeContractError(
            "candidate_prob_matrix probabilities must lie in [0, 1]."
        )

    return (
        source_ids,
        target_ids,
        probabilities,
    )


def call_association_provider(
    dataset_id: str,
    nodes: Any,
    t: int,
    *,
    context: PortableCurrentBestContext,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Call the explicit provider and validate its frozen output contract."""

    dataset_id = _validate_dataset_id(
        dataset_id
    )

    result = context.association_provider(
        dataset_id,
        nodes,
        t,
        context=context,
    )

    if not isinstance(
        result,
        (
            tuple,
            list,
        ),
    ):

        raise PortableRuntimeContractError(
            "association_provider must return a 3-item tuple/list."
        )

    if len(result) != 3:

        raise PortableRuntimeContractError(
            "association_provider must return exactly "
            "(source_ids, target_ids, probability_matrix)."
        )

    return _validate_association_matrix(
        result[0],
        result[1],
        result[2],
    )


def reconstruct_mutual_cont_v1_actions(
    dataset_id: str,
    source_node_ids: Sequence[int],
    target_node_ids: Sequence[int],
    candidate_prob_matrix: Any,
    *,
    source_outdegree: Mapping[int, int],
    target_indegree: Mapping[int, int],
) -> tuple[MutualContV1Action, ...]:
    """Reconstruct frozen MUTUAL-CONT-V1 actions.

    Frozen semantics:

    1. source-side Top1 on the current association matrix;
    2. target-side Top1 on the same matrix;
    3. keep reciprocal Top1 pairs;
    4. require candidate probability >= 0.25;
    5. require source final outdegree == 0;
    6. require target final indegree == 0.

    ``numpy.argmax`` supplies deterministic first-occurrence behavior for
    exact ties. Input source/target order is therefore part of the explicit
    association-provider contract and must remain deterministic.

    This function does not consult historical action rows.
    """

    dataset_id = _validate_dataset_id(
        dataset_id
    )

    (
        source_ids,
        target_ids,
        probabilities,
    ) = _validate_association_matrix(
        source_node_ids,
        target_node_ids,
        candidate_prob_matrix,
    )

    if (
        len(source_ids) == 0
        or
        len(target_ids) == 0
    ):

        return tuple()

    row_top = np.argmax(
        probabilities,
        axis=1,
    )

    column_top = np.argmax(
        probabilities,
        axis=0,
    )

    actions: list[
        MutualContV1Action
    ] = []

    used_sources: set[int] = set()
    used_targets: set[int] = set()

    for source_index, target_index in enumerate(
        row_top.tolist()
    ):

        if int(
            column_top[
                target_index
            ]
        ) != source_index:

            continue

        probability = float(
            probabilities[
                source_index,
                target_index,
            ]
        )

        if probability < MUTUAL_CONT_V1_PROB_MIN:

            continue

        source_id = int(
            source_ids[
                source_index
            ]
        )

        target_id = int(
            target_ids[
                target_index
            ]
        )

        try:
            source_degree = int(
                source_outdegree.get(
                    source_id,
                    0,
                )
            )

            target_degree = int(
                target_indegree.get(
                    target_id,
                    0,
                )
            )

        except Exception as exc:

            raise PortableRuntimeContractError(
                "source_outdegree and target_indegree must expose "
                "integer-compatible .get(node_id, 0) values."
            ) from exc

        if source_degree != 0:

            continue

        if target_degree != 0:

            continue

        if source_id in used_sources:

            raise PortableRuntimeContractError(
                "Internal invariant violated: duplicate V1 source action."
            )

        if target_id in used_targets:

            raise PortableRuntimeContractError(
                "Internal invariant violated: duplicate V1 target action."
            )

        used_sources.add(
            source_id
        )

        used_targets.add(
            target_id
        )

        actions.append(
            MutualContV1Action(
                dataset_id=dataset_id,
                source_node_id=source_id,
                target_node_id=target_id,
                candidate_prob=probability,
            )
        )

    return tuple(
        actions
    )


def _build_portable_frozen_v9(
    dataset_id: str,
    *,
    context: PortableCurrentBestContext,
) -> Any:
    """Build the frozen Stage-V9 graph from explicit runtime inputs.

    The policy bundle is supplied explicitly by ``context``. This wrapper
    deliberately does not rebuild or retune the corpus-level policies.
    """

    dataset_id = _validate_dataset_id(
        dataset_id
    )

    expected_prefix = (
        "dataset_name",
        "pred_995_dir",
        "pred_9975_dir",
        "policies",
        "train_dir",
    )

    signature = inspect.signature(
        _frozen_v9.apply_frozen_v9
    )

    parameter_names = tuple(
        signature.parameters
    )

    if (
        parameter_names[
            :5
        ]
        !=
        expected_prefix
    ):

        raise PortableRuntimeContractError(
            "FrozenV9 source signature drift: expected prefix "
            f"{expected_prefix}, got {parameter_names[:5]}."
        )

    return _frozen_v9.apply_frozen_v9(
        dataset_id,
        Path(
            context.b256_pred_995_root
        ),
        Path(
            context.b256_pred_9975_root
        ),
        context.frozen_v9_policy_bundle,
        Path(
            context.dataset_root
        ),
    )


def _apply_portable_morph_div_v2(
    graph: Any,
    dataset_id: str,
    *,
    context: PortableCurrentBestContext,
) -> Any:
    """Reserved MorphDivV2 source-derived runtime boundary."""

    raise PortableImplementationIncompleteError(
        "MorphDivV2 portable source wiring is not yet implemented in 13.3D."
    )


def _reconstruct_portable_mutual_cont_v1(
    graph: Any,
    dataset_id: str,
    *,
    context: PortableCurrentBestContext,
) -> Any:
    """Reconstruct frozen MUTUAL-CONT-V1 from the corrected specification.

    Runtime authority is specification-derived reconstruction only.

    Historical locked V1 actions are forbidden as runtime input. They remain
    available only for a later structural-fidelity replay.

    Frozen scientific policy:
        - reciprocal source/target top-1
        - candidate probability >= 0.25
        - source final outdegree == 0
        - target final indegree == 0
        - add one continuation edge

    The association provider is explicit in ``context`` and is identity-locked
    to the frozen engineering reference provider.
    """

    dataset_id = _validate_dataset_id(
        dataset_id
    )

    if not isinstance(
        context,
        PortableCurrentBestContext,
    ):

        raise PortableRuntimeContractError(
            "context must be PortableCurrentBestContext."
        )

    (
        provider,
        provider_identity,
    ) = _portable_v2_resolve_association_provider(
        context
    )

    if (
        str(
            provider_identity
        )
        !=
        REFERENCE_ASSOCIATION_PROVIDER_SHA256
    ):

        raise PortableRuntimeContractError(
            "MUTUAL-CONT-V1 association-provider identity drift: "
            f"expected {REFERENCE_ASSOCIATION_PROVIDER_SHA256}, "
            f"got {provider_identity}."
        )

    output_graph = _clone_portable_morph_graph(
        graph
    )

    (
        nodes,
        _,
        source_outdegree,
        target_indegree,
    ) = _portable_r1p50_source_graph_state(
        output_graph
    )

    provider_nodes = (
        nodes
        .reset_index()
        .copy(
            deep=True
        )
    )

    node_ids = {
        int(
            value
        )
        for value in (
            nodes.index.tolist()
        )
    }

    times = sorted(
        {
            int(
                value
            )
            for value in (
                provider_nodes[
                    "t"
                ].tolist()
            )
        }
    )

    for t in times:

        result = provider(
            dataset_id,
            provider_nodes.copy(
                deep=True
            ),
            int(
                t
            ),
            context=context,
        )

        (
            source_node_ids,
            target_node_ids,
            candidate_prob_matrix,
        ) = _portable_v2_validate_association_output(
            result
        )

        if (
            candidate_prob_matrix
            is None
        ):

            continue

        missing_sources = sorted(
            {
                int(
                    value
                )
                for value in (
                    source_node_ids
                )
                if int(
                    value
                )
                not in node_ids
            }
        )

        missing_targets = sorted(
            {
                int(
                    value
                )
                for value in (
                    target_node_ids
                )
                if int(
                    value
                )
                not in node_ids
            }
        )

        if (
            missing_sources
            or missing_targets
        ):

            raise PortableRuntimeContractError(
                "MUTUAL-CONT-V1 association provider returned node IDs "
                "outside the current post-Morph graph: "
                f"missing_sources={missing_sources}, "
                f"missing_targets={missing_targets}."
            )

        actions = reconstruct_mutual_cont_v1_actions(
            dataset_id,
            source_node_ids,
            target_node_ids,
            candidate_prob_matrix,
            source_outdegree=source_outdegree,
            target_indegree=target_indegree,
        )

        for action in actions:

            source = int(
                action.source_node_id
            )

            target = int(
                action.target_node_id
            )

            probability = float(
                action.candidate_prob
            )

            if (
                source_outdegree.get(
                    source,
                    0,
                )
                !=
                0
            ):

                continue

            if (
                target_indegree.get(
                    target,
                    0,
                )
                !=
                0
            ):

                continue

            distance = _frozen_v9.recovered_edge_dist(
                nodes,
                source,
                target,
                _frozen_v9.FROZEN_V9_CONFIG,
            )

            output_graph.add_edge(
                source,
                target,
                {
                    "edge_dist":
                        float(
                            distance
                        ),

                    "edge_prob":
                        probability,
                },
                validate_keys=True,
            )

            source_outdegree[
                source
            ] = (
                source_outdegree.get(
                    source,
                    0,
                )
                +
                1
            )

            target_indegree[
                target
            ] = (
                target_indegree.get(
                    target,
                    0,
                )
                +
                1
            )

    return output_graph




def _apply_portable_mutual_cont_v2(
    graph: Any,
    dataset_id: str,
    *,
    context: PortableCurrentBestContext,
) -> Any:
    """Reserved V2 source-derived runtime boundary."""

    raise PortableImplementationIncompleteError(
        "MUTUAL-CONT-V2 portable source wiring is not yet implemented "
        "in 13.3D."
    )


def _apply_portable_det_bridge_r1p50(
    graph: Any,
    dataset_id: str,
    *,
    context: PortableCurrentBestContext,
) -> Any:
    """Reserved R1P50 function-level source-derived runtime boundary."""

    raise PortableImplementationIncompleteError(
        "DET-BRIDGE-R1P50 portable source wiring is not yet implemented "
        "in 13.3D."
    )


def build_portable_pre_g3h3_current_best(
    dataset_id: str,
    *,
    context: PortableCurrentBestContext,
) -> Any:
    """Build the frozen pre-G3H3 graph.

    13.3D deliberately remains fail-closed because source-derived wiring
    for all five historical pipeline stages has not yet been completed.

    This entrypoint must not silently substitute historical action tables.
    """

    _validate_dataset_id(
        dataset_id
    )

    if not isinstance(
        context,
        PortableCurrentBestContext,
    ):

        raise PortableRuntimeContractError(
            "context must be PortableCurrentBestContext."
        )

    raise PortableImplementationIncompleteError(
        "Portable pre-G3H3 pipeline implementation is partial. "
        "FrozenV9 and the MUTUAL-CONT-V1 matrix reconstruction core are "
        "implemented, but MorphDiv/V1 graph wiring/V2/R1P50 and structural "
        "fidelity remain outstanding."
    )


__all__ = [
    "MUTUAL_CONT_V1_PROB_MIN",
    "REFERENCE_ASSOCIATION_PROVIDER_SHA256",
    "MutualContV1Action",
    "PortableCurrentBestContext",
    "PortableCurrentBestError",
    "PortableImplementationIncompleteError",
    "PortableRuntimeContractError",
    "build_portable_pre_g3h3_current_best",
    "call_association_provider",
    "reconstruct_mutual_cont_v1_actions",
]


# BEGIN NOTEBOOK13 13.3D3 MORPH PROVIDER RUNTIME
#
# Engineering-only portable MorphDivV2 runtime boundary.
#
# Scientific authority:
#
#   selection:
#       K3_R3_P100
#
#       morph_rank     <= 3
#       candidate_rank <= 3
#       candidate_prob >= 0.10
#
#   application:
#       source outdegree == 1
#       target indegree == 0
#       source->target absent
#       add source->target with edge_prob=candidate_prob
#
# IMPORTANT:
#
#   No concrete Morph candidate generator is implemented here.
#   No historical Morph action artifact may be used at runtime.
#
# END-STATE CLAIM:
#
#   engineering runtime interface implemented
#   scientific historical structural fidelity NOT verified
#

from typing import Protocol
import copy as _portable_copy
import re as _portable_re

import numpy as _portable_np
import pandas as _portable_pd


MORPH_DIV_V2_CONFIG_ID = "K3_R3_P100"

MORPH_DIV_V2_CANDIDATE_COLUMNS = (
    "dataset",
    "morph_rank",
    "t",
    "source_pred_node_id",
    "target_pred_node_id",
    "candidate_rank",
    "candidate_prob",
)

MORPH_DIV_V2_RUNTIME_ORDER = (
    "dataset",
    "morph_rank",
    "t",
    "source_pred_node_id",
    "target_pred_node_id",
)

_MORPH_DIV_V2_MORPH_RANK_MAX = 3
_MORPH_DIV_V2_CANDIDATE_RANK_MAX = 3
_MORPH_DIV_V2_CANDIDATE_PROB_MIN = 0.10

_MORPH_PROVIDER_ALLOWED_AUTHORITY_STATUS = frozenset(
    {
        "ENGINEERING_UNVERIFIED",
        "HISTORICAL_STRUCTURAL_FIDELITY_PENDING",
        "HISTORICAL_STRUCTURAL_FIDELITY_VERIFIED",
    }
)

_MORPH_PROVIDER_SHA_RE = _portable_re.compile(
    r"^[0-9a-f]{64}$"
)


class MorphProviderContractError(RuntimeError):
    """Fail-closed Morph candidate-provider/runtime contract violation."""


class MorphCandidateProvider(Protocol):
    """
    Engineering interface for GT-blind, PRE-SELECTION Morph candidates.

    A provider is NOT the owner of K3_R3_P100 selection and MUST NOT
    mutate the supplied post-FrozenV9 graph.
    """

    provider_id: str
    provider_version: str
    provider_source_sha256: str
    provider_authority_status: str

    def build_candidates(
        self,
        dataset_id,
        post_frozen_v9_graph,
        *,
        context,
    ):
        ...


def validate_morph_candidate_provider(provider):
    """
    Validate provider identity/provenance.

    Anonymous functions and unidentifiable callables are intentionally
    rejected. This function does NOT certify scientific fidelity.
    """

    if provider is None:
        raise MorphProviderContractError(
            "MorphCandidateProvider is required; "
            "historical-action fallback is forbidden."
        )

    if not hasattr(provider, "build_candidates"):
        raise MorphProviderContractError(
            "Morph candidate provider must expose build_candidates()."
        )

    if not callable(provider.build_candidates):
        raise MorphProviderContractError(
            "provider.build_candidates must be callable."
        )

    required = (
        "provider_id",
        "provider_version",
        "provider_source_sha256",
        "provider_authority_status",
    )

    values = {}

    for field in required:

        if not hasattr(provider, field):
            raise MorphProviderContractError(
                f"provider missing required identity field: {field}"
            )

        value = getattr(
            provider,
            field,
        )

        if not isinstance(value, str) or not value.strip():
            raise MorphProviderContractError(
                f"provider identity field {field} must be a nonempty string"
            )

        values[field] = value.strip()

    if not _MORPH_PROVIDER_SHA_RE.fullmatch(
        values["provider_source_sha256"]
    ):
        raise MorphProviderContractError(
            "provider_source_sha256 must be a lowercase 64-hex SHA256"
        )

    if (
        values["provider_authority_status"]
        not in
        _MORPH_PROVIDER_ALLOWED_AUTHORITY_STATUS
    ):
        raise MorphProviderContractError(
            "unrecognized Morph provider authority status: "
            f"{values['provider_authority_status']}"
        )

    return dict(values)


def _require_integer_like_series(series, name):
    values = series.to_numpy(
        dtype=float,
        copy=False,
    )

    if not _portable_np.isfinite(values).all():
        raise MorphProviderContractError(
            f"{name} contains non-finite values"
        )

    if not _portable_np.equal(
        values,
        _portable_np.floor(values),
    ).all():
        raise MorphProviderContractError(
            f"{name} must be integer-like"
        )

    return values


def validate_morph_candidate_table(
    candidate_table,
    *,
    dataset_id,
):
    """
    Strictly validate PRE-SELECTION Morph candidate rows.

    Selection gates are deliberately NOT applied here.
    """

    if not isinstance(
        candidate_table,
        _portable_pd.DataFrame,
    ):
        raise MorphProviderContractError(
            "Morph provider must return a pandas DataFrame"
        )

    expected = list(
        MORPH_DIV_V2_CANDIDATE_COLUMNS
    )

    actual = list(
        candidate_table.columns
    )

    if actual != expected:
        raise MorphProviderContractError(
            "Morph candidate schema mismatch: "
            f"expected {expected}, got {actual}"
        )

    table = candidate_table.copy(
        deep=True
    )

    if table.isna().any().any():
        raise MorphProviderContractError(
            "Morph candidate table contains null values"
        )

    # Empty table is explicitly allowed, but exact schema is still required.
    if table.empty:
        return table

    if not (
        table["dataset"].astype(str)
        ==
        str(dataset_id)
    ).all():
        raise MorphProviderContractError(
            "Morph candidate dataset identity mismatch"
        )

    numeric_columns = (
        "morph_rank",
        "t",
        "source_pred_node_id",
        "target_pred_node_id",
        "candidate_rank",
        "candidate_prob",
    )

    for column in numeric_columns:

        if not _portable_pd.api.types.is_numeric_dtype(
            table[column]
        ):
            raise MorphProviderContractError(
                f"{column} must be numeric"
            )

        values = table[column].to_numpy(
            dtype=float,
            copy=False,
        )

        if not _portable_np.isfinite(values).all():
            raise MorphProviderContractError(
                f"{column} contains non-finite values"
            )

    morph_rank = _require_integer_like_series(
        table["morph_rank"],
        "morph_rank",
    )

    candidate_rank = _require_integer_like_series(
        table["candidate_rank"],
        "candidate_rank",
    )

    _require_integer_like_series(
        table["t"],
        "t",
    )

    _require_integer_like_series(
        table["source_pred_node_id"],
        "source_pred_node_id",
    )

    _require_integer_like_series(
        table["target_pred_node_id"],
        "target_pred_node_id",
    )

    if (morph_rank < 1).any():
        raise MorphProviderContractError(
            "morph_rank must be >= 1"
        )

    if (candidate_rank < 1).any():
        raise MorphProviderContractError(
            "candidate_rank must be >= 1"
        )

    probs = table["candidate_prob"].to_numpy(
        dtype=float,
        copy=False,
    )

    if (
        (probs < 0.0).any()
        or
        (probs > 1.0).any()
    ):
        raise MorphProviderContractError(
            "candidate_prob must lie in [0, 1]"
        )

    return table


def select_morph_div_v2_candidates(
    candidate_table,
    *,
    dataset_id,
):
    """
    Apply the frozen K3_R3_P100 selection policy.

    No threshold arguments are accepted by design.
    """

    table = validate_morph_candidate_table(
        candidate_table,
        dataset_id=dataset_id,
    )

    if table.empty:
        return table.copy(
            deep=True
        )

    selected = table[
        (table["morph_rank"] <= _MORPH_DIV_V2_MORPH_RANK_MAX)
        &
        (
            table["candidate_rank"]
            <=
            _MORPH_DIV_V2_CANDIDATE_RANK_MAX
        )
        &
        (
            table["candidate_prob"]
            >=
            _MORPH_DIV_V2_CANDIDATE_PROB_MIN
        )
    ].copy()

    selected = (
        selected
        .sort_values(
            list(
                MORPH_DIV_V2_RUNTIME_ORDER
            ),
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    return selected


def _portable_table_to_pandas(value):
    if isinstance(
        value,
        _portable_pd.DataFrame,
    ):
        return value.copy(
            deep=True
        )

    if hasattr(value, "to_pandas"):
        converted = value.to_pandas()

        if not isinstance(
            converted,
            _portable_pd.DataFrame,
        ):
            raise MorphProviderContractError(
                "graph table to_pandas() did not return DataFrame"
            )

        return converted.copy(
            deep=True
        )

    raise MorphProviderContractError(
        "unsupported graph table object; pandas/to_pandas required"
    )


def _portable_morph_node_table(graph):
    if not hasattr(
        graph,
        "node_attrs",
    ):
        raise MorphProviderContractError(
            "graph does not expose node_attrs()"
        )

    try:
        raw = graph.node_attrs(
            attr_keys=[
                "node_id",
            ]
        )
    except TypeError:
        raw = graph.node_attrs(
            [
                "node_id",
            ]
        )

    nodes = _portable_table_to_pandas(
        raw
    )

    if "node_id" not in nodes.columns:
        raise MorphProviderContractError(
            "graph node table lacks node_id"
        )

    return nodes


def _portable_morph_edge_table(graph):
    if not hasattr(
        graph,
        "edge_attrs",
    ):
        raise MorphProviderContractError(
            "graph does not expose edge_attrs()"
        )

    requested = [
        "source",
        "target",
        "edge_prob",
    ]

    try:
        raw = graph.edge_attrs(
            attr_keys=requested
        )
    except Exception:
        try:
            raw = graph.edge_attrs(
                requested
            )
        except Exception as exc:
            raise MorphProviderContractError(
                "unable to read graph edge source/target/edge_prob"
            ) from exc

    edges = _portable_table_to_pandas(
        raw
    )

    aliases = {
        "source_node_id":
            "source",

        "target_node_id":
            "target",
    }

    for old, new in aliases.items():
        if (
            new not in edges.columns
            and
            old in edges.columns
        ):
            edges = edges.rename(
                columns={
                    old:
                        new
                }
            )

    required = {
        "source",
        "target",
    }

    if not required <= set(
        edges.columns
    ):
        raise MorphProviderContractError(
            "graph edge table lacks source/target"
        )

    if "edge_prob" not in edges.columns:
        # edge_prob is needed for mutation fingerprints.
        # Existing historical edges without a stored edge_prob are
        # represented by NaN only in the private fingerprint copy.
        # No runtime graph value is changed.
        edges = edges.copy()

        edges["edge_prob"] = _portable_np.nan

    return edges[
        [
            "source",
            "target",
            "edge_prob",
        ]
    ].copy()


def _portable_morph_graph_state(graph):
    """
    Structural state needed by frozen Morph application semantics.
    """

    nodes = _portable_morph_node_table(
        graph
    )

    edges = _portable_morph_edge_table(
        graph
    )

    node_ids = {
        int(value)

        for value in nodes[
            "node_id"
        ].tolist()
    }

    outdegree = {
        node_id:
            0

        for node_id in node_ids
    }

    indegree = {
        node_id:
            0

        for node_id in node_ids
    }

    edge_set = set()

    for row in edges.itertuples(
        index=False
    ):
        source = int(
            row.source
        )

        target = int(
            row.target
        )

        edge_set.add(
            (
                source,
                target,
            )
        )

        outdegree[source] = (
            outdegree.get(
                source,
                0,
            )
            +
            1
        )

        indegree[target] = (
            indegree.get(
                target,
                0,
            )
            +
            1
        )

    return {
        "node_ids":
            node_ids,

        "edge_set":
            edge_set,

        "outdegree":
            outdegree,

        "indegree":
            indegree,
    }


def _portable_morph_graph_fingerprint(graph):
    """
    Detect provider topology/edge-probability mutation.

    This is an engineering invariant, not a scientific metric.
    """

    nodes = _portable_morph_node_table(
        graph
    )

    edges = _portable_morph_edge_table(
        graph
    )

    node_ids = tuple(
        sorted(
            int(value)

            for value in nodes[
                "node_id"
            ].tolist()
        )
    )

    edge_records = []

    for row in edges.itertuples(
        index=False
    ):
        prob = row.edge_prob

        if _portable_pd.isna(
            prob
        ):
            prob_value = None
        else:
            prob_value = float(
                prob
            )

        edge_records.append(
            (
                int(
                    row.source
                ),
                int(
                    row.target
                ),
                prob_value,
            )
        )

    edge_records = tuple(
        sorted(
            edge_records,
            key=lambda value: (
                value[0],
                value[1],
                (
                    -1.0
                    if value[2] is None
                    else value[2]
                ),
            ),
        )
    )

    return (
        node_ids,
        edge_records,
    )


def _clone_portable_morph_graph(graph):
    """
    Clone graph so provider cannot mutate the real pipeline graph.

    Fail closed if graph cannot be cloned.
    """

    try:
        clone = _portable_copy.deepcopy(
            graph
        )
    except Exception as exc:
        raise MorphProviderContractError(
            "unable to clone graph for fail-closed Morph provider call"
        ) from exc

    return clone


def _portable_morph_add_edge(
    graph,
    *,
    source,
    target,
    edge_prob,
):
    """
    Add the frozen Morph source->target edge.

    Keyword form is intentionally explicit and fail-closed.
    """

    if not hasattr(
        graph,
        "add_edge",
    ):
        raise MorphProviderContractError(
            "graph does not expose add_edge()"
        )

    try:
        graph.add_edge(
            source=int(
                source
            ),
            target=int(
                target
            ),
            edge_prob=float(
                edge_prob
            ),
        )
    except Exception as exc:
        raise MorphProviderContractError(
            "Morph graph add_edge failed"
        ) from exc


def apply_portable_morph_div_v2(
    dataset_id,
    post_frozen_v9_graph,
    *,
    context,
    provider,
    return_audit=False,
):
    """
    Engineering implementation of the frozen Morph runtime boundary.

    Pipeline:
        provider
        -> strict validation
        -> K3_R3_P100 selection
        -> deterministic ordering
        -> frozen application semantics

    This function does NOT certify provider historical equivalence.
    """

    provider_identity = validate_morph_candidate_provider(
        provider
    )

    provider_graph = _clone_portable_morph_graph(
        post_frozen_v9_graph
    )

    provider_before = _portable_morph_graph_fingerprint(
        provider_graph
    )

    candidate_table = provider.build_candidates(
        dataset_id,
        provider_graph,
        context=context,
    )

    provider_after = _portable_morph_graph_fingerprint(
        provider_graph
    )

    if provider_before != provider_after:
        raise MorphProviderContractError(
            "Morph candidate provider mutated its graph input"
        )

    candidate_table = validate_morph_candidate_table(
        candidate_table,
        dataset_id=dataset_id,
    )

    selected = select_morph_div_v2_candidates(
        candidate_table,
        dataset_id=dataset_id,
    )

    graph = _clone_portable_morph_graph(
        post_frozen_v9_graph
    )

    state = _portable_morph_graph_state(
        graph
    )

    accepted_records = []

    for row in selected.itertuples(
        index=False
    ):
        source = int(
            row.source_pred_node_id
        )

        target = int(
            row.target_pred_node_id
        )

        candidate_prob = float(
            row.candidate_prob
        )

        if (
            source
            not in
            state["node_ids"]
        ):
            raise MorphProviderContractError(
                f"Morph source node not present in graph: {source}"
            )

        if (
            target
            not in
            state["node_ids"]
        ):
            raise MorphProviderContractError(
                f"Morph target node not present in graph: {target}"
            )

        # Frozen R3 Morph application semantics.
        if (
            state[
                "outdegree"
            ].get(
                source,
                0,
            )
            !=
            1
        ):
            continue

        if (
            state[
                "indegree"
            ].get(
                target,
                0,
            )
            !=
            0
        ):
            continue

        if (
            source,
            target,
        ) in state[
            "edge_set"
        ]:
            continue

        _portable_morph_add_edge(
            graph,
            source=source,
            target=target,
            edge_prob=candidate_prob,
        )

        state[
            "edge_set"
        ].add(
            (
                source,
                target,
            )
        )

        state[
            "outdegree"
        ][
            source
        ] = (
            state[
                "outdegree"
            ].get(
                source,
                0,
            )
            +
            1
        )

        state[
            "indegree"
        ][
            target
        ] = (
            state[
                "indegree"
            ].get(
                target,
                0,
            )
            +
            1
        )

        accepted_records.append(
            {
                "dataset":
                    str(
                        row.dataset
                    ),

                "morph_rank":
                    int(
                        row.morph_rank
                    ),

                "t":
                    int(
                        row.t
                    ),

                "source_pred_node_id":
                    source,

                "target_pred_node_id":
                    target,

                "candidate_rank":
                    int(
                        row.candidate_rank
                    ),

                "candidate_prob":
                    candidate_prob,
            }
        )

    accepted = _portable_pd.DataFrame(
        accepted_records,
        columns=list(
            MORPH_DIV_V2_CANDIDATE_COLUMNS
        ),
    )

    if not accepted.empty:
        accepted = (
            accepted
            .sort_values(
                list(
                    MORPH_DIV_V2_RUNTIME_ORDER
                ),
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

    if return_audit:

        return {
            "graph":
                graph,

            "provider_identity":
                provider_identity,

            "candidate_count":
                int(
                    len(
                        candidate_table
                    )
                ),

            "selected_count":
                int(
                    len(
                        selected
                    )
                ),

            "accepted_count":
                int(
                    len(
                        accepted
                    )
                ),

            "selected_actions":
                selected,

            "accepted_actions":
                accepted,

            "config_id":
                MORPH_DIV_V2_CONFIG_ID,

            "historical_structural_fidelity_verified":
                (
                    provider_identity[
                        "provider_authority_status"
                    ]
                    ==
                    "HISTORICAL_STRUCTURAL_FIDELITY_VERIFIED"
                ),
        }

    return graph


# END NOTEBOOK13 13.3D3 MORPH PROVIDER RUNTIME


# BEGIN NOTEBOOK13 13.3D4-R2R3 MUTUAL-CONT-V2 PORTABLE RUNTIME

MUTUAL_CONT_V2_CANDIDATE_PROB_MIN = 0.25

_MUTUAL_CONT_V2_ACTION_LOCK_SOURCE_SHA256 = "8e2684669baeaf8bc0d4b889f5c6e213a9127cf09e36c3d76ea2b3d77c2f4ef5"

_MUTUAL_CONT_V2_RECIPROCAL_SOURCE_SHA256 = "d082c4776cece9efb61f6d3bbe9fa91659036525daf670dd6564dc323d54074c"

_MUTUAL_CONT_V2_FAST_ASSOCIATION_REFERENCE_SHA256 = "7aae212c3ae52e1070925386fb2775feec0c964337ec48441f1da02d7287a1b1"


class MutualContV2ContractError(RuntimeError):
    """Fail-closed portable MUTUAL-CONT-V2 runtime contract error."""


def _portable_v2_source_residual_reciprocal_pairs(src_ids, tgt_ids, probs, free_sources, free_targets):
    """
    Round-2 reciprocal matching.

    The rank is recomputed only inside the remaining deployment-
    valid free-source × free-target submatrix.

    Returns:
        (source_id, target_id, original_probability)
    """
    if probs is None:
        return []
    src_idx = _portable_np.asarray([i for i, node_id in enumerate(src_ids) if int(node_id) in free_sources], dtype=int)
    tgt_idx = _portable_np.asarray([j for j, node_id in enumerate(tgt_ids) if int(node_id) in free_targets], dtype=int)
    if len(src_idx) == 0 or len(tgt_idx) == 0:
        return []
    sub = probs[_portable_np.ix_(src_idx, tgt_idx)]
    row_top = _portable_np.argmax(sub, axis=1)
    col_top = _portable_np.argmax(sub, axis=0)
    pairs = []
    for a, b in enumerate(row_top):
        if int(col_top[b]) != a:
            continue
        i = int(src_idx[a])
        j = int(tgt_idx[b])
        pairs.append((int(src_ids[i]), int(tgt_ids[j]), float(probs[i, j])))
    return pairs


def _portable_v2_node_table(graph):

    if not hasattr(
        graph,
        "node_attrs",
    ):
        raise MutualContV2ContractError(
            "graph does not expose node_attrs()"
        )

    required = [
        "node_id",
        "t",
        "z",
        "y",
        "x",
    ]

    try:
        raw = graph.node_attrs(
            attr_keys=required
        )

    except TypeError:
        raw = graph.node_attrs(
            required
        )

    table = _portable_table_to_pandas(
        raw
    )

    missing = [
        column
        for column
        in required
        if column not in table.columns
    ]

    if missing:
        raise MutualContV2ContractError(
            "V2 node table missing columns: "
            + repr(missing)
        )

    table = table[
        required
    ].copy()

    if table.isna().any().any():
        raise MutualContV2ContractError(
            "V2 node table contains null values"
        )

    for column in required:

        if not _portable_pd.api.types.is_numeric_dtype(
            table[column]
        ):
            raise MutualContV2ContractError(
                f"V2 node column {column} must be numeric"
            )

        values = table[
            column
        ].to_numpy(
            dtype=float,
            copy=False,
        )

        if not _portable_np.isfinite(
            values
        ).all():
            raise MutualContV2ContractError(
                f"V2 node column {column} contains non-finite values"
            )

    for column in [
        "node_id",
        "t",
    ]:

        values = table[
            column
        ].to_numpy(
            dtype=float,
            copy=False,
        )

        if not _portable_np.equal(
            values,
            _portable_np.floor(
                values
            ),
        ).all():
            raise MutualContV2ContractError(
                f"V2 {column} must be integer-like"
            )

    return table


def _portable_v2_resolve_association_provider(
    context,
):

    if context is None:
        raise MutualContV2ContractError(
            "V2 requires explicit runtime context"
        )

    if not hasattr(
        context,
        "association_provider",
    ):
        raise MutualContV2ContractError(
            "runtime context lacks association_provider"
        )

    if not hasattr(
        context,
        "association_provider_identity_sha256",
    ):
        raise MutualContV2ContractError(
            "runtime context lacks association_provider_identity_sha256"
        )

    provider = (
        context.association_provider
    )

    if not callable(
        provider
    ):
        raise MutualContV2ContractError(
            "association_provider must be callable"
        )

    identity = str(
        context.association_provider_identity_sha256
    )

    if (
        len(identity)
        !=
        64
        or
        any(
            character
            not in
            "0123456789abcdef"
            for character
            in identity
        )
    ):
        raise MutualContV2ContractError(
            "association_provider_identity_sha256 "
            "must be lowercase 64-hex"
        )

    return (
        provider,
        identity,
    )


def _portable_v2_validate_association_output(
    result,
):

    if (
        not isinstance(
            result,
            tuple,
        )
        or
        len(result)
        !=
        3
    ):
        raise MutualContV2ContractError(
            "association_provider must return "
            "(source_ids, target_ids, candidate_prob_matrix)"
        )

    source_ids, target_ids, probs = result

    source_ids = _portable_np.asarray(
        source_ids
    )

    target_ids = _portable_np.asarray(
        target_ids
    )

    if source_ids.ndim != 1:
        raise MutualContV2ContractError(
            "source_ids must be one-dimensional"
        )

    if target_ids.ndim != 1:
        raise MutualContV2ContractError(
            "target_ids must be one-dimensional"
        )

    source_values = source_ids.astype(
        float
    )

    target_values = target_ids.astype(
        float
    )

    if not _portable_np.isfinite(
        source_values
    ).all():
        raise MutualContV2ContractError(
            "source_ids contain non-finite values"
        )

    if not _portable_np.isfinite(
        target_values
    ).all():
        raise MutualContV2ContractError(
            "target_ids contain non-finite values"
        )

    if not _portable_np.equal(
        source_values,
        _portable_np.floor(
            source_values
        ),
    ).all():
        raise MutualContV2ContractError(
            "source_ids must be integer-like"
        )

    if not _portable_np.equal(
        target_values,
        _portable_np.floor(
            target_values
        ),
    ).all():
        raise MutualContV2ContractError(
            "target_ids must be integer-like"
        )

    source_ids = source_values.astype(
        int
    )

    target_ids = target_values.astype(
        int
    )

    if probs is None:
        return (
            source_ids,
            target_ids,
            None,
        )

    probs = _portable_np.asarray(
        probs,
        dtype=float,
    )

    if probs.ndim != 2:
        raise MutualContV2ContractError(
            "candidate probability matrix must be two-dimensional"
        )

    if probs.shape != (
        len(source_ids),
        len(target_ids),
    ):
        raise MutualContV2ContractError(
            "candidate probability matrix shape mismatch"
        )

    if not _portable_np.isfinite(
        probs
    ).all():
        raise MutualContV2ContractError(
            "candidate probability matrix contains non-finite values"
        )

    if (
        (probs < 0.0).any()
        or
        (probs > 1.0).any()
    ):
        raise MutualContV2ContractError(
            "candidate probabilities must lie in [0, 1]"
        )

    return (
        source_ids,
        target_ids,
        probs,
    )


def build_portable_mutual_cont_v2_actions(
    dataset_id,
    post_v1_graph,
    *,
    context,
):
    """
    GT-blind source-derived MUTUAL-CONT-V2 engineering runtime.

    Frozen gate:
        reject if p < 0.25

    Structural fidelity remains unverified.
    """

    provider, provider_identity = (
        _portable_v2_resolve_association_provider(
            context
        )
    )

    nodes = _portable_v2_node_table(
        post_v1_graph
    )

    state = _portable_morph_graph_state(
        post_v1_graph
    )

    node_ids = set(
        int(value)
        for value
        in nodes[
            "node_id"
        ].tolist()
    )

    times = sorted(
        set(
            int(value)
            for value
            in nodes[
                "t"
            ].tolist()
        )
    )

    records = []

    for t in times:

        result = provider(
            dataset_id,
            nodes.copy(
                deep=True
            ),
            int(t),
            context=context,
        )

        (
            source_ids,
            target_ids,
            probs,
        ) = _portable_v2_validate_association_output(
            result
        )

        if probs is None:
            continue

        free_sources = {
            int(source)
            for source
            in source_ids
            if (
                int(source)
                in node_ids
                and
                state[
                    "outdegree"
                ].get(
                    int(source),
                    0,
                )
                ==
                0
            )
        }

        free_targets = {
            int(target)
            for target
            in target_ids
            if (
                int(target)
                in node_ids
                and
                state[
                    "indegree"
                ].get(
                    int(target),
                    0,
                )
                ==
                0
            )
        }

        pairs = (
            _portable_v2_source_residual_reciprocal_pairs(
                source_ids,
                target_ids,
                probs,
                free_sources,
                free_targets,
            )
        )

        for source, target, probability in pairs:

            probability = float(
                probability
            )

            if (
                probability
                <
                MUTUAL_CONT_V2_CANDIDATE_PROB_MIN
            ):
                continue

            records.append(
                {
                    "dataset":
                        str(
                            dataset_id
                        ),

                    "t":
                        int(
                            t
                        ),

                    "source_pred_node_id":
                        int(
                            source
                        ),

                    "target_pred_node_id":
                        int(
                            target
                        ),

                    "candidate_prob":
                        probability,

                    "round2_row_rank":
                        1,

                    "round2_column_rank":
                        1,
                }
            )

    actions = _portable_pd.DataFrame(
        records,
        columns=[
            "dataset",
            "t",
            "source_pred_node_id",
            "target_pred_node_id",
            "candidate_prob",
            "round2_row_rank",
            "round2_column_rank",
        ],
    )

    if not actions.empty:

        actions = (
            actions
            .reset_index(
                drop=True
            )
        )

    actions.attrs[
        "association_provider_identity_sha256"
    ] = provider_identity

    actions.attrs[
        "structural_fidelity_verified"
    ] = False

    return actions


def apply_portable_mutual_cont_v2(
    dataset_id,
    post_v1_graph,
    *,
    context,
    return_audit=False,
):

    actions = (
        build_portable_mutual_cont_v2_actions(
            dataset_id,
            post_v1_graph,
            context=context,
        )
    )

    graph = _clone_portable_morph_graph(
        post_v1_graph
    )

    state = _portable_morph_graph_state(
        graph
    )

    accepted = []

    for row in actions.itertuples(
        index=False
    ):

        source = int(
            row.source_pred_node_id
        )

        target = int(
            row.target_pred_node_id
        )

        probability = float(
            row.candidate_prob
        )

        if source not in state[
            "node_ids"
        ]:
            raise MutualContV2ContractError(
                f"V2 source node absent: {source}"
            )

        if target not in state[
            "node_ids"
        ]:
            raise MutualContV2ContractError(
                f"V2 target node absent: {target}"
            )

        if (
            state[
                "outdegree"
            ].get(
                source,
                0,
            )
            !=
            0
        ):
            continue

        if (
            state[
                "indegree"
            ].get(
                target,
                0,
            )
            !=
            0
        ):
            continue

        if (
            source,
            target,
        ) in state[
            "edge_set"
        ]:
            continue

        _portable_morph_add_edge(
            graph,
            source=source,
            target=target,
            edge_prob=probability,
        )

        state[
            "edge_set"
        ].add(
            (
                source,
                target,
            )
        )

        state[
            "outdegree"
        ][
            source
        ] = (
            state[
                "outdegree"
            ].get(
                source,
                0,
            )
            +
            1
        )

        state[
            "indegree"
        ][
            target
        ] = (
            state[
                "indegree"
            ].get(
                target,
                0,
            )
            +
            1
        )

        accepted.append(
            {
                "dataset":
                    str(
                        row.dataset
                    ),

                "t":
                    int(
                        row.t
                    ),

                "source_pred_node_id":
                    source,

                "target_pred_node_id":
                    target,

                "candidate_prob":
                    probability,
            }
        )

    accepted_table = _portable_pd.DataFrame(
        accepted,
        columns=[
            "dataset",
            "t",
            "source_pred_node_id",
            "target_pred_node_id",
            "candidate_prob",
        ],
    )

    if return_audit:

        return {
            "graph":
                graph,

            "generated_actions":
                actions,

            "accepted_actions":
                accepted_table,

            "generated_action_count":
                int(
                    len(
                        actions
                    )
                ),

            "accepted_action_count":
                int(
                    len(
                        accepted_table
                    )
                ),

            "candidate_prob_min":
                MUTUAL_CONT_V2_CANDIDATE_PROB_MIN,

            "association_provider_identity_sha256":
                actions.attrs.get(
                    "association_provider_identity_sha256"
                ),

            "structural_fidelity_verified":
                False,
        }

    return graph


# END NOTEBOOK13 13.3D4-R2R3 MUTUAL-CONT-V2 PORTABLE RUNTIME


# BEGIN NOTEBOOK13 13.3D4-R2R3 R1P50 PORTABLE SOURCE PACKAGE

_R1P50_HISTORICAL_SOURCE_SHA256 = "5ada61f1ee4a090dd6f22d5eaa38adaeacba20e21e6b59d09616c7a1f8dfe1cc"

_R1P50_APPLY_SOURCE_SHA256 = "91bd2463f833e1e7a31623c51d73092c978aef1c35001115d810db4ebc2cffe8"

_R1P50_PHYSICAL_DISTANCE_SOURCE_SHA256 = "bdfc076ebfd8db3e5cc20e7145f170c23f1fc596543136f634194012d1c19dbb"

_R1P50_HISTORICAL_SCALE_ASSIGNMENT_SHA256 = "7b7c5f4da3eefa438c3b7b705753c761f2eaf33ce6f7acad1b838b7490a18407"


import numpy as np

def _portable_r1p50_source_graph_state(g):
    nodes = g.node_attrs(attr_keys=['node_id', 't', 'z', 'y', 'x']).to_pandas().set_index('node_id')
    edges = g.edge_attrs(unpack=True).to_pandas()
    outdeg = edges.groupby('source_id').size().to_dict()
    indeg = edges.groupby('target_id').size().to_dict()
    return (nodes, edges, outdeg, indeg)

def _portable_r1p50_source_physical_distance(a, b, *, scale):
    aa = np.asarray([float(a['z']), float(a['y']), float(a['x'])])
    bb = np.asarray([float(b['z']), float(b['y']), float(b['x'])])
    return float(np.linalg.norm((aa - bb) * scale))

def apply_portable_det_bridge_r1p50_core(g, name, *, eligible, scale):
    aa = eligible[eligible['dataset'].eq(name)].sort_values(['bridge_min_prob', 'detector_prob', 'midpoint_residual_um', 't', 'z_ds', 'y_ds', 'x_ds', 'nearest_source_id', 'nearest_target_id'], ascending=[False, False, True, True, True, True, True, True, True], kind='mergesort').reset_index(drop=True)
    nodes, _, outdeg, indeg = _portable_r1p50_source_graph_state(g)
    used_sources = set()
    used_targets = set()
    selected = []
    for r in aa.itertuples(index=False):
        s = int(r.nearest_source_id)
        q = int(r.nearest_target_id)
        assert outdeg.get(s, 0) == 0
        assert indeg.get(q, 0) == 0
        if s in used_sources:
            continue
        if q in used_targets:
            continue
        src = nodes.loc[s]
        tgt = nodes.loc[q]
        assert int(src['t']) == int(r.t) - 1
        assert int(tgt['t']) == int(r.t) + 1
        attrs = {'t': int(r.t), 'z': float(r.z), 'y': float(r.y), 'x': float(r.x)}
        in_dist = _portable_r1p50_source_physical_distance(src, attrs, scale=scale)
        out_dist = _portable_r1p50_source_physical_distance(attrs, tgt, scale=scale)
        assert np.isclose(in_dist, float(r.src_dist_um), atol=1e-06)
        assert np.isclose(out_dist, float(r.tgt_dist_um), atol=1e-06)
        new_id = g.add_node(attrs, validate_keys=True)
        g.add_edge(s, new_id, {'edge_dist': in_dist, 'edge_prob': float(r.in_prob)}, validate_keys=True)
        g.add_edge(new_id, q, {'edge_dist': out_dist, 'edge_prob': float(r.out_prob)}, validate_keys=True)
        used_sources.add(s)
        used_targets.add(q)
        selected.append({'dataset': name, 'source_id': s, 'target_id': q, 'new_node_id': int(new_id), 't': int(r.t), 'z': float(r.z), 'y': float(r.y), 'x': float(r.x), 'detector_prob': float(r.detector_prob), 'bridge_min_prob': float(r.bridge_min_prob), 'in_prob': float(r.in_prob), 'out_prob': float(r.out_prob), 'src_dist_um': float(r.src_dist_um), 'tgt_dist_um': float(r.tgt_dist_um), 'midpoint_residual_um': float(r.midpoint_residual_um)})
    return (selected, len(aa))


# END NOTEBOOK13 13.3D4-R2R3 R1P50 PORTABLE SOURCE PACKAGE


# === P995_BASELINE_NODE_STATE_PROVIDER_V1 ===

from dataclasses import dataclass as _p995_dataclass
import importlib as _p995_importlib
from pathlib import Path as _p995_Path
from typing import Any as _p995_Any

import numpy as _p995_np


P995_BASELINE_NODE_STATE_PROVIDER_ID = (
    "P995_BASELINE_NODE_STATE_PROVIDER_V1"
)

P995_BASELINE_PREDICT_AUTHORITY_MODULE = (
    "scripts.predict_unet_transformer_mps"
)

P995_BASELINE_PREDICT_SOURCE_SHA256 = (
    "23b76f7c90ffe698e955efee2b4333277a07f5b58b7eff6e580f2bb2ff49bc55"
)

P995_BASELINE_LOAD_MODEL_SOURCE_SHA256 = (
    "bbabeb0d3116689002ca9f9c955fba716c145282f08ed104e67ab0764e616e1b"
)

P995_BASELINE_PROVIDER_AUTHORITY_STATUS = (
    "SOURCE_IMPLEMENTED_STRUCTURAL_FIDELITY_PENDING"
)

P995_BASELINE_NODE_COLUMNS = (
    "t",
    "z",
    "y",
    "x",
)

P995_BASELINE_REQUIRED_CFG_FIELDS = (
    "det_threshold",
    "det_tta",
    "edge_activation",
    "ilp_appearance_weight",
    "ilp_disappearance_weight",
    "ilp_division_weight",
    "ilp_edge_weight",
    "max_children_per_node",
    "max_parents_per_node",
    "pool_kernel_um",
    "threshold",
    "use_ilp",
)


@_p995_dataclass(frozen=True)
class P995BaselineNodeStateProviderContext:
    """Frozen fresh-runtime inputs for the P995 baseline provider."""

    data_root: _p995_Path
    weights_path: _p995_Path
    cfg: _p995_Any
    max_frames: int | None = None
    unet_batch_size: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "data_root",
            _p995_Path(self.data_root),
        )
        object.__setattr__(
            self,
            "weights_path",
            _p995_Path(self.weights_path),
        )

        if (
            not isinstance(
                self.unet_batch_size,
                int,
            )
            or
            self.unet_batch_size <= 0
        ):
            raise ValueError(
                "unet_batch_size must be a positive integer"
            )

        if (
            self.max_frames is not None
            and
            (
                not isinstance(
                    self.max_frames,
                    int,
                )
                or
                self.max_frames <= 0
            )
        ):
            raise ValueError(
                "max_frames must be None or a positive integer"
            )


P995_BASELINE_IMPORT_ROOT_RELATIVE_PATHS = (
    ".",
    "scripts",
    "external/official/scripts",
    "external/official/src",
)


def _load_p995_predict_authority_module():
    """Load the frozen historical inference authority portably.

    Historical inference source depends on both package imports
    and bare repo-local imports. Exact repo-local import roots
    are frozen here so behavior does not depend on notebook cwd
    or ambient PYTHONPATH.
    """

    import sys as _p995_runtime_sys

    project_root = (
        _p995_Path(__file__)
        .resolve()
        .parents[2]
    )

    import_roots = []

    for relative_root in (
        P995_BASELINE_IMPORT_ROOT_RELATIVE_PATHS
    ):
        root = (
            project_root
            if relative_root == "."
            else (
                project_root
                /
                relative_root
            ).resolve()
        )

        import_roots.append(
            root
        )

    missing = [
        str(root)
        for root
        in import_roots
        if not root.exists()
    ]

    if missing:
        raise RuntimeError(
            "P995 frozen repo-local import roots are missing: "
            + ", ".join(missing)
        )

    # Reverse insertion preserves the frozen priority order.
    for root in reversed(
        import_roots
    ):
        root_text = str(root)

        if (
            root_text
            in
            _p995_runtime_sys.path
        ):
            _p995_runtime_sys.path.remove(
                root_text
            )

        _p995_runtime_sys.path.insert(
            0,
            root_text,
        )

    return _p995_importlib.import_module(
        P995_BASELINE_PREDICT_AUTHORITY_MODULE
    )



def _validate_p995_cfg(cfg: _p995_Any) -> None:
    missing = [
        field
        for field in P995_BASELINE_REQUIRED_CFG_FIELDS
        if not hasattr(cfg, field)
    ]

    if missing:
        raise TypeError(
            "P995 cfg is missing frozen required fields: "
            + ", ".join(missing)
        )


def _p995_source_derived_device(authority):
    """Exact source-derived CUDA -> MPS -> CPU priority."""

    torch = authority.torch

    return (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
            else torch.device("cpu")
        )
    )


def baseline_node_state_provider(
    dataset_id: str,
    *,
    context: P995BaselineNodeStateProviderContext,
) -> _p995_np.ndarray:
    """Materialize the fresh GT-blind P995 baseline node state."""

    if (
        not isinstance(dataset_id, str)
        or
        not dataset_id
    ):
        raise TypeError(
            "dataset_id must be a non-empty string"
        )

    if not isinstance(
        context,
        P995BaselineNodeStateProviderContext,
    ):
        raise TypeError(
            "context must be "
            "P995BaselineNodeStateProviderContext"
        )

    _validate_p995_cfg(
        context.cfg
    )

    authority = (
        _load_p995_predict_authority_module()
    )

    device = (
        _p995_source_derived_device(
            authority
        )
    )

    model, window_size, downsample = (
        authority.load_model(
            context.weights_path,
            device,
        )
    )

    dataset_path = (
        context.data_root
        / dataset_id
    )

    coords, edges = (
        authority.predict_video(
            model,
            dataset_path,
            device,
            cfg=context.cfg,
            window_size=window_size,
            unet_batch_size=(
                context.unet_batch_size
            ),
            downsample=downsample,
            max_frames=(
                context.max_frames
            ),
        )
    )

    graph = (
        authority.build_graph(
            coords,
            edges,
        )
    )

    if (
        context.cfg.use_ilp
        and
        graph.num_edges() > 0
    ):
        solver = (
            authority.td.solvers.ILPSolver(
                edge_weight=(
                    context.cfg.ilp_edge_weight
                    *
                    authority.td.EdgeAttr(
                        "edge_prob"
                    )
                ),
                appearance_weight=(
                    context.cfg.ilp_appearance_weight
                ),
                disappearance_weight=(
                    context.cfg.ilp_disappearance_weight
                ),
                division_weight=(
                    context.cfg.ilp_division_weight
                ),
            )
        )

        with authority.suppress_output():
            graph = solver.solve(
                graph
            )

    node_table = (
        graph
        .node_attrs(
            attr_keys=list(
                P995_BASELINE_NODE_COLUMNS
            )
        )
        .to_pandas()
        [list(
            P995_BASELINE_NODE_COLUMNS
        )]
        .to_numpy()
        .astype(
            _p995_np.int64
        )
    )

    if (
        node_table.ndim != 2
        or
        node_table.shape[1] != 4
    ):
        raise RuntimeError(
            "P995 baseline provider produced invalid "
            f"node table shape: {node_table.shape}"
        )

    return node_table
