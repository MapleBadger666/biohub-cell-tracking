"""Source-derived suppressed-peak fresh provider.

This module does not reimplement the suppressed-peak algorithm.

At runtime it reads the source-locked canonical writer,
mechanically transforms only the frozen orchestration boundaries,
and executes the canonical source prefix through construction of
``universe``.

Scientific / runtime boundary substitutions:

1. historical fold0 dataset enumeration -> explicit dataset_id
2. historical P995 GEFF graph -> fresh structurally-verified
   P995 node-state provider
3. historical output locations -> temporary isolated paths

All detector/model/peak extraction logic remains sourced from the
canonical writer AST.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import ast
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from biohub_cell_tracking.stage12_portable_current_best import (
    P995BaselineNodeStateProviderContext,
    _load_p995_predict_authority_module,
    baseline_node_state_provider as _baseline_node_state_provider,
)


SUPPRESSED_PEAK_PROVIDER_ID = (
    "SUPPRESSED_PEAK_SOURCE_DERIVED_PROVIDER_V1"
)

SUPPRESSED_PEAK_CANONICAL_WRITER_RELATIVE_PATH = (
    "scripts/run_stage12_fold0_suppressed_peak_universe.py"
)

SUPPRESSED_PEAK_CANONICAL_WRITER_SHA256 = (
    "35843437ccc783cb07751efbb9d6d241b411a3c22cc6bb294928e2c62c22f3fc"
)

SUPPRESSED_PEAK_EXTRACT_PEAKS_SHA256 = (
    "56cc697bc41121124e0ae8c634860f9205c7591d5e69a4edd22e0c41727fd47a"
)

SUPPRESSED_PEAK_DATASET_LOOP_SHA256 = (
    "0fcb9c80c8d9ebce51ecb8d8b671e09155596fb082c1761a75fa4231b3ebe152"
)

SUPPRESSED_PEAK_HISTORICAL_P995_GRAPH_ASSIGNMENT_SHA256 = (
    "45827f639b117d41a4950c8f3adccdc2eab2ac1b1fe1f2e9f4fe059e3743fc1d"
)

SUPPRESSED_PEAK_UNIVERSE_ASSIGNMENT_SHA256 = (
    "51b0bf467e56b9623b3f59e9419d2703fff8b1bca783e42c0e44f102b6ad6806"
)

SUPPRESSED_PEAK_CANONICAL_UNIVERSE_LINE = (
    810
)

SUPPRESSED_PEAK_ALLOWED_HISTORICAL_GRAPH_ATTRIBUTES = (
    "node_attrs",
)

# These names are orchestration/runtime boundaries only.
# Their canonical writer assignments are removed and fresh values
# are injected into the execution namespace.
_SUPPRESSED_CONTROLLED_ASSIGNMENT_NAMES = frozenset(
    {
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
    }
)


@dataclass(frozen=True)
class SuppressedPeakProviderContext:
    data_root: Path
    weights_path: Path
    p995_context: P995BaselineNodeStateProviderContext
    source_fold_label: int = 0


class _NodeAttrsFrame:
    def __init__(
        self,
        frame: pd.DataFrame,
    ) -> None:
        self._frame = frame

    def to_pandas(
        self,
    ) -> pd.DataFrame:
        return self._frame.copy()


class _FreshP995NodeGraph:
    """Minimal graph facade justified by canonical writer usage.

    R3R7-R1 source audit proves the historical ``graph`` object is
    consumed only through ``graph.node_attrs(...)`` inside the
    suppressed-peak dataset loop.  No edge API is emulated.
    """

    def __init__(
        self,
        tzyx: np.ndarray,
    ) -> None:
        array = np.asarray(
            tzyx
        )

        if (
            array.ndim != 2
            or
            array.shape[1] != 4
        ):
            raise ValueError(
                "Fresh P995 node table must have shape (N, 4) "
                "with columns [t,z,y,x]."
            )

        if (
            array.dtype
            !=
            np.int64
        ):
            raise TypeError(
                "Fresh P995 node table must be int64."
            )

        self._frame = pd.DataFrame(
            array,
            columns=[
                "t",
                "z",
                "y",
                "x",
            ],
        )

    def node_attrs(
        self,
        *,
        attr_keys,
    ):
        keys = list(
            attr_keys
        )

        missing = [
            key
            for key
            in keys
            if key
            not in
            self._frame.columns
        ]

        if missing:
            raise KeyError(
                "Fresh P995 graph facade missing node attributes: "
                + repr(
                    missing
                )
            )

        return _NodeAttrsFrame(
            self._frame[
                keys
            ]
        )


def _project_root() -> Path:
    return (
        Path(__file__)
        .resolve()
        .parents[2]
    )


def _canonical_writer_path() -> Path:
    return (
        _project_root()
        /
        SUPPRESSED_PEAK_CANONICAL_WRITER_RELATIVE_PATH
    )


def _sha256_file(
    path: Path,
) -> str:
    import hashlib

    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def _source_segment(
    source: str,
    node: ast.AST,
) -> str:
    return (
        ast.get_source_segment(
            source,
            node,
        )
        or
        ""
    )


def _sha256_text(
    text: str,
) -> str:
    import hashlib

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def _assignment_target_names(
    node: ast.AST,
) -> set[str]:
    names: set[str] = set()

    def collect(
        target,
    ):
        if isinstance(
            target,
            ast.Name,
        ):
            names.add(
                target.id
            )

        elif isinstance(
            target,
            (
                ast.Tuple,
                ast.List,
            ),
        ):
            for item in (
                target.elts
            ):
                collect(
                    item
                )

    if isinstance(
        node,
        ast.Assign,
    ):
        for target in (
            node.targets
        ):
            collect(
                target
            )

    elif isinstance(
        node,
        ast.AnnAssign,
    ):
        collect(
            node.target
        )

    return names


def _contains_loaded_name(
    node: ast.AST,
    name: str,
) -> bool:
    return any(
        (
            isinstance(
                child,
                ast.Name,
            )
            and
            isinstance(
                child.ctx,
                ast.Load,
            )
            and
            child.id == name
        )
        for child
        in ast.walk(
            node
        )
    )


def _graph_attribute_names(
    node: ast.AST,
) -> set[str]:
    attrs: set[str] = set()

    for child in ast.walk(
        node
    ):
        if (
            isinstance(
                child,
                ast.Attribute,
            )
            and
            isinstance(
                child.value,
                ast.Name,
            )
            and
            child.value.id
            ==
            "graph"
        ):
            attrs.add(
                child.attr
            )

    return attrs


def _find_canonical_dataset_loop(
    tree: ast.Module,
    source: str,
) -> ast.For:
    matches = []

    for node in (
        tree.body
    ):
        if not isinstance(
            node,
            ast.For,
        ):
            continue

        target_text = ast.unparse(
            node.target
        )

        body_text = "\n".join(
            _source_segment(
                source,
                stmt,
            )
            for stmt
            in node.body
        )

        if (
            "name"
            in
            target_text
            and
            "candidate_parts"
            in
            body_text
            and
            "DatasetReplay"
            in
            body_text
        ):
            matches.append(
                node
            )

    if (
        len(matches)
        !=
        1
    ):
        raise RuntimeError(
            "Canonical suppressed-peak dataset loop is not unique: "
            + repr(
                [
                    node.lineno
                    for node
                    in matches
                ]
            )
        )

    return matches[
        0
    ]


def _find_universe_assignment(
    tree: ast.Module,
) -> ast.Assign:
    matches = []

    for node in (
        tree.body
    ):
        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        if (
            "universe"
            in
            _assignment_target_names(
                node
            )
        ):
            matches.append(
                node
            )

    if (
        len(matches)
        !=
        1
    ):
        raise RuntimeError(
            "Canonical universe assignment is not unique."
        )

    return matches[
        0
    ]


class _SuppressedPeakTransformer(
    ast.NodeTransformer
):
    """Mechanical runtime-boundary transformer only."""

    def __init__(
        self,
        *,
        dataset_loop: ast.For,
        universe_assignment: ast.Assign,
    ) -> None:
        super().__init__()

        self.dataset_loop = (
            dataset_loop
        )

        self.universe_assignment = (
            universe_assignment
        )

        self.replaced_p995_graph_assignments = 0

        self.removed_controlled_assignments = []
        self.removed_historical_enumeration_assertions = 0

    def visit_Assert(
        self,
        node: ast.Assert,
    ):
        expression = ast.unparse(
            node
        )

        if (
            node.lineno == 169
            and
            expression == 'assert len(corp) == 25'
        ):
            self.removed_historical_enumeration_assertions += 1
            return None

        return self.generic_visit(
            node
        )

    def visit_Assign(
        self,
        node: ast.Assign,
    ):
        targets = (
            _assignment_target_names(
                node
            )
        )

        controlled = (
            targets
            &
            _SUPPRESSED_CONTROLLED_ASSIGNMENT_NAMES
        )

        if controlled:
            self.removed_controlled_assignments.extend(
                sorted(
                    controlled
                )
            )
            return None

        if (
            "graph"
            in
            targets
            and
            _contains_loaded_name(
                node.value,
                "P995",
            )
        ):
            self.replaced_p995_graph_assignments += 1

            replacement = ast.Assign(
                targets=[
                    ast.Name(
                        id="graph",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Name(
                        id="_suppressed_fresh_graph",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Name(
                            id="name",
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=[],
                ),
            )

            return ast.copy_location(
                replacement,
                node,
            )

        return self.generic_visit(
            node
        )

    def visit_AnnAssign(
        self,
        node: ast.AnnAssign,
    ):
        targets = (
            _assignment_target_names(
                node
            )
        )

        controlled = (
            targets
            &
            _SUPPRESSED_CONTROLLED_ASSIGNMENT_NAMES
        )

        if controlled:
            self.removed_controlled_assignments.extend(
                sorted(
                    controlled
                )
            )
            return None

        if (
            "graph"
            in
            targets
            and
            node.value
            is not None
            and
            _contains_loaded_name(
                node.value,
                "P995",
            )
        ):
            self.replaced_p995_graph_assignments += 1

            replacement = ast.Assign(
                targets=[
                    ast.Name(
                        id="graph",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Name(
                        id="_suppressed_fresh_graph",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Name(
                            id="name",
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=[],
                ),
            )

            return ast.copy_location(
                replacement,
                node,
            )

        return self.generic_visit(
            node
        )


def _remaining_loaded_name_count(
    tree: ast.AST,
    name: str,
) -> int:
    return sum(
        1
        for child
        in ast.walk(
            tree
        )
        if (
            isinstance(
                child,
                ast.Name,
            )
            and
            isinstance(
                child.ctx,
                ast.Load,
            )
            and
            child.id
            ==
            name
        )
    )


@lru_cache(
    maxsize=1
)
def _build_source_derived_code():
    writer_path = (
        _canonical_writer_path()
    )

    if (
        _sha256_file(
            writer_path
        )
        !=
        SUPPRESSED_PEAK_CANONICAL_WRITER_SHA256
    ):
        raise RuntimeError(
            "Canonical suppressed-peak writer SHA changed."
        )

    source = (
        writer_path.read_text(
            encoding="utf-8"
        )
    )

    tree = ast.parse(
        source,
        filename=str(
            writer_path
        ),
    )

    dataset_loop = (
        _find_canonical_dataset_loop(
            tree,
            source,
        )
    )

    universe_assignment = (
        _find_universe_assignment(
            tree
        )
    )

    enumeration_assertions = [
        node
        for node
        in ast.walk(
            tree
        )
        if (
            isinstance(
                node,
                ast.Assert,
            )
            and
            node.lineno == 169
        )
    ]

    if len(
        enumeration_assertions
    ) != 1:
        raise RuntimeError(
            "Historical Fold0 enumeration assertion is not unique."
        )

    enumeration_assertion_source = (
        _source_segment(
            source,
            enumeration_assertions[0],
        )
    )

    if (
        enumeration_assertion_source
        !=
        'assert len(corp) == 25'
    ):
        raise RuntimeError(
            "Historical Fold0 enumeration assertion source changed."
        )

    if (
        _sha256_text(
            enumeration_assertion_source
        )
        !=
        'f7e90f6be28ace0e5b33ed34e0afc176db5ac21c3ee044e2078511c96f2a6257'
    ):
        raise RuntimeError(
            "Historical Fold0 enumeration assertion SHA changed."
        )

    if (
        universe_assignment.lineno
        !=
        SUPPRESSED_PEAK_CANONICAL_UNIVERSE_LINE
    ):
        raise RuntimeError(
            "Canonical universe assignment line changed."
        )

    if (
        _sha256_text(
            _source_segment(
                source,
                dataset_loop,
            )
        )
        !=
        SUPPRESSED_PEAK_DATASET_LOOP_SHA256
    ):
        raise RuntimeError(
            "Canonical suppressed-peak dataset loop changed."
        )

    if (
        _sha256_text(
            _source_segment(
                source,
                universe_assignment,
            )
        )
        !=
        SUPPRESSED_PEAK_UNIVERSE_ASSIGNMENT_SHA256
    ):
        raise RuntimeError(
            "Canonical universe assignment source changed."
        )

    graph_attrs = (
        _graph_attribute_names(
            dataset_loop
        )
    )

    if (
        graph_attrs
        !=
        set(
            SUPPRESSED_PEAK_ALLOWED_HISTORICAL_GRAPH_ATTRIBUTES
        )
    ):
        raise RuntimeError(
            "Historical P995 graph usage changed: "
            + repr(
                sorted(
                    graph_attrs
                )
            )
        )

    # Keep only canonical statements through universe construction.
    retained = [
        statement
        for statement
        in tree.body
        if (
            getattr(
                statement,
                "lineno",
                0,
            )
            <=
            universe_assignment.end_lineno
        )
    ]

    runtime_tree = ast.Module(
        body=retained,
        type_ignores=[],
    )

    transformer = (
        _SuppressedPeakTransformer(
            dataset_loop=dataset_loop,
            universe_assignment=universe_assignment,
        )
    )

    runtime_tree = (
        transformer.visit(
            runtime_tree
        )
    )

    runtime_tree = (
        ast.fix_missing_locations(
            runtime_tree
        )
    )

    if (
        transformer.replaced_p995_graph_assignments
        !=
        1
    ):
        raise RuntimeError(
            "Expected exactly one historical P995 graph "
            "assignment replacement; observed "
            + str(
                transformer.replaced_p995_graph_assignments
            )
        )

    if (
        transformer.removed_historical_enumeration_assertions
        !=
        1
    ):
        raise RuntimeError(
            "Expected exactly one historical Fold0 "
            "enumeration assertion removal."
        )

    # After the graph substitution, no runtime P995 variable may
    # remain.  Strings such as "P995 parity" do not count.
    remaining_p995_loads = (
        _remaining_loaded_name_count(
            runtime_tree,
            "P995",
        )
    )

    if (
        remaining_p995_loads
        !=
        0
    ):
        raise RuntimeError(
            "Historical P995 runtime dependency remains after "
            "source-derived transformation: "
            + str(
                remaining_p995_loads
            )
        )

    return (
        compile(
            runtime_tree,
            filename=str(
                writer_path
            )
            + "::SOURCE_DERIVED_PROVIDER",
            mode="exec",
        ),
        {
            "dataset_loop_line":
                dataset_loop.lineno,

            "dataset_loop_end_line":
                dataset_loop.end_lineno,

            "universe_line":
                universe_assignment.lineno,

            "graph_attributes":
                sorted(
                    graph_attrs
                ),

            "P995_graph_replacements":
                transformer.replaced_p995_graph_assignments,
            "historical_enumeration_assertions_removed":
                transformer.removed_historical_enumeration_assertions,

            "historical_enumeration_assertion_line":
                169,

            "historical_enumeration_assertion_sha256":
                'f7e90f6be28ace0e5b33ed34e0afc176db5ac21c3ee044e2078511c96f2a6257',

            "remaining_P995_runtime_loads":
                remaining_p995_loads,

            "removed_controlled_assignments":
                sorted(
                    set(
                        transformer.removed_controlled_assignments
                    )
                ),
        },
    )


def source_derived_transform_audit():
    _, audit = (
        _build_source_derived_code()
    )

    return dict(
        audit
    )


def _validate_context(
    context: SuppressedPeakProviderContext,
) -> None:
    if not isinstance(
        context,
        SuppressedPeakProviderContext,
    ):
        raise TypeError(
            "context must be SuppressedPeakProviderContext"
        )

    data_root = (
        Path(
            context.data_root
        )
        .expanduser()
        .resolve()
    )

    weights_path = (
        Path(
            context.weights_path
        )
        .expanduser()
        .resolve()
    )

    if not (
        data_root.exists()
        and
        data_root.is_dir()
    ):
        raise FileNotFoundError(
            "Suppressed-peak data_root does not exist: "
            + str(
                data_root
            )
        )

    if not (
        weights_path.exists()
        and
        weights_path.is_file()
    ):
        raise FileNotFoundError(
            "Suppressed-peak weights_path does not exist: "
            + str(
                weights_path
            )
        )

    p995_data_root = (
        Path(
            context.p995_context.data_root
        )
        .expanduser()
        .resolve()
    )

    p995_weights_path = (
        Path(
            context.p995_context.weights_path
        )
        .expanduser()
        .resolve()
    )

    if (
        p995_data_root
        !=
        data_root
    ):
        raise ValueError(
            "Suppressed-peak and P995 provider data roots differ."
        )

    if (
        p995_weights_path
        !=
        weights_path
    ):
        raise ValueError(
            "Suppressed-peak and P995 provider checkpoints differ."
        )


def _fresh_p995_graph_for_dataset(
    dataset_id: str,
    context: SuppressedPeakProviderContext,
):
    table = (
        _baseline_node_state_provider(
            dataset_id,
            context=context.p995_context,
        )
    )

    table = np.asarray(
        table
    )

    return _FreshP995NodeGraph(
        table
    )


def _execute_source_derived(
    dataset_id: str,
    context: SuppressedPeakProviderContext,
) -> pd.DataFrame:
    # Establish the already-frozen repo-local import closure
    # before canonical source imports execute.
    authority = (
        _load_p995_predict_authority_module()
    )

    del authority

    runtime_code, _ = (
        _build_source_derived_code()
    )

    data_root = (
        Path(
            context.data_root
        )
        .expanduser()
        .resolve()
    )

    weights_path = (
        Path(
            context.weights_path
        )
        .expanduser()
        .resolve()
    )

    dataset_path = (
        data_root
        /
        f"{dataset_id}.zarr"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset Zarr does not exist: "
            + str(
                dataset_path
            )
        )

    # The original writer's Fold0 filtering is replaced only as
    # an enumeration boundary.  A one-row metadata table keeps
    # source statements that consume ``fold_map`` unchanged.
    metadata = pd.DataFrame(
        {
            "dataset":
                [
                    dataset_id
                ],

            "fold":
                [
                    int(
                        context.source_fold_label
                    )
                ],
        }
    )

    with tempfile.TemporaryDirectory(
        prefix="biohub_suppressed_peak_"
    ) as tmp_text:
        tmp = Path(
            tmp_text
        )

        virtual_p995_root = (
            tmp
            /
            "fresh_p995"
        )

        out_dir = (
            tmp
            /
            "suppressed_outputs"
        )

        cv_path = (
            tmp
            /
            "explicit_dataset_boundary.csv"
        )

        virtual_p995_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata.to_csv(
            cv_path,
            index=False,
        )

        namespace = {
            "__file__":
                str(
                    _canonical_writer_path()
                ),

            "__name__":
                (
                    "biohub_cell_tracking."
                    "_source_derived_suppressed_peak_runtime"
                ),

            # Frozen orchestration replacements.
            "PROJECT":
                _project_root(),

            "TRAIN_ROOT":
                data_root,

            "P995":
                virtual_p995_root,

            "CKPT":
                weights_path,

            "OUT_DIR":
                out_dir,

            "CV_PATH":
                cv_path,

            "cv":
                metadata.copy(),

            "corp":
                metadata.copy(),

            "names":
                [
                    dataset_id
                ],

            "fold_map":
                {
                    dataset_id:
                        int(
                            context.source_fold_label
                        )
                },

            # Fresh replacement for historical P995 graph load.
            "_suppressed_fresh_graph":
                lambda name:
                    _fresh_p995_graph_for_dataset(
                        name,
                        context,
                    ),
        }

        exec(
            runtime_code,
            namespace,
            namespace,
        )

        universe = (
            namespace.get(
                "universe"
            )
        )

        if not isinstance(
            universe,
            pd.DataFrame,
        ):
            raise RuntimeError(
                "Canonical source-derived execution did not "
                "produce a pandas DataFrame named 'universe'."
            )

        result = (
            universe.copy()
        )

    return result


def suppressed_peak_provider(
    dataset_id: str,
    *,
    context: SuppressedPeakProviderContext,
) -> pd.DataFrame:
    """Generate the fresh GT-blind suppressed-peak universe.

    No historical suppressed-peak pickle or historical P995 GEFF
    is read.  P995 node state comes from the structurally-verified
    fresh P995 provider.
    """

    if not isinstance(
        dataset_id,
        str,
    ):
        raise TypeError(
            "dataset_id must be str"
        )

    dataset_id = (
        dataset_id.strip()
    )

    if not dataset_id:
        raise ValueError(
            "dataset_id must be non-empty"
        )

    _validate_context(
        context
    )

    result = (
        _execute_source_derived(
            dataset_id,
            context,
        )
    )

    if "dataset" not in result.columns:
        raise RuntimeError(
            "Source-derived suppressed-peak provider output is missing required dataset column."
        )

    if (
        "dataset"
        in
        result.columns
    ):
        observed = set(
            result[
                "dataset"
            ]
            .astype(
                str
            )
            .unique()
            .tolist()
        )

        if (
            observed
            not in
            (
                set(),
                {
                    dataset_id
                },
            )
        ):
            raise RuntimeError(
                "Source-derived suppressed-peak provider leaked "
                "rows from another dataset: "
                + repr(
                    sorted(
                        observed
                    )
                )
            )

    return result.reset_index(
        drop=True
    )
