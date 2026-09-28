from __future__ import annotations

import ast
import hashlib
import textwrap

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from biohub_cell_tracking.stage12_suppressed_peak_provider import (
    suppressed_peak_provider,
)


CANONICAL_WRITER_REL = Path(
    "scripts/run_stage12_fold0_bridge_geometry_universe.py"
)

CANONICAL_WRITER_SHA256 = "c7fb950c03711ac9c85f0476013f88554a42fc2b8fb21c8cc4afc1bb6b5649a4"

CANONICAL_DATASET_LOOP_SHA256 = "a47787269ee66281709a2760d794846d362c6befe72d3abd0dd677132bd938fc"

CANONICAL_DATASET_LOOP_START = 355
CANONICAL_DATASET_LOOP_END = 673
CANONICAL_B12_SINK_LINE = 908


FORBIDDEN_HISTORICAL_RUNTIME_FILENAMES = frozenset({
    "stage11_fold0_morphdiv_v2_actions_locked.pkl",
    "stage12_fold0_mutual_cont_v1_actions_locked.pkl",
    "stage12_fold0_mutual_cont_v2_actions_locked.pkl",
    "stage12_fold0_suppressed_peak_universe.pkl",
})


@dataclass(frozen=True)
class B12PortableContext:
    post_v2_graph_context: Any
    suppressed_peak_context: Any

    # Explicit source-derived namespace additions only.
    # Historical action/suppressed tables are not accepted.
    canonical_namespace: Mapping[str, Any] = field(
        default_factory=dict
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(
        text.encode("utf-8")
    )


def _project_root() -> Path:
    return (
        Path(__file__)
        .resolve()
        .parents[2]
    )


def _dotted_name(
    node: ast.AST,
) -> str | None:

    if isinstance(
        node,
        ast.Name,
    ):
        return node.id

    if isinstance(
        node,
        ast.Attribute,
    ):

        left = _dotted_name(
            node.value
        )

        if left:
            return (
                f"{left}.{node.attr}"
            )

        return node.attr

    return None


def _target_names(
    node: ast.AST,
) -> set[str]:

    result = set()

    def visit(x: ast.AST):

        if isinstance(
            x,
            ast.Name,
        ):

            result.add(
                x.id
            )

        elif isinstance(
            x,
            (
                ast.Tuple,
                ast.List,
            ),
        ):

            for elt in x.elts:
                visit(
                    elt
                )

        elif isinstance(
            x,
            ast.Starred,
        ):

            visit(
                x.value
            )

        elif isinstance(
            x,
            ast.Subscript,
        ):

            visit(
                x.value
            )

        elif isinstance(
            x,
            ast.Attribute,
        ):

            visit(
                x.value
            )

    visit(node)

    return result


def _node_source(
    text: str,
    node: ast.AST,
) -> str:

    source = ast.get_source_segment(
        text,
        node,
    )

    if source is None:

        raise AssertionError(
            "Canonical AST source extraction failed."
        )

    return source


def _load_canonical_source(
    project_root: Path,
) -> tuple[
    Path,
    str,
    ast.Module,
]:

    path = (
        project_root
        / CANONICAL_WRITER_REL
    )

    raw = path.read_bytes()

    actual_sha = (
        _sha256_bytes(
            raw
        )
    )

    if (
        actual_sha
        != CANONICAL_WRITER_SHA256
    ):

        raise AssertionError(
            "Canonical B12 writer SHA mismatch: "
            f"{actual_sha}"
        )


    text = raw.decode(
        "utf-8"
    )

    tree = ast.parse(
        text,
        filename=str(
            path
        ),
    )


    return (
        path,
        text,
        tree,
    )


def _extract_plan(
    project_root: Path | None = None,
) -> dict[str, Any]:

    if project_root is None:

        project_root = (
            _project_root()
        )


    path, text, tree = (
        _load_canonical_source(
            project_root
        )
    )


    loops = [
        node
        for node in ast.walk(
            tree
        )
        if (
            isinstance(
                node,
                (
                    ast.For,
                    ast.AsyncFor,
                ),
            )
            and node.lineno
            == CANONICAL_DATASET_LOOP_START
            and node.end_lineno
            == CANONICAL_DATASET_LOOP_END
        )
    ]


    if len(loops) != 1:

        raise AssertionError(
            "Canonical B12 dataset loop "
            "was not uniquely identified."
        )


    loop = loops[0]


    loop_source = (
        _node_source(
            text,
            loop,
        )
    )


    if (
        _sha256_text(
            loop_source
        )
        != CANONICAL_DATASET_LOOP_SHA256
    ):

        raise AssertionError(
            "Canonical dataset-loop SHA mismatch."
        )


    if len(loop.body) < 4:

        raise AssertionError(
            "Canonical B12 dataset loop "
            "contains fewer than four statements."
        )


    s0, s1, s2, s3 = (
        loop.body[:4]
    )


    # -------------------------------------------------------------
    # Canonical statement 0
    # -------------------------------------------------------------

    if not (
        isinstance(
            s0,
            ast.Assign,
        )
        and isinstance(
            s0.value,
            ast.Call,
        )
        and _dotted_name(
            s0.value.func
        )
        == "build_current_v2"
        and "g"
        in _target_names(
            s0.targets[0]
        )
    ):

        raise AssertionError(
            "Canonical statement 0 changed."
        )


    # -------------------------------------------------------------
    # Canonical statement 1
    # -------------------------------------------------------------

    if not (
        isinstance(
            s1,
            ast.Assign,
        )
        and isinstance(
            s1.value,
            ast.Call,
        )
        and _dotted_name(
            s1.value.func
        )
        == "graph_state"
    ):

        raise AssertionError(
            "Canonical statement 1 changed."
        )


    # -------------------------------------------------------------
    # Canonical statement 2
    # -------------------------------------------------------------

    if not (
        isinstance(
            s2,
            ast.Assign,
        )
        and isinstance(
            s2.value,
            ast.Call,
        )
        and _dotted_name(
            s2.value.func
        )
        == "nodes.reset_index"
    ):

        raise AssertionError(
            "Canonical statement 2 changed."
        )


    # -------------------------------------------------------------
    # Canonical statement 3
    # -------------------------------------------------------------

    if not (
        isinstance(
            s3,
            ast.Assign,
        )
        and "pp"
        in _target_names(
            s3.targets[0]
        )
    ):

        raise AssertionError(
            "Canonical statement 3 changed."
        )


    module_loop_index = None


    for index, stmt in enumerate(
        tree.body
    ):

        if stmt is loop:

            module_loop_index = (
                index
            )

            break


    if (
        module_loop_index
        is None
    ):

        raise AssertionError(
            "Canonical dataset loop missing from module body."
        )


    # -------------------------------------------------------------
    # Source-derived mutable-container initializers.
    # -------------------------------------------------------------

    mutation_methods = {
        "append",
        "extend",
        "add",
        "update",
    }


    mutated_receivers = set()


    for call in ast.walk(
        loop
    ):

        if not isinstance(
            call,
            ast.Call,
        ):

            continue


        if not isinstance(
            call.func,
            ast.Attribute,
        ):

            continue


        if (
            call.func.attr
            not in mutation_methods
        ):

            continue


        if isinstance(
            call.func.value,
            ast.Name,
        ):

            mutated_receivers.add(
                call.func.value.id
            )


    preloop_initializers = []


    for receiver in sorted(
        mutated_receivers
    ):

        candidates = []


        for stmt in (
            tree.body[
                :module_loop_index
            ]
        ):

            names = set()


            if isinstance(
                stmt,
                ast.Assign,
            ):

                for target in (
                    stmt.targets
                ):

                    names |= (
                        _target_names(
                            target
                        )
                    )


            elif isinstance(
                stmt,
                ast.AnnAssign,
            ):

                names |= (
                    _target_names(
                        stmt.target
                    )
                )


            if receiver in names:

                candidates.append(
                    stmt
                )


        if not candidates:

            continue


        selected = (
            candidates[-1]
        )


        source = (
            _node_source(
                text,
                selected,
            )
        )


        if (
            "read_pickle"
            in source
        ):

            continue


        preloop_initializers.append(
            source
        )


    # -------------------------------------------------------------
    # Canonical retained loop body:
    #
    # statement 1
    # statement 2
    # statement 4+
    # -------------------------------------------------------------

    retained_loop_sources = [
        _node_source(
            text,
            node,
        )
        for node in [
            s1,
            s2,
            *loop.body[4:],
        ]
    ]


    # -------------------------------------------------------------
    # Canonical post-loop source up to B12 sink.
    # External output sinks are excluded.
    # -------------------------------------------------------------

    output_suffixes = (
        ".to_pickle",
        ".to_csv",
        ".to_json",
        ".write_text",
        ".write_bytes",
    )


    postloop_sources = []


    for stmt in (
        tree.body[
            module_loop_index + 1:
        ]
    ):

        if (
            stmt.lineno
            >= CANONICAL_B12_SINK_LINE
        ):

            break


        call_names = []


        for call in ast.walk(
            stmt
        ):

            if not isinstance(
                call,
                ast.Call,
            ):

                continue


            name = _dotted_name(
                call.func
            )


            if name:

                call_names.append(
                    name
                )


        if any(
            (
                name == "open"
                or name.endswith(
                    output_suffixes
                )
            )
            for name in (
                call_names
            )
        ):

            continue


        source = (
            _node_source(
                text,
                stmt,
            )
        )


        if (
            "read_pickle"
            in source
        ):

            raise AssertionError(
                "Historical pickle read entered "
                "post-loop portable source."
            )


        postloop_sources.append(
            source
        )


    # Must construct b12 before sink.
    parsed_postloop = ast.parse(
        (
            "\n\n".join(
                postloop_sources
            )
            or "pass"
        )
    )


    b12_defs = 0


    for stmt in (
        parsed_postloop.body
    ):

        names = set()


        if isinstance(
            stmt,
            ast.Assign,
        ):

            for target in (
                stmt.targets
            ):

                names |= (
                    _target_names(
                        target
                    )
                )


        elif isinstance(
            stmt,
            ast.AnnAssign,
        ):

            names |= (
                _target_names(
                    stmt.target
                )
            )


        if (
            "b12"
            in names
        ):

            b12_defs += 1


    if b12_defs < 1:

        raise AssertionError(
            "Canonical post-loop source does not construct b12."
        )


    return {
        "path":
            path,

        "text":
            text,

        "tree":
            tree,

        "loop":
            loop,

        "canonical_statement_0":
            _node_source(
                text,
                s0,
            ),

        "canonical_statement_1":
            _node_source(
                text,
                s1,
            ),

        "canonical_statement_2":
            _node_source(
                text,
                s2,
            ),

        "canonical_statement_3":
            _node_source(
                text,
                s3,
            ),

        "preloop_initializers":
            preloop_initializers,

        "retained_loop_sources":
            retained_loop_sources,

        "postloop_sources":
            postloop_sources,
    }


def _statement_has_forbidden_io(
    stmt: ast.AST,
) -> bool:

    for call in ast.walk(
        stmt
    ):

        if not isinstance(
            call,
            ast.Call,
        ):

            continue


        name = _dotted_name(
            call.func
        )


        if name in {
            "open",
            "pd.read_pickle",
            "pd.read_csv",
            "read_pickle",
            "read_csv",
        }:

            return True


        if (
            name
            and (
                name.endswith(
                    ".read_text"
                )
                or name.endswith(
                    ".read_bytes"
                )
                or name.endswith(
                    ".to_pickle"
                )
            )
        ):

            return True


    return False


def _build_canonical_namespace(
    plan: Mapping[str, Any],
    extra_namespace: Mapping[
        str,
        Any,
    ] | None = None,
) -> dict[str, Any]:

    text = plan[
        "text"
    ]

    tree = plan[
        "tree"
    ]

    loop = plan[
        "loop"
    ]


    namespace: dict[
        str,
        Any,
    ] = {
        "__builtins__":
            __builtins__,

        "__file__":
            str(
                plan[
                    "path"
                ]
            ),

        "__name__":
            "_stage12_b12_canonical_runtime",
    }


    # These historical/runtime-state variables are NEVER reconstructed here.
    forbidden_names = {
        "morph",
        "v1",
        "v2",
        "peaks",
        "cv",
        "corp",
        "names",
        "prefix_map",
        "policies",
    }


    # -------------------------------------------------------------
    # Canonical imports.
    # -------------------------------------------------------------

    for stmt in (
        tree.body
    ):

        if not isinstance(
            stmt,
            (
                ast.Import,
                ast.ImportFrom,
            ),
        ):

            continue


        code = compile(
            ast.Module(
                body=[
                    stmt
                ],
                type_ignores=[],
            ),
            filename=str(
                plan[
                    "path"
                ]
            ),
            mode="exec",
        )


        exec(
            code,
            namespace,
            namespace,
        )


    # -------------------------------------------------------------
    # Safe source-derived module constants / path definitions.
    # -------------------------------------------------------------

    for stmt in (
        tree.body
    ):

        if (
            stmt.lineno
            >= loop.lineno
        ):

            break


        if not isinstance(
            stmt,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):

            continue


        targets = set()


        if isinstance(
            stmt,
            ast.Assign,
        ):

            for target in (
                stmt.targets
            ):

                targets |= (
                    _target_names(
                        target
                    )
                )


        else:

            targets |= (
                _target_names(
                    stmt.target
                )
            )


        if (
            targets
            & forbidden_names
        ):

            continue


        if (
            _statement_has_forbidden_io(
                stmt
            )
        ):

            continue


        code = compile(
            ast.Module(
                body=[
                    stmt
                ],
                type_ignores=[],
            ),
            filename=str(
                plan[
                    "path"
                ]
            ),
            mode="exec",
        )


        try:

            exec(
                code,
                namespace,
                namespace,
            )


        except (
            NameError,
            AttributeError,
            TypeError,
            FileNotFoundError,
        ):

            # Safe to defer; unresolved source-derived names may be supplied
            # explicitly via context.canonical_namespace.
            continue


    # -------------------------------------------------------------
    # Canonical helper function/class definitions.
    #
    # build_current_v2 is deliberately excluded so portable runtime cannot
    # accidentally call historical graph reconstruction.
    # -------------------------------------------------------------

    for stmt in (
        tree.body
    ):

        if not isinstance(
            stmt,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):

            continue


        if (
            isinstance(
                stmt,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and stmt.name
            == "build_current_v2"
        ):

            continue


        code = compile(
            ast.Module(
                body=[
                    stmt
                ],
                type_ignores=[],
            ),
            filename=str(
                plan[
                    "path"
                ]
            ),
            mode="exec",
        )


        try:

            exec(
                code,
                namespace,
                namespace,
            )


        except NameError:

            continue


    if extra_namespace:

        namespace.update(
            dict(
                extra_namespace
            )
        )


    # Never allow caller to smuggle historical runtime tables back in.
    for name in (
        forbidden_names
    ):

        namespace.pop(
            name,
            None,
        )


    return namespace


def _indent_source(
    source: str,
    spaces: int,
) -> str:

    return textwrap.indent(
        textwrap.dedent(
            source
        ).rstrip(),
        " " * spaces,
    )


def _build_executor_source(
    plan: Mapping[str, Any],
) -> str:

    lines = [
        "def __portable_b12_executor(",
        "    dataset_id,",
        "    *,",
        "    post_v2_graph_provider,",
        "    post_v2_graph_context,",
        "    suppressed_peak_context,",
        "):",
        "    names = [dataset_id]",
    ]


    for source in (
        plan[
            "preloop_initializers"
        ]
    ):

        lines.append(
            _indent_source(
                source,
                4,
            )
        )


    # Use the canonical loop target + iterator semantics.
    loop = plan[
        "loop"
    ]


    loop_target = ast.unparse(
        loop.target
    )


    # We intentionally use [dataset_id] for explicit one-dataset runtime,
    # while preserving canonical enumerate/start semantics.
    #
    # Canonical loop is known to be enumerate(names, start=1).
    loop_iter = ast.unparse(
        loop.iter
    )


    if (
        "enumerate(names"
        not in loop_iter.replace(
            " ",
            "",
        )
        and "enumerate(names"
        not in loop_iter
    ):

        # More robust normalized check.
        normalized = (
            loop_iter
            .replace(
                " ",
                ""
            )
        )

        if (
            not normalized.startswith(
                "enumerate(names"
            )
        ):

            raise AssertionError(
                "Canonical B12 loop iterator changed: "
                f"{loop_iter}"
            )


    lines.extend([
        "",
        f"    for {loop_target} in {loop_iter}:",
        (
            "        g = post_v2_graph_provider("
        ),
        (
            "            name,"
        ),
        (
            "            context=post_v2_graph_context,"
        ),
        (
            "        )"
        ),
    ])


    # Canonical statement 1 and statement 2.
    for source in (
        plan[
            "retained_loop_sources"
        ][:2]
    ):

        lines.append(
            _indent_source(
                source,
                8,
            )
        )


    # Frozen semantic replacement for canonical statement 3.
    lines.extend([
        (
            "        pp = suppressed_peak_provider("
        ),
        (
            "            name,"
        ),
        (
            "            context=suppressed_peak_context,"
        ),
        (
            "        )"
        ),
    ])


    # Canonical statement 4+.
    for source in (
        plan[
            "retained_loop_sources"
        ][2:]
    ):

        lines.append(
            _indent_source(
                source,
                8,
            )
        )


    lines.append(
        ""
    )


    # Canonical post-loop geom -> b12 construction.
    for source in (
        plan[
            "postloop_sources"
        ]
    ):

        lines.append(
            _indent_source(
                source,
                4,
            )
        )


    lines.extend([
        "",
        "    return b12",
        "",
    ])


    source = "\n".join(
        lines
    )


    if (
        "read_pickle"
        in source
    ):

        raise AssertionError(
            "Portable executor contains read_pickle."
        )


    for filename in (
        FORBIDDEN_HISTORICAL_RUNTIME_FILENAMES
    ):

        if filename in source:

            raise AssertionError(
                "Historical artifact leaked into portable executor: "
                f"{filename}"
            )


    if (
        "build_current_v2("
        in source
    ):

        raise AssertionError(
            "Historical build_current_v2 leaked into portable executor."
        )


    return source


def materialize_b12_dataset(
    dataset_id: str,
    *,
    post_v2_graph_provider: Callable[..., Any],
    context: B12PortableContext,
) -> pd.DataFrame:

    if (
        not isinstance(
            dataset_id,
            str,
        )
        or not dataset_id
    ):

        raise ValueError(
            "dataset_id must be a non-empty string."
        )


    if not callable(
        post_v2_graph_provider
    ):

        raise TypeError(
            "post_v2_graph_provider must be callable."
        )


    plan = _extract_plan(
        _project_root()
    )


    namespace = (
        _build_canonical_namespace(
            plan,
            context.canonical_namespace,
        )
    )


    namespace[
        "post_v2_graph_provider"
    ] = post_v2_graph_provider


    namespace[
        "suppressed_peak_provider"
    ] = suppressed_peak_provider


    executor_source = (
        _build_executor_source(
            plan
        )
    )


    code = compile(
        executor_source,
        filename="<stage12_b12_portable_executor>",
        mode="exec",
    )


    exec(
        code,
        namespace,
        namespace,
    )


    executor = namespace[
        "__portable_b12_executor"
    ]


    b12 = executor(
        dataset_id,
        post_v2_graph_provider=(
            post_v2_graph_provider
        ),
        post_v2_graph_context=(
            context.post_v2_graph_context
        ),
        suppressed_peak_context=(
            context.suppressed_peak_context
        ),
    )


    if not isinstance(
        b12,
        pd.DataFrame,
    ):

        raise TypeError(
            "Portable B12 executor did not return pandas.DataFrame."
        )


    if "dataset" not in b12.columns:
        raise AssertionError(
            "Portable B12 output is missing required dataset column."
        )

    if (
        "dataset"
        in b12.columns
    ):

        wrong_dataset = (
            b12[
                "dataset"
            ]
            .astype(str)
            .ne(
                dataset_id
            )
        )


        if bool(
            wrong_dataset.any()
        ):

            raise AssertionError(
                "Portable B12 output contains another dataset."
            )


    return b12.copy(
        deep=True
    )


__all__ = [
    "B12PortableContext",
    "materialize_b12_dataset",
]
