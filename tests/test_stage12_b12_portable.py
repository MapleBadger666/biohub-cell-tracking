from __future__ import annotations

import ast
import sys

from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]

SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(
        0,
        str(SRC),
    )


from biohub_cell_tracking import stage12_b12_portable as b12p


def test_canonical_source_lock_and_frozen_statements():

    plan = (
        b12p._extract_plan(
            ROOT
        )
    )


    assert (
        "build_current_v2"
        in plan[
            "canonical_statement_0"
        ]
    )


    assert (
        "graph_state"
        in plan[
            "canonical_statement_1"
        ]
    )


    assert (
        "reset_index"
        in plan[
            "canonical_statement_2"
        ]
    )


    assert (
        "peaks"
        in plan[
            "canonical_statement_3"
        ]
    )


def test_executor_has_exact_portable_seams():

    plan = (
        b12p._extract_plan(
            ROOT
        )
    )


    source = (
        b12p._build_executor_source(
            plan
        )
    )


    tree = ast.parse(
        source
    )


    calls = []


    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue


        name = b12p._dotted_name(
            node.func
        )


        if name:

            calls.append(
                name
            )


    assert (
        "post_v2_graph_provider"
        in calls
    )


    assert (
        "suppressed_peak_provider"
        in calls
    )


    assert (
        "build_current_v2"
        not in calls
    )


def test_executor_contains_no_historical_runtime_artifacts():

    plan = (
        b12p._extract_plan(
            ROOT
        )
    )


    source = (
        b12p._build_executor_source(
            plan
        )
    )


    assert (
        "read_pickle"
        not in source
    )


    for filename in (
        b12p.FORBIDDEN_HISTORICAL_RUNTIME_FILENAMES
    ):

        assert (
            filename
            not in source
        )


def test_executor_compiles():

    plan = (
        b12p._extract_plan(
            ROOT
        )
    )


    source = (
        b12p._build_executor_source(
            plan
        )
    )


    compile(
        source,
        "<b12-portable-executor-test>",
        "exec",
    )


def test_postloop_constructs_b12():

    plan = (
        b12p._extract_plan(
            ROOT
        )
    )


    source = "\n\n".join(
        plan[
            "postloop_sources"
        ]
    )


    tree = ast.parse(
        source
        if source.strip()
        else "pass"
    )


    found = False


    for stmt in (
        tree.body
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
                    b12p._target_names(
                        target
                    )
                )


        elif isinstance(
            stmt,
            ast.AnnAssign,
        ):

            names |= (
                b12p._target_names(
                    stmt.target
                )
            )


        if (
            "b12"
            in names
        ):

            found = True


    assert found


def test_public_surface_is_narrow():

    assert (
        b12p.__all__
        == [
            "B12PortableContext",
            "materialize_b12_dataset",
        ]
    )


# === NB14_14_9C_MISSING_DATASET_COLUMN_BOUNDARY_TEST ===

import pandas as pd
import pytest


def _run_b12_boundary_with_controlled_output(
    monkeypatch,
    frame,
):

    # Replace plan extraction, canonical namespace and executor
    # construction so only the final return boundary of
    # materialize_b12_dataset runs on a controlled DataFrame.
    monkeypatch.setattr(
        b12p,
        "_extract_plan",
        lambda project_root=None: {},
    )


    monkeypatch.setattr(
        b12p,
        "_build_canonical_namespace",
        lambda plan, extra_namespace=None: {
            "__controlled_b12_frame": frame,
        },
    )


    monkeypatch.setattr(
        b12p,
        "_build_executor_source",
        lambda plan: (
            "def __portable_b12_executor(dataset_id, **kwargs):\n"
            "    return __controlled_b12_frame\n"
        ),
    )


    def fail_provider(*args, **kwargs):

        raise AssertionError(
            "Controlled B12 boundary test must not call a provider."
        )


    context = b12p.B12PortableContext(
        post_v2_graph_context=None,
        suppressed_peak_context=None,
    )


    return b12p.materialize_b12_dataset(
        "demo",
        post_v2_graph_provider=fail_provider,
        context=context,
    )


def test_b12_boundary_accepts_matching_dataset_column(monkeypatch):

    frame = pd.DataFrame(
        {
            "dataset": [
                "demo",
            ],
            "score": [
                0.5,
            ],
        }
    )


    result = (
        _run_b12_boundary_with_controlled_output(
            monkeypatch,
            frame,
        )
    )


    pd.testing.assert_frame_equal(
        result,
        frame,
    )


    assert result is not frame


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(
            {
                "score": [
                    0.5,
                ],
            }
        ),
        pd.DataFrame(
            {
                "score": pd.Series(
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
def test_b12_boundary_rejects_output_missing_dataset_column(
    monkeypatch,
    frame,
):

    with pytest.raises(
        AssertionError,
        match="missing required dataset column",
    ):

        _run_b12_boundary_with_controlled_output(
            monkeypatch,
            frame,
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
def test_b12_boundary_rejects_output_from_another_dataset(
    monkeypatch,
    datasets,
):

    frame = pd.DataFrame(
        {
            "dataset": datasets,
            "score": [
                0.5,
            ]
            * len(
                datasets
            ),
        }
    )


    with pytest.raises(
        AssertionError,
        match="contains another dataset",
    ):

        _run_b12_boundary_with_controlled_output(
            monkeypatch,
            frame,
        )
