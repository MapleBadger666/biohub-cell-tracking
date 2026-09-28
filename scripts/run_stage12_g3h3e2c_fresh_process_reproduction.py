
from pathlib import Path

import ast
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    "/Users/heqiuyan/Desktop/biohub-cell-tracking"
)


SRC_ROOT = (
    PROJECT
    / "src"
)


S12 = (
    PROJECT
    / "reports"
    / "stage12_residual_audit"
)


CANONICAL = (
    PROJECT
    / "scripts"
    / "run_stage12_current_best_edge_residual_census.py"
)


BACKUP = (
    PROJECT
    / "scripts"
    / "run_stage12_current_best_edge_residual_census.pre_g3h3e1.py"
)


RUNTIME = (
    SRC_ROOT
    / "biohub_cell_tracking"
    / "stage12_confirmed_rewire.py"
)


INTEGRATION_MANIFEST = (
    S12
    / "stage12_g3h3e2b_canonical_runtime_integration.json"
)


FROZEN_ACTIONS_PATH = (
    S12
    / "stage12_fold0_rewire_asym1x5_q9990_no_target_cut_v1_actions_locked.pkl"
)


E0_METRICS_PATH = (
    S12
    / "stage12_fold0_rewire_asym1x5_q9990_no_target_cut_v1_confirmation_metrics.pkl"
)


E0_SUMMARY_PATH = (
    S12
    / "stage12_fold0_rewire_asym1x5_q9990_no_target_cut_v1_confirmation_summary.json"
)


PROMOTION_PATH = (
    S12
    / "stage12_rewire_asym1x5_q9990_no_target_cut_v1_promoted.json"
)


REPORT = (
    S12
    / "stage12_g3h3e2c_fresh_process_canonical_reproduction.json"
)


# ============================================================
# EXPECTED LOCKS
# ============================================================

POLICY_ID = (
    "ASYM1x5_Q9990_NO_TARGET_CUT_V1"
)


EXPECTED_RUNTIME_SHA = (
    "52c183629ecf80d0fa1568c8c3b2fa3a"
    "4ad6f58a1c34a98dc2bf59feca0ffad0"
)


EXPECTED_PRE_SHA = (
    "d85404884c4c74948676bbbefa4eafac"
    "2c5ac1ef9538e417ae52a4c04a9a7c60"
)


EXPECTED_POST_SHA = (
    "6f2bcc83715e305d0f01517a929382a8"
    "ac6f9f31a0bf32c29c11ed95c5be2181"
)


def sha256_file(path):

    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


# ============================================================
# FRESH-PROCESS PROOF
# ============================================================

print(
    "========== FRESH PROCESS =========="
)


print(
    "pid:",
    os.getpid()
)


print(
    "python:",
    sys.executable
)


print(
    "preloaded confirmed runtime:",
    (
        "biohub_cell_tracking.stage12_confirmed_rewire"
        in
        sys.modules
    )
)


assert (
    "biohub_cell_tracking.stage12_confirmed_rewire"
    not in
    sys.modules
)


if str(
    SRC_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            SRC_ROOT
        ),
    )


# ============================================================
# SOURCE LOCK
# ============================================================

assert sha256_file(
    RUNTIME
) == EXPECTED_RUNTIME_SHA


assert sha256_file(
    BACKUP
) == EXPECTED_PRE_SHA


assert sha256_file(
    CANONICAL
) == EXPECTED_POST_SHA


manifest = json.loads(
    INTEGRATION_MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert manifest[
    "policy_id"
] == POLICY_ID


assert manifest[
    "scientific_selection_closed"
] is True


assert manifest[
    "fold0_consumed"
] is True


assert manifest[
    "post_fold0_retuning_allowed"
] is False


print(
    "\n========== FRESH SOURCE LOCK =========="
)


print(
    "runtime:",
    EXPECTED_RUNTIME_SHA
)


print(
    "historical canonical:",
    EXPECTED_PRE_SHA
)


print(
    "integrated canonical:",
    EXPECTED_POST_SHA
)


print(
    "FRESH_SOURCE_LOCK: PASS"
)


# ============================================================
# SAFE PREFIX EXECUTION
#
# Execute each script only through the end of its relevant
# build_current_best definition.
#
# No evaluation code in the original residual-census script is
# executed here.
# ============================================================

def safe_builder_namespace(
    path,
    *,
    expected_builder_name,
):

    source = Path(
        path
    ).read_text(
        encoding="utf-8"
    )


    tree = ast.parse(
        source
    )


    matching = [
        node

        for node
        in tree.body

        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and
            node.name
            ==
            expected_builder_name
        )
    ]


    assert len(
        matching
    ) == 1, (
        path,
        expected_builder_name,
        len(
            matching
        ),
    )


    node = matching[
        0
    ]


    assert node.end_lineno is not None


    lines = source.splitlines()


    safe_prefix = "\n".join(
        lines[
            :
            node.end_lineno
        ]
    )


    # No GT evaluation in safe prefix.
    assert "fv9.evaluate_graph(" not in safe_prefix

    assert "official_evaluate(" not in safe_prefix

    assert "fv9.load_graph(" not in safe_prefix


    ns = {
        "__name__":
            (
                "__g3h3e2c_"
                +
                expected_builder_name
                +
                "__"
            ),

        "__file__":
            str(
                path
            ),
    }


    exec(
        compile(
            safe_prefix,
            str(
                path
            ),
            "exec",
        ),
        ns,
        ns,
    )


    assert expected_builder_name in ns

    assert callable(
        ns[
            expected_builder_name
        ]
    )


    return (
        ns,
        node.end_lineno,
    )


# ------------------------------------------------------------
# Historical exact CURRENT_BEST
# ------------------------------------------------------------

old_ns, old_end = safe_builder_namespace(
    BACKUP,
    expected_builder_name=
        "build_current_best",
)


old_builder = old_ns[
    "build_current_best"
]


# ------------------------------------------------------------
# Integrated CURRENT_BEST
#
# Current canonical contains:
#
#   _build_current_best_pre_g3h3e1
#   build_current_best
#
# Safe prefix through CURRENT build_current_best includes both.
# ------------------------------------------------------------

new_ns, new_end = safe_builder_namespace(
    CANONICAL,
    expected_builder_name=
        "build_current_best",
)


new_builder = new_ns[
    "build_current_best"
]


assert (
    "_build_current_best_pre_g3h3e1"
    in new_ns
)


print(
    "\n========== FRESH BUILDER RECOVERY =========="
)


print(
    "historical safe-prefix lines:",
    old_end
)


print(
    "integrated safe-prefix lines:",
    new_end
)


print(
    "integrated historical-base builder present:",
    "YES"
)


print(
    "GT touched:",
    "NO"
)


print(
    "FRESH_CANONICAL_BUILDER_RECOVERY: PASS"
)


# ============================================================
# CORPUS CONTRACT
# ============================================================

old_names_44 = list(
    old_ns[
        "names_44"
    ]
)


old_names_6 = list(
    old_ns[
        "names_6"
    ]
)


new_names_44 = list(
    new_ns[
        "names_44"
    ]
)


new_names_6 = list(
    new_ns[
        "names_6"
    ]
)


assert old_names_44 == new_names_44

assert old_names_6 == new_names_6


assert len(
    old_names_44
) == 15


assert len(
    old_names_6
) == 25


names = (
    old_names_44
    +
    old_names_6
)


assert len(
    names
) == 40


# ============================================================
# LOAD FROZEN REFERENCE ARTIFACTS
#
# These are already-consumed locked-test outputs.
# They are references for implementation fidelity, not model
# selection.
# ============================================================

frozen_actions = pd.read_pickle(
    FROZEN_ACTIONS_PATH
)


e0_metrics = pd.read_pickle(
    E0_METRICS_PATH
)


e0 = json.loads(
    E0_SUMMARY_PATH.read_text(
        encoding="utf-8"
    )
)


promotion = json.loads(
    PROMOTION_PATH.read_text(
        encoding="utf-8"
    )
)


assert len(
    frozen_actions
) == 2712


assert len(
    e0_metrics
) == 40


assert e0[
    "confirmed"
] is True


assert e0[
    "decision"
] == (
    "LOCKED_FOLD0_CONFIRMED"
)


assert promotion[
    "status"
] == (
    "CONFIRMED_POLICY_PROMOTED"
)


# ============================================================
# GRAPH HELPERS
# ============================================================

def edge_pairs(
    graph,
):

    edges = (
        graph.edge_attrs(
            unpack=True
        )
        .to_pandas()
    )


    return set(
        zip(
            edges[
                "source_id"
            ].astype(
                int
            ),

            edges[
                "target_id"
            ].astype(
                int
            ),
        )
    )


def node_ids(
    graph,
):

    nodes = (
        graph.node_attrs(
            attr_keys=[
                "node_id",
            ]
        )
        .to_pandas()
    )


    return set(
        nodes[
            "node_id"
        ].astype(
            int
        )
    )


# ============================================================
# GT-FREE FULL-40 TOPOLOGY FIDELITY
#
# Build both historical and integrated graphs first.
#
# Compare topology to frozen Fold0 action artifact BEFORE
# loading any GT.
# ============================================================

print(
    "\n============================================================"
)


print(
    "PART A — FRESH FULL-40 GT-FREE TOPOLOGY FIDELITY"
)


print(
    "============================================================"
)


graph_pairs = {}

topology_rows = []


for i, name in enumerate(
    names,
    start=1,
):

    old_graph, old_scale = (
        old_builder(
            name
        )
    )


    new_graph, new_scale = (
        new_builder(
            name
        )
    )


    assert np.allclose(
        np.asarray(
            old_scale,
            dtype=float,
        ),
        np.asarray(
            new_scale,
            dtype=float,
        ),
        atol=1e-12,
        rtol=0.0,
    )


    old_nodes = node_ids(
        old_graph
    )


    new_nodes = node_ids(
        new_graph
    )


    # Confirmed rewire adds/removes edges only.
    assert old_nodes == new_nodes, name


    old_edges = edge_pairs(
        old_graph
    )


    new_edges = edge_pairs(
        new_graph
    )


    added = (
        new_edges
        -
        old_edges
    )


    removed = (
        old_edges
        -
        new_edges
    )


    if name.startswith(
        "44b6_"
    ):

        expected_actions = (
            frozen_actions.iloc[
                0:0
            ]
        )


        assert len(
            added
        ) == 0, name


        assert len(
            removed
        ) == 0, name


        assert old_edges == new_edges, name


        expected_action_count = 0

        expected_source_occ = 0


    else:

        assert name.startswith(
            "6bba_"
        )


        expected_actions = (
            frozen_actions[
                frozen_actions[
                    "dataset"
                ].eq(
                    name
                )
            ]
            .copy()
        )


        expected_pairs = set(
            zip(
                expected_actions[
                    "source_id"
                ].astype(
                    int
                ),

                expected_actions[
                    "target_id"
                ].astype(
                    int
                ),
            )
        )


        expected_action_count = int(
            len(
                expected_actions
            )
        )


        expected_source_occ = int(
            pd.to_numeric(
                expected_actions[
                    "source_occupied"
                ],
                errors="raise",
            )
            .astype(
                int
            )
            .sum()
        )


        # Exact action-edge identity.
        assert added == expected_pairs, (
            name,
            len(
                added
            ),
            len(
                expected_pairs
            ),
        )


        # NO_TARGET_CUT only removes a source incumbent.
        assert len(
            removed
        ) == expected_source_occ, (
            name,
            len(
                removed
            ),
            expected_source_occ,
        )


        assert (
            len(
                new_edges
            )
            -
            len(
                old_edges
            )
        ) == (
            expected_action_count
            -
            expected_source_occ
        )


    topology_rows.append(
        {
            "dataset":
                name,

            "prefix":
                (
                    "44b6"
                    if name.startswith(
                        "44b6_"
                    )
                    else
                    "6bba"
                ),

            "expected_actions":
                expected_action_count,

            "added_edges":
                int(
                    len(
                        added
                    )
                ),

            "expected_source_occupied":
                expected_source_occ,

            "removed_edges":
                int(
                    len(
                        removed
                    )
                ),

            "node_set_exact":
                True,

            "scale_exact":
                True,
        }
    )


    # Preserve graph objects for official evaluation only after
    # all GT-free topology checks for this dataset have passed.
    graph_pairs[
        name
    ] = (
        old_graph,
        new_graph,
        old_scale,
    )


    print(
        f"{i:2d}/40"
        f" | {name}"
        f" | actions={expected_action_count}"
        f" | added={len(added)}"
        f" | removed={len(removed)}"
    )


topology_df = pd.DataFrame(
    topology_rows
)


assert int(
    topology_df[
        "expected_actions"
    ].sum()
) == 2712


assert int(
    topology_df[
        "added_edges"
    ].sum()
) == 2712


assert int(
    topology_df[
        "expected_source_occupied"
    ].sum()
) == 82


assert int(
    topology_df[
        "removed_edges"
    ].sum()
) == 82


print(
    "\n========== FRESH TOPOLOGY FIDELITY SUMMARY =========="
)


print(
    "datasets:",
    len(
        topology_df
    )
)


print(
    "frozen actions:",
    int(
        topology_df[
            "expected_actions"
        ].sum()
    )
)


print(
    "exact added edges:",
    int(
        topology_df[
            "added_edges"
        ].sum()
    )
)


print(
    "frozen source-occupied actions:",
    int(
        topology_df[
            "expected_source_occupied"
        ].sum()
    )
)


print(
    "exact removed edges:",
    int(
        topology_df[
            "removed_edges"
        ].sum()
    )
)


print(
    "GT loaded:",
    "NO"
)


print(
    "FULL40_CANONICAL_TOPOLOGY_FIDELITY: PASS"
)


# ============================================================
# PART B — OFFICIAL METRIC REPRODUCTION
#
# Fold0 was already consumed in G3H3E0.
# These GT reads are strictly implementation-fidelity checks.
# ============================================================

print(
    "\n============================================================"
)


print(
    "PART B — CONSUMED FOLD0 IMPLEMENTATION REPRODUCTION"
)


print(
    "============================================================"
)


fv9 = new_ns[
    "fv9"
]


TRAIN_ROOT = new_ns[
    "TRAIN_ROOT"
]


estimated_nodes = new_ns[
    "estimated_nodes"
]


fresh_rows = []

baseline_records = []

policy_records = []


for i, name in enumerate(
    names,
    start=1,
):

    (
        old_graph,
        new_graph,
        scale,
    ) = graph_pairs[
        name
    ]


    gt = fv9.load_graph(
        TRAIN_ROOT
        / f"{name}.geff"
    )


    n_total = float(
        estimated_nodes[
            name
        ]
    )


    # Evaluator mutates predictions, so evaluate copies.
    old_eval_graph = old_graph.copy()

    new_eval_graph = new_graph.copy()


    base_metric = fv9.evaluate_graph(
        old_eval_graph,
        gt,
        scale,
        n_total,
        fv9.FROZEN_V9_CONFIG,
    )


    policy_metric = fv9.evaluate_graph(
        new_eval_graph,
        gt,
        scale,
        n_total,
        fv9.FROZEN_V9_CONFIG,
    )


    baseline_records.append(
        base_metric
    )


    policy_records.append(
        policy_metric
    )


    fresh_rows.append(
        {
            "dataset":
                name,

            "base_edge_tp":
                int(
                    base_metric[
                        "edge_tp"
                    ]
                ),

            "base_edge_fp":
                int(
                    base_metric[
                        "edge_fp"
                    ]
                ),

            "base_edge_fn":
                int(
                    base_metric[
                        "edge_fn"
                    ]
                ),

            "base_edge_jaccard":
                float(
                    base_metric[
                        "edge_jaccard"
                    ]
                ),

            "base_adj_edge_jaccard":
                float(
                    base_metric[
                        "adj_edge_jaccard"
                    ]
                ),

            "base_node_recall":
                float(
                    base_metric[
                        "node_recall"
                    ]
                ),

            "policy_edge_tp":
                int(
                    policy_metric[
                        "edge_tp"
                    ]
                ),

            "policy_edge_fp":
                int(
                    policy_metric[
                        "edge_fp"
                    ]
                ),

            "policy_edge_fn":
                int(
                    policy_metric[
                        "edge_fn"
                    ]
                ),

            "policy_edge_jaccard":
                float(
                    policy_metric[
                        "edge_jaccard"
                    ]
                ),

            "policy_adj_edge_jaccard":
                float(
                    policy_metric[
                        "adj_edge_jaccard"
                    ]
                ),

            "policy_node_recall":
                float(
                    policy_metric[
                        "node_recall"
                    ]
                ),
        }
    )


    print(
        f"{i:2d}/40"
        f" | {name}"
        f" | base TP/FP/FN="
        f"{base_metric['edge_tp']}/"
        f"{base_metric['edge_fp']}/"
        f"{base_metric['edge_fn']}"
        f" | policy="
        f"{policy_metric['edge_tp']}/"
        f"{policy_metric['edge_fp']}/"
        f"{policy_metric['edge_fn']}"
    )


fresh_df = pd.DataFrame(
    fresh_rows
)


# ============================================================
# DATASET-BY-DATASET FIDELITY AGAINST E0
# ============================================================

reference = (
    e0_metrics[
        [
            "dataset",

            "base_edge_tp",
            "base_edge_fp",
            "base_edge_fn",
            "base_edge_jaccard",
            "base_adj_edge_jaccard",
            "base_node_recall",

            "policy_edge_tp",
            "policy_edge_fp",
            "policy_edge_fn",
            "policy_edge_jaccard",
            "policy_adj_edge_jaccard",
            "policy_node_recall",
        ]
    ]
    .sort_values(
        "dataset",
        kind="stable",
    )
    .reset_index(
        drop=True
    )
)


fresh_cmp = (
    fresh_df[
        reference.columns.tolist()
    ]
    .sort_values(
        "dataset",
        kind="stable",
    )
    .reset_index(
        drop=True
    )
)


assert fresh_cmp[
    "dataset"
].tolist() == reference[
    "dataset"
].tolist()


integer_cols = [
    "base_edge_tp",
    "base_edge_fp",
    "base_edge_fn",

    "policy_edge_tp",
    "policy_edge_fp",
    "policy_edge_fn",
]


for col in integer_cols:

    assert np.array_equal(
        fresh_cmp[
            col
        ].to_numpy(
            dtype=int
        ),
        reference[
            col
        ].to_numpy(
            dtype=int
        ),
    ), col


float_cols = [
    "base_edge_jaccard",
    "base_adj_edge_jaccard",
    "base_node_recall",

    "policy_edge_jaccard",
    "policy_adj_edge_jaccard",
    "policy_node_recall",
]


max_metric_diff = {}


for col in float_cols:

    fresh_values = fresh_cmp[
        col
    ].to_numpy(
        dtype=float
    )


    reference_values = reference[
        col
    ].to_numpy(
        dtype=float
    )


    max_diff = float(
        np.max(
            np.abs(
                fresh_values
                -
                reference_values
            )
        )
    )


    max_metric_diff[
        col
    ] = max_diff


    assert np.allclose(
        fresh_values,
        reference_values,
        rtol=0.0,
        atol=1e-12,
    ), (
        col,
        max_diff,
    )


print(
    "\n========== DATASET-LEVEL E0 REPRODUCTION =========="
)


print(
    "integer metrics exact:",
    "YES"
)


print(
    "max absolute metric diffs:"
)


for key, value in max_metric_diff.items():

    print(
        f"  {key}: {value}"
    )


print(
    "DATASET_LEVEL_IMPLEMENTATION_FIDELITY: PASS"
)


# ============================================================
# AGGREGATE FIDELITY
# ============================================================

baseline_summary = fv9.summarise(
    baseline_records
)


policy_summary = fv9.summarise(
    policy_records
)


base_tp = int(
    fresh_df[
        "base_edge_tp"
    ].sum()
)


base_fp = int(
    fresh_df[
        "base_edge_fp"
    ].sum()
)


base_fn = int(
    fresh_df[
        "base_edge_fn"
    ].sum()
)


policy_tp = int(
    fresh_df[
        "policy_edge_tp"
    ].sum()
)


policy_fp = int(
    fresh_df[
        "policy_edge_fp"
    ].sum()
)


policy_fn = int(
    fresh_df[
        "policy_edge_fn"
    ].sum()
)


assert base_tp == int(
    e0[
        "baseline"
    ][
        "edge_tp"
    ]
)


assert base_fp == int(
    e0[
        "baseline"
    ][
        "edge_fp"
    ]
)


assert base_fn == int(
    e0[
        "baseline"
    ][
        "edge_fn"
    ]
)


assert policy_tp == int(
    e0[
        "policy"
    ][
        "edge_tp"
    ]
)


assert policy_fp == int(
    e0[
        "policy"
    ][
        "edge_fp"
    ]
)


assert policy_fn == int(
    e0[
        "policy"
    ][
        "edge_fn"
    ]
)


aggregate_fields = [
    "edge_jaccard",
    "adj_edge_jaccard",
    "division_jaccard",
    "node_recall",
    "score",
]


aggregate_diffs = {}


for field in aggregate_fields:

    b_diff = abs(
        float(
            baseline_summary[
                field
            ]
        )
        -
        float(
            e0[
                "baseline"
            ][
                field
            ]
        )
    )


    p_diff = abs(
        float(
            policy_summary[
                field
            ]
        )
        -
        float(
            e0[
                "policy"
            ][
                field
            ]
        )
    )


    aggregate_diffs[
        f"baseline_{field}"
    ] = b_diff


    aggregate_diffs[
        f"policy_{field}"
    ] = p_diff


    assert b_diff <= 1e-12, (
        field,
        "baseline",
        b_diff,
    )


    assert p_diff <= 1e-12, (
        field,
        "policy",
        p_diff,
    )


delta_score = float(
    policy_summary[
        "score"
    ]
    -
    baseline_summary[
        "score"
    ]
)


assert abs(
    delta_score
    -
    float(
        e0[
            "delta"
        ][
            "score"
        ]
    )
) <= 1e-12


print(
    "\n========== FRESH AGGREGATE REPRODUCTION =========="
)


print(
    "baseline TP/FP/FN:",
    base_tp,
    "/",
    base_fp,
    "/",
    base_fn
)


print(
    "expected:",
    e0[
        "baseline"
    ][
        "edge_tp"
    ],
    "/",
    e0[
        "baseline"
    ][
        "edge_fp"
    ],
    "/",
    e0[
        "baseline"
    ][
        "edge_fn"
    ],
)


print(
    "policy TP/FP/FN:",
    policy_tp,
    "/",
    policy_fp,
    "/",
    policy_fn
)


print(
    "expected:",
    e0[
        "policy"
    ][
        "edge_tp"
    ],
    "/",
    e0[
        "policy"
    ][
        "edge_fp"
    ],
    "/",
    e0[
        "policy"
    ][
        "edge_fn"
    ],
)


print(
    "baseline score:",
    f"{baseline_summary['score']:.15f}"
)


print(
    "expected:",
    f"{float(e0['baseline']['score']):.15f}"
)


print(
    "policy score:",
    f"{policy_summary['score']:.15f}"
)


print(
    "expected:",
    f"{float(e0['policy']['score']):.15f}"
)


print(
    "delta:",
    f"{delta_score:+.15f}"
)


print(
    "expected:",
    f"{float(e0['delta']['score']):+.15f}"
)


print(
    "FRESH_AGGREGATE_IMPLEMENTATION_FIDELITY: PASS"
)


# ============================================================
# FORMAL REPORT
# ============================================================

REPORT_DATA = {
    "stage":
        "12.11G3H3E2C",

    "status":
        "FRESH_PROCESS_CANONICAL_REPRODUCTION_PASS",

    "purpose":
        "IMPLEMENTATION_FIDELITY_ONLY",

    "policy_id":
        POLICY_ID,

    "scientific_selection_closed":
        True,

    "fold0_consumed_before_this_stage":
        True,

    "post_fold0_retuning_allowed":
        False,

    "source_integrity": {
        "runtime_sha256":
            EXPECTED_RUNTIME_SHA,

        "historical_canonical_sha256":
            EXPECTED_PRE_SHA,

        "integrated_canonical_sha256":
            EXPECTED_POST_SHA,
    },

    "fresh_process": {
        "pid":
            os.getpid(),

        "python":
            sys.executable,

        "runtime_preloaded":
            False,
    },

    "corpus": {
        "datasets":
            40,

        "44b6":
            15,

        "6bba":
            25,
    },

    "topology_fidelity": {
        "frozen_actions":
            2712,

        "exact_added_edges":
            int(
                topology_df[
                    "added_edges"
                ].sum()
            ),

        "frozen_source_occupied_actions":
            82,

        "exact_removed_edges":
            int(
                topology_df[
                    "removed_edges"
                ].sum()
            ),

        "node_sets_exact":
            True,

        "scales_exact":
            True,

        "all_action_edge_pairs_exact":
            True,
    },

    "dataset_metric_fidelity": {
        "integer_metrics_exact":
            True,

        "max_absolute_float_differences":
            max_metric_diff,
    },

    "aggregate": {
        "baseline": {
            "edge_tp":
                base_tp,

            "edge_fp":
                base_fp,

            "edge_fn":
                base_fn,

            "score":
                float(
                    baseline_summary[
                        "score"
                    ]
                ),
        },

        "policy": {
            "edge_tp":
                policy_tp,

            "edge_fp":
                policy_fp,

            "edge_fn":
                policy_fn,

            "score":
                float(
                    policy_summary[
                        "score"
                    ]
                ),
        },

        "delta_score":
            delta_score,

        "aggregate_metric_absolute_differences":
            aggregate_diffs,
    },

    "matches_consumed_e0_exactly":
        True,

    "source_modified":
        False,

    "prediction_geff_written":
        False,

    "development_reopened":
        False,

    "next":
        "CLOSE_STAGE12_REWIRE_INTEGRATION_AND_ADVANCE",
}


REPORT.write_text(
    json.dumps(
        REPORT_DATA,
        indent=2,
    ),
    encoding="utf-8",
)


assert REPORT.exists()


print(
    "\n========== 12.11G3H3E2C DECISION =========="
)


print(
    "FRESH_PROCESS_CANONICAL_REPRODUCTION: PASS"
)


print(
    "implementation fidelity:",
    "EXACT"
)


print(
    "all 2712 action edge pairs reproduced:",
    "YES"
)


print(
    "all 82 source-incumbent removals reproduced:",
    "YES"
)


print(
    "dataset-level official metrics reproduced:",
    "YES"
)


print(
    "aggregate locked result reproduced:",
    "YES"
)


print(
    "policy score:",
    f"{policy_summary['score']:.15f}"
)


print(
    "delta:",
    f"{delta_score:+.15f}"
)


print(
    "scientific selection reopened:",
    "NO"
)


print(
    "post-Fold0 tuning allowed:",
    "NO"
)


print(
    "source modified:",
    "NO"
)


print(
    "prediction GEFF written:",
    "NO"
)


print(
    "next:"
)


print(
    "CLOSE_STAGE12_REWIRE_INTEGRATION_AND_ADVANCE"
)
