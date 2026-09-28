"""Frozen V9 graph pipeline — reusable, prediction/model-agnostic.

This module migrates the *already validated* Frozen V9 pipeline out of the
Stage09/Stage10 notebooks so it can be reused without re-deriving it. Function
bodies are migrated verbatim; the only changes are mechanical:

- notebook globals (prediction roots, GT root, prefix map, candidate tables)
  became explicit arguments, and
- frozen thresholds became :class:`FrozenV9Config` fields.

No scientific or algorithmic behavior was changed. Provenance for each migrated
function is recorded in its own comment.

Pipeline order is frozen and must not be reordered::

    G2 division policy
      -> K9 confidence-aware pruning
      -> single-edge gap recovery
      -> strict two-edge isolated-middle recovery
      -> component cleanup
      -> locked official metric

Deliberately absent (rejected experimental variants): synthetic node insertion,
motion correction, ILP association, endpoint trim, reverse repair order, and any
prefix-specific graph policy.

The module is model agnostic: candidate tables are regenerated from *this*
prediction set's node IDs on every call. Stage09 candidate tables tied to Warm2
node IDs are never reused, and nothing here mutates the source prediction GEFFs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tracksdata as td
from scipy.spatial import cKDTree
from tracking_cellmot.io import open_dataset
from tracking_cellmot.metrics import evaluate as official_evaluate
from tracking_cellmot.metrics import node_recall, per_sample_metrics, summarise

__all__ = [
    "FROZEN_V9_CONFIG",
    "FrozenV9Application",
    "FrozenV9Config",
    "FrozenV9Evaluation",
    "FrozenV9Policies",
    "add_available_repairs",
    "add_two_edge_bridges",
    "apply_frozen_component_cleanup",
    "apply_frozen_g2",
    "apply_frozen_v9",
    "apply_low_conf_component_pruning",
    "build_frozen_v9_policies",
    "build_mutual_policy",
    "build_strict_two_edge_policy",
    "census_isolated_bridges",
    "census_single_edge_gaps",
    "component_sets",
    "evaluate_frozen_v9",
    "load_graph",
    "node_coordinate_keys",
    "prepare_precleanup_graph",
    "recovered_edge_dist",
    "summarize_by_prefix",
]


@dataclass(frozen=True)
class FrozenV9Config:
    """Immutable Frozen V9 parameters.

    Every scientific threshold in the pipeline lives here. The defaults are the
    frozen V9 values that produced the locked Warm2 Fold0 reproduction; changing
    any of them produces a different pipeline, not Frozen V9.
    """

    # Detection thresholds naming the two prediction sets the pipeline consumes.
    det_threshold: float = 0.995
    high_threshold: float = 0.9975

    # K9 confidence-aware pruning: only components this size or smaller may lose
    # their low-confidence nodes.
    confidence_component_limit: int = 9

    # G2 division policy.
    division_p2_min: float = 0.80
    division_daughter_sep_min_um: float = 6.0
    division_angle_min_deg: float = 120.0
    division_max_parent_daughter_um: float = 8.0
    division_distance_balance_max: float = 0.50

    # Shared single-edge and strict two-edge recovery geometry.
    recovery_residual_max_um: float = 4.0
    recovery_max_step_um: float = 8.0
    recovery_step_balance_max: float = 0.50
    recovery_turn_angle_max_deg: float = 60.0
    recovery_ambiguity_margin_min_um: float = 1.0

    # Strict two-edge recovery additionally requires a unique local pair.
    two_edge_required_pair_count: int = 1

    # Component cleanup.
    cleanup_two_node_max_um: float = 3.5

    # Structural: model inference grid, used to report edge_dist on model scale.
    model_downsample_zyx: tuple[float, float, float] = (1.0, 4.0, 4.0)

    # Locked official metric matching radius.
    metric_max_distance: float = 7.0


FROZEN_V9_CONFIG = FrozenV9Config()


def load_graph(path: Path):
    # Verbatim body migrated from 07_detection_threshold_sweep.ipynb cell 0 (load_graph).
    """Load a GEFF into a tracksdata graph.

    Exact loader behavior validated in Stage07/Stage09: ``from_geff`` may return
    either a graph or a ``(graph, metadata)`` tuple.
    """

    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def component_sets(graph) -> list[list[int]]:
    # Verbatim body migrated from 07_detection_threshold_sweep.ipynb cell 3 (component_sets).
    node_ids = [int(x) for x in graph.node_attrs(unpack=True)["node_id"].to_list()]

    parent = {n: n for n in node_ids}

    size = {n: 1 for n in node_ids}  # noqa: C420 - verbatim migrated body

    def find(x):
        root = x

        while parent[root] != root:
            root = parent[root]

        while parent[x] != x:
            nxt = parent[x]
            parent[x] = root
            x = nxt

        return root

    def union(a, b):
        ra = find(a)
        rb = find(b)

        if ra == rb:
            return

        if size[ra] < size[rb]:
            ra, rb = rb, ra

        parent[rb] = ra
        size[ra] += size[rb]

    for source, target in graph.edge_list():
        union(
            int(source),
            int(target),
        )

    components = defaultdict(list)

    for node_id in node_ids:
        components[find(node_id)].append(node_id)

    return list(components.values())


def node_coordinate_keys(graph) -> set[tuple[int, int, int, int]]:
    # Verbatim body migrated from 07_detection_threshold_sweep.ipynb cell 16 (node_coordinate_keys).
    df = graph.node_attrs(unpack=True).to_pandas()

    keys = {
        (
            int(row.t),
            int(round(row.z)),  # noqa: RUF046 - verbatim migrated body
            int(round(row.y)),  # noqa: RUF046 - verbatim migrated body
            int(round(row.x)),  # noqa: RUF046 - verbatim migrated body
        )
        for row in df.itertuples(index=False)
    }

    return keys


def apply_frozen_g2(graph, scale, config: FrozenV9Config = FROZEN_V9_CONFIG) -> int:
    # Verbatim body migrated from 07_detection_threshold_sweep.ipynb cell 0 (apply_frozen_g2), with frozen G2
    # thresholds bound from config.
    G2_P2_MIN = config.division_p2_min
    G2_DAUGHTER_SEP_MIN_UM = config.division_daughter_sep_min_um
    G2_ANGLE_MIN_DEG = config.division_angle_min_deg
    G2_MAX_PARENT_DAUGHTER_UM = config.division_max_parent_daughter_um
    G2_DISTANCE_BALANCE_MAX = config.division_distance_balance_max

    """
    Frozen G2 division filter.

    For every predicted fork:
      - keep highest-probability child
      - keep second child only if G2 geometry passes
    """

    scale = np.asarray(
        scale,
        dtype=float,
    )

    nodes = graph.node_attrs(unpack=True).to_pandas().set_index("node_id")

    edges = graph.edge_attrs(
        attr_keys=["edge_prob"],
        unpack=True,
    ).to_pandas()

    remove_ids = []
    forks_kept = 0

    for source_id, group in edges.groupby("source_id"):
        if len(group) != 2:
            continue

        group = group.sort_values(
            ["edge_prob", "edge_id"],
            ascending=[False, True],
        )

        e1 = group.iloc[0]
        e2 = group.iloc[1]

        source_id = int(source_id)
        target1 = int(e1["target_id"])
        target2 = int(e2["target_id"])

        parent = nodes.loc[source_id]
        child1 = nodes.loc[target1]
        child2 = nodes.loc[target2]

        p = (
            np.array(
                [parent["z"], parent["y"], parent["x"]],
                dtype=float,
            )
            * scale
        )

        c1 = (
            np.array(
                [child1["z"], child1["y"], child1["x"]],
                dtype=float,
            )
            * scale
        )

        c2 = (
            np.array(
                [child2["z"], child2["y"], child2["x"]],
                dtype=float,
            )
            * scale
        )

        v1 = c1 - p
        v2 = c2 - p

        d1 = float(np.linalg.norm(v1))
        d2 = float(np.linalg.norm(v2))

        daughter_sep = float(np.linalg.norm(c1 - c2))

        if d1 > 0 and d2 > 0:
            cosine = float(np.dot(v1, v2) / (d1 * d2))

            cosine = float(
                np.clip(
                    cosine,
                    -1.0,
                    1.0,
                )
            )

            angle = float(np.degrees(np.arccos(cosine)))
        else:
            angle = np.nan

        distance_balance = abs(d1 - d2) / max(d1 + d2, 1e-12)

        keep_second = (
            float(e2["edge_prob"]) >= G2_P2_MIN
            and daughter_sep >= G2_DAUGHTER_SEP_MIN_UM
            and np.isfinite(angle)
            and angle >= G2_ANGLE_MIN_DEG
            and max(d1, d2) <= G2_MAX_PARENT_DAUGHTER_UM
            and distance_balance <= G2_DISTANCE_BALANCE_MAX
        )

        if keep_second:
            forks_kept += 1
        else:
            remove_ids.append(int(e2["edge_id"]))

    if remove_ids:
        graph.bulk_remove_edges(remove_ids)

    return forks_kept


def apply_low_conf_component_pruning(
    graph,
    high_conf_coordinate_set,
    max_component_size,
) -> dict:
    # Verbatim body migrated from 07_detection_threshold_sweep.ipynb cell 19 (apply_low_conf_component_pruning).
    """
    Remove low-confidence nodes iff their component size
    BEFORE pruning is <= max_component_size.

    low confidence means:
      node exists at det=0.995
      but not at det=0.9975
    """

    if max_component_size == 0:
        return {
            "low_conf_nodes_removed": 0,
            "affected_components": 0,
        }

    nodes = graph.node_attrs(unpack=True).to_pandas().set_index("node_id")

    comps = component_sets(graph)

    remove_nodes = set()
    affected_components = 0

    for comp in comps:
        comp = list(map(int, comp))

        comp_size = len(comp)

        if not np.isinf(max_component_size) and comp_size > max_component_size:
            continue

        component_remove = []

        for node_id in comp:
            row = nodes.loc[node_id]

            key = (
                int(row["t"]),
                int(round(row["z"])),  # noqa: RUF046 - verbatim migrated body
                int(round(row["y"])),  # noqa: RUF046 - verbatim migrated body
                int(round(row["x"])),  # noqa: RUF046 - verbatim migrated body
            )

            is_high_conf = key in high_conf_coordinate_set

            if not is_high_conf:
                component_remove.append(node_id)

        if component_remove:
            affected_components += 1

            remove_nodes.update(component_remove)

    if remove_nodes:
        graph.bulk_remove_nodes(sorted(remove_nodes))

    return {
        "low_conf_nodes_removed": len(remove_nodes),
        "affected_components": affected_components,
    }


def apply_frozen_component_cleanup(
    graph,
    scale,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
) -> dict:
    # Verbatim body migrated from 07_detection_threshold_sweep.ipynb cell 19 (apply_frozen_component_cleanup).
    TWO_NODE_MAX_DIST_UM = config.cleanup_two_node_max_um

    scale = np.asarray(
        scale,
        dtype=float,
    )

    nodes = graph.node_attrs(unpack=True).to_pandas().set_index("node_id")

    edges = graph.edge_attrs(
        attr_keys=["edge_prob"],
        unpack=True,
    ).to_pandas()

    edge_lookup = {}

    for _, e in edges.iterrows():
        s = int(e["source_id"])
        t = int(e["target_id"])

        edge_lookup[frozenset([s, t])] = (s, t)

    comps = component_sets(graph)

    remove_nodes = set()

    singleton_components = 0
    long_two_node_components = 0

    for comp in comps:
        comp = list(map(int, comp))

        # -----------------------------------------
        # Singleton
        # -----------------------------------------

        if len(comp) == 1:
            remove_nodes.add(comp[0])

            singleton_components += 1

        # -----------------------------------------
        # 2-node component
        # -----------------------------------------

        elif len(comp) == 2:
            key = frozenset(comp)

            if key not in edge_lookup:
                continue

            source, target = edge_lookup[key]

            a = nodes.loc[source]
            b = nodes.loc[target]

            delta_um = (
                np.array(
                    [
                        float(b["z"] - a["z"]),
                        float(b["y"] - a["y"]),
                        float(b["x"] - a["x"]),
                    ]
                )
                * scale
            )

            dist_um = float(np.linalg.norm(delta_um))

            if dist_um > TWO_NODE_MAX_DIST_UM:
                remove_nodes.update(comp)

                long_two_node_components += 1

    if remove_nodes:
        graph.bulk_remove_nodes(sorted(remove_nodes))

    return {
        "nodes_removed": len(remove_nodes),
        "singleton_components": singleton_components,
        "long_two_node_components": long_two_node_components,
    }


def turn_angle_deg(a, b, c) -> float:
    # Verbatim body migrated from 09_gap_recovery.ipynb cell 2 (turn_angle_deg).
    """
    0 degrees = perfectly straight continuation:
        A -> B -> C
    """

    v1 = b - a
    v2 = c - b

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 <= 1e-12 or n2 <= 1e-12:
        return np.nan

    cosine = np.dot(
        v1,
        v2,
    ) / (n1 * n2)

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return float(np.degrees(np.arccos(cosine)))


def step_balance(d1, d2) -> float:
    # Verbatim body migrated from 09_gap_recovery.ipynb cell 2 (step_balance).
    denom = max(
        d1,
        d2,
        1e-12,
    )

    return abs(d1 - d2) / denom


def turn_angle_deg_0910(a, b, c) -> float:
    # Verbatim body migrated from 09_gap_recovery.ipynb cell 10 (turn_angle_deg_0910).
    """
    0 deg = straight continuation A -> B -> C.
    """

    v1 = b - a
    v2 = c - b

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 <= 1e-12 or n2 <= 1e-12:
        return np.nan

    cosine = np.dot(
        v1,
        v2,
    ) / (n1 * n2)

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return float(np.degrees(np.arccos(cosine)))


def step_balance_0910(d1, d2) -> float:
    # Verbatim body migrated from 09_gap_recovery.ipynb cell 10 (step_balance_0910).
    return float(
        abs(d1 - d2)
        / max(
            d1,
            d2,
            1e-12,
        )
    )


def recovered_edge_dist(
    node_table,
    source_id,
    target_id,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
) -> float:
    # Verbatim body migrated from 09_gap_recovery.ipynb cell 5 (recovered_edge_dist).
    MODEL_DOWNSAMPLE_ZYX = np.asarray(config.model_downsample_zyx, dtype=float)

    a = node_table.loc[source_id]

    b = node_table.loc[target_id]

    delta_zyx = np.array(
        [
            float(b["z"] - a["z"]),
            float(b["y"] - a["y"]),
            float(b["x"] - a["x"]),
        ],
        dtype=float,
    )

    return float(np.linalg.norm(delta_zyx / MODEL_DOWNSAMPLE_ZYX))


def prepare_precleanup_graph(
    name: str,
    pred_995_dir: Path,
    pred_9975_dir: Path,
    train_dir: Path,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
):
    # Verbatim body migrated from 09_gap_recovery.ipynb cell 5 (prepare_precleanup_graph), with prediction/GT
    # roots and K bound from arguments.
    WARM2_0995 = Path(pred_995_dir)
    WARM2_09975 = Path(pred_9975_dir)
    TRAIN_DIR = Path(train_dir)
    CONF_COMPONENT_LIMIT = config.confidence_component_limit

    g = load_graph(WARM2_0995 / f"{name}.geff")

    high_g = load_graph(WARM2_09975 / f"{name}.geff")

    scale = np.asarray(
        open_dataset(
            TRAIN_DIR / name,
            load_image=False,
        ).scale,
        dtype=float,
    )

    apply_frozen_g2(
        g,
        scale,
        config,
    )

    high_keys = node_coordinate_keys(high_g)

    apply_low_conf_component_pruning(
        g,
        high_keys,
        CONF_COMPONENT_LIMIT,
    )

    return (
        g,
        scale,
    )


def census_single_edge_gaps(
    name: str,
    pred_995_dir: Path,
    pred_9975_dir: Path,
    train_dir: Path,
    prefix: str,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
):
    # Verbatim body migrated from 09_gap_recovery.ipynb cell 2 (census_single_edge_gaps), with prediction/GT
    # roots, prefix and K bound from arguments.
    WARM2_0995 = Path(pred_995_dir)
    WARM2_09975 = Path(pred_9975_dir)
    TRAIN_DIR = Path(train_dir)
    CONF_COMPONENT_LIMIT = config.confidence_component_limit
    prefix_map = {name: prefix}

    g = load_graph(WARM2_0995 / f"{name}.geff")

    high_g = load_graph(WARM2_09975 / f"{name}.geff")

    scale = np.asarray(
        open_dataset(
            TRAIN_DIR / name,
            load_image=False,
        ).scale,
        dtype=float,
    )

    # Expected physical-axis order is z, y, x.
    assert len(scale) == 3

    # --------------------------------------------------------
    # Baseline-v7 BEFORE final cleanup
    # --------------------------------------------------------

    apply_frozen_g2(
        g,
        scale,
        config,
    )

    high_keys = node_coordinate_keys(high_g)

    apply_low_conf_component_pruning(
        g,
        high_keys,
        CONF_COMPONENT_LIMIT,
    )

    # --------------------------------------------------------
    # Node table after G2 + K9
    # --------------------------------------------------------

    attrs = g.node_attrs(unpack=True)

    node_ids = attrs["node_id"].to_list()

    t_arr = attrs["t"].to_numpy().astype(int)

    z_arr = attrs["z"].to_numpy().astype(float)

    y_arr = attrs["y"].to_numpy().astype(float)

    x_arr = attrs["x"].to_numpy().astype(float)

    # Physical coordinates in microns: z, y, x.
    pos = np.column_stack(
        [
            z_arr * scale[0],
            y_arr * scale[1],
            x_arr * scale[2],
        ]
    )

    id_to_i = {node_id: i for i, node_id in enumerate(node_ids)}

    in_deg = np.asarray(
        g.in_degree(node_ids),
        dtype=int,
    )

    out_deg = np.asarray(
        g.out_degree(node_ids),
        dtype=int,
    )

    # --------------------------------------------------------
    # Frame-indexed track endpoints / starts
    # --------------------------------------------------------

    ends_by_t = {}
    starts_by_t = {}

    for t in np.unique(t_arr):
        frame_idx = np.flatnonzero(t_arr == t)

        ends = frame_idx[out_deg[frame_idx] == 0]

        starts = frame_idx[in_deg[frame_idx] == 0]

        ends_by_t[int(t)] = ends
        starts_by_t[int(t)] = starts

    # --------------------------------------------------------
    # KD trees
    # --------------------------------------------------------

    end_trees = {}
    start_trees = {}

    for t, idx in ends_by_t.items():
        if len(idx):
            end_trees[t] = (
                cKDTree(pos[idx]),
                idx,
            )

    for t, idx in starts_by_t.items():
        if len(idx):
            start_trees[t] = (
                cKDTree(pos[idx]),
                idx,
            )

    rows = []

    # ========================================================
    # TYPE L
    #
    # A(t)    B(t+1) -> C(t+2)
    #   X---->
    #
    # B is a track start with exactly one successor.
    # Predict A using constant velocity:
    #
    #   A_expected = 2B - C
    # ========================================================

    left_middle_idx = np.flatnonzero((in_deg == 0) & (out_deg == 1) & (t_arr > t_arr.min()))

    for bi in left_middle_idx:
        b_id = node_ids[bi]
        tb = int(t_arr[bi])

        succ = g.successors(b_id)

        if len(succ) != 1:
            continue

        c_id = succ[0]

        if c_id not in id_to_i:
            continue

        ci = id_to_i[c_id]

        # Existing edge must be exactly B(t+1) -> C(t+2).
        if int(t_arr[ci]) != tb + 1:
            continue

        candidate_frame = tb - 1

        if candidate_frame not in end_trees:
            continue

        tree, endpoint_idx = end_trees[candidate_frame]

        expected_a = 2.0 * pos[bi] - pos[ci]

        k = min(
            2,
            len(endpoint_idx),
        )

        distances, locs = tree.query(
            expected_a,
            k=k,
        )

        distances = np.atleast_1d(distances)

        locs = np.atleast_1d(locs)

        ai = endpoint_idx[int(locs[0])]

        a_id = node_ids[ai]

        # By construction B has no predecessor.
        assert not g.has_edge(
            a_id,
            b_id,
        )

        second = float(distances[1]) if len(distances) > 1 else np.inf

        d_ab = float(np.linalg.norm(pos[bi] - pos[ai]))

        d_bc = float(np.linalg.norm(pos[ci] - pos[bi]))

        rows.append(
            {
                "dataset": name,
                "prefix": prefix_map[name],
                "type": "LEFT_MISSING",
                "a_id": a_id,
                "b_id": b_id,
                "c_id": c_id,
                "t_mid": tb,
                "prediction_residual_um": float(distances[0]),
                "second_residual_um": second,
                "ambiguity_margin_um": (second - float(distances[0])),
                "d_ab_um": d_ab,
                "d_bc_um": d_bc,
                "max_step_um": max(
                    d_ab,
                    d_bc,
                ),
                "step_balance": step_balance(
                    d_ab,
                    d_bc,
                ),
                "turn_angle_deg": turn_angle_deg(
                    pos[ai],
                    pos[bi],
                    pos[ci],
                ),
            }
        )

    # ========================================================
    # TYPE R
    #
    # A(t) -> B(t+1)    C(t+2)
    #                    ^
    #                    X
    #
    # B is a track end with exactly one predecessor.
    # Predict C:
    #
    #   C_expected = 2B - A
    # ========================================================

    right_middle_idx = np.flatnonzero((in_deg == 1) & (out_deg == 0) & (t_arr < t_arr.max()))

    for bi in right_middle_idx:
        b_id = node_ids[bi]
        tb = int(t_arr[bi])

        pred = g.predecessors(b_id)

        if len(pred) != 1:
            continue

        a_id = pred[0]

        if a_id not in id_to_i:
            continue

        ai = id_to_i[a_id]

        if int(t_arr[ai]) != tb - 1:
            continue

        candidate_frame = tb + 1

        if candidate_frame not in start_trees:
            continue

        tree, start_idx = start_trees[candidate_frame]

        expected_c = 2.0 * pos[bi] - pos[ai]

        k = min(
            2,
            len(start_idx),
        )

        distances, locs = tree.query(
            expected_c,
            k=k,
        )

        distances = np.atleast_1d(distances)

        locs = np.atleast_1d(locs)

        ci = start_idx[int(locs[0])]

        c_id = node_ids[ci]

        assert not g.has_edge(
            b_id,
            c_id,
        )

        second = float(distances[1]) if len(distances) > 1 else np.inf

        d_ab = float(np.linalg.norm(pos[bi] - pos[ai]))

        d_bc = float(np.linalg.norm(pos[ci] - pos[bi]))

        rows.append(
            {
                "dataset": name,
                "prefix": prefix_map[name],
                "type": "RIGHT_MISSING",
                "a_id": a_id,
                "b_id": b_id,
                "c_id": c_id,
                "t_mid": tb,
                "prediction_residual_um": float(distances[0]),
                "second_residual_um": second,
                "ambiguity_margin_um": (second - float(distances[0])),
                "d_ab_um": d_ab,
                "d_bc_um": d_bc,
                "max_step_um": max(
                    d_ab,
                    d_bc,
                ),
                "step_balance": step_balance(
                    d_ab,
                    d_bc,
                ),
                "turn_angle_deg": turn_angle_deg(
                    pos[ai],
                    pos[bi],
                    pos[ci],
                ),
            }
        )

    return pd.DataFrame(rows)


def census_isolated_bridges(
    name: str,
    pred_995_dir: Path,
    pred_9975_dir: Path,
    train_dir: Path,
    prefix: str,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
):
    # Verbatim body migrated from 09_gap_recovery.ipynb cell 10 (census_isolated_bridges_0910), with prediction/GT
    # roots, prefix and frozen limits bound from arguments.
    WARM2_0995 = Path(pred_995_dir)
    WARM2_09975 = Path(pred_9975_dir)
    TRAIN_DIR = Path(train_dir)
    CONF_COMPONENT_LIMIT = config.confidence_component_limit
    prefix_map = {name: prefix}
    MAX_STEP_UM_0910 = config.recovery_max_step_um

    g = load_graph(WARM2_0995 / f"{name}.geff")

    high_g = load_graph(WARM2_09975 / f"{name}.geff")

    scale = np.asarray(
        open_dataset(
            TRAIN_DIR / name,
            load_image=False,
        ).scale,
        dtype=float,
    )

    assert scale.shape == (3,)

    # --------------------------------------------------------
    # Frozen G2 + K9, BEFORE cleanup / gap repair
    # --------------------------------------------------------

    apply_frozen_g2(
        g,
        scale,
        config,
    )

    high_keys = node_coordinate_keys(high_g)

    apply_low_conf_component_pruning(
        g,
        high_keys,
        CONF_COMPONENT_LIMIT,
    )

    # --------------------------------------------------------
    # Node arrays
    # --------------------------------------------------------

    attrs = g.node_attrs(unpack=True).to_pandas()

    node_ids = attrs["node_id"].to_numpy(dtype=int)

    t_arr = attrs["t"].to_numpy(dtype=int)

    z_arr = attrs["z"].to_numpy(dtype=float)

    y_arr = attrs["y"].to_numpy(dtype=float)

    x_arr = attrs["x"].to_numpy(dtype=float)

    # Physical z,y,x coordinates in microns.
    pos = np.column_stack(
        [
            z_arr * scale[0],
            y_arr * scale[1],
            x_arr * scale[2],
        ]
    )

    in_deg = np.asarray(
        g.in_degree(node_ids.tolist()),
        dtype=int,
    )

    out_deg = np.asarray(
        g.out_degree(node_ids.tolist()),
        dtype=int,
    )

    # --------------------------------------------------------
    # Relevant topology classes
    # --------------------------------------------------------

    isolated_mask = (in_deg == 0) & (out_deg == 0) & (t_arr > t_arr.min()) & (t_arr < t_arr.max())

    end_mask = out_deg == 0

    start_mask = in_deg == 0

    isolated_idx = np.flatnonzero(isolated_mask)

    # --------------------------------------------------------
    # Frame indexes / KD trees
    # --------------------------------------------------------

    ends_by_t = {}
    starts_by_t = {}

    for t in np.unique(t_arr):
        frame = np.flatnonzero(t_arr == t)

        ends = frame[end_mask[frame]]

        starts = frame[start_mask[frame]]

        if len(ends):
            ends_by_t[int(t)] = (
                cKDTree(pos[ends]),
                ends,
            )

        if len(starts):
            starts_by_t[int(t)] = (
                cKDTree(pos[starts]),
                starts,
            )

    # --------------------------------------------------------
    # Best A-B-C geometry for every isolated B
    # --------------------------------------------------------

    rows = []

    isolated_with_pair = 0

    for bi in isolated_idx:
        tb = int(t_arr[bi])

        # Need:
        #   A at tb-1
        #   C at tb+1
        if tb - 1 not in ends_by_t or tb + 1 not in starts_by_t:
            continue

        end_tree, end_idx = ends_by_t[tb - 1]

        start_tree, start_idx = starts_by_t[tb + 1]

        # ----------------------------------------
        # Candidate endpoints within 8 um of B
        # ----------------------------------------

        local_a = end_tree.query_ball_point(
            pos[bi],
            r=MAX_STEP_UM_0910,
        )

        local_c = start_tree.query_ball_point(
            pos[bi],
            r=MAX_STEP_UM_0910,
        )

        if not local_a or not local_c:
            continue

        isolated_with_pair += 1

        # ----------------------------------------
        # Enumerate local A/C combinations.
        #
        # Usually very small because the physical radius is only
        # 8 um. Keep exact enumeration rather than guessing one
        # nearest endpoint independently.
        # ----------------------------------------

        pair_rows = []

        for a_local in local_a:
            ai = end_idx[int(a_local)]

            d_ab = float(np.linalg.norm(pos[bi] - pos[ai]))

            for c_local in local_c:
                ci = start_idx[int(c_local)]

                d_bc = float(np.linalg.norm(pos[ci] - pos[bi]))

                # Exact Stage-09 constant-velocity residual:
                #
                # A_expected = 2B - C
                #
                # residual = ||A - A_expected||
                #          = ||A + C - 2B||
                #
                # The same value is obtained from the opposite
                # direction.
                cv_residual = float(np.linalg.norm(pos[ai] + pos[ci] - 2.0 * pos[bi]))

                angle = turn_angle_deg_0910(
                    pos[ai],
                    pos[bi],
                    pos[ci],
                )

                balance = step_balance_0910(
                    d_ab,
                    d_bc,
                )

                pair_rows.append(
                    (
                        cv_residual,
                        max(
                            d_ab,
                            d_bc,
                        ),
                        angle,
                        balance,
                        ai,
                        ci,
                        d_ab,
                        d_bc,
                    )
                )

        # ----------------------------------------
        # Sort primarily by constant-velocity residual.
        # Tie-break using max step.
        # ----------------------------------------

        pair_rows.sort(
            key=lambda x: (
                x[0],
                x[1],
            )
        )

        best = pair_rows[0]

        second_residual = float(pair_rows[1][0]) if len(pair_rows) > 1 else np.inf

        (
            best_residual,
            best_max_step,
            best_angle,
            best_balance,
            ai,
            ci,
            d_ab,
            d_bc,
        ) = best

        rows.append(
            {
                "dataset": name,
                "prefix": prefix_map[name],
                "a_id": int(node_ids[ai]),
                "b_id": int(node_ids[bi]),
                "c_id": int(node_ids[ci]),
                "t_mid": tb,
                "local_a_count": len(local_a),
                "local_c_count": len(local_c),
                "pair_count": len(pair_rows),
                "cv_residual_um": float(best_residual),
                "second_cv_residual_um": second_residual,
                "ambiguity_margin_um": (second_residual - float(best_residual)),
                "d_ab_um": float(d_ab),
                "d_bc_um": float(d_bc),
                "max_step_um": float(best_max_step),
                "step_balance": float(best_balance),
                "turn_angle_deg": float(best_angle),
                "isolated_nodes_dataset": len(isolated_idx),
            }
        )

    return (
        pd.DataFrame(rows),
        {
            "dataset": name,
            "prefix": prefix_map[name],
            "isolated_nodes": len(isolated_idx),
            "isolated_with_local_pair": isolated_with_pair,
        },
    )


def build_mutual_policy(
    gap_candidates,
    residual_radius_um: float | None = None,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
):
    # Verbatim body migrated from 09_gap_recovery.ipynb cell 7 (build_mutual_policy_0908), with the candidate
    # table and frozen limits bound from arguments.
    if residual_radius_um is None:
        residual_radius_um = config.recovery_residual_max_um

    p = gap_candidates.copy()

    p["source_id"] = np.where(
        p["type"] == "LEFT_MISSING",
        p["a_id"],
        p["b_id"],
    ).astype(int)

    p["target_id"] = np.where(
        p["type"] == "LEFT_MISSING",
        p["b_id"],
        p["c_id"],
    ).astype(int)

    # ----------------------------------------
    # Same conservative geometry as 09.06.
    # Only residual_radius_um changes.
    # ----------------------------------------

    p["geometry_pass"] = (
        (p["prediction_residual_um"] <= residual_radius_um)
        & (p["max_step_um"] <= config.recovery_max_step_um)
        & (p["step_balance"] <= config.recovery_step_balance_max)
        & (p["turn_angle_deg"] <= config.recovery_turn_angle_max_deg)
        & (p["ambiguity_margin_um"] >= config.recovery_ambiguity_margin_min_um)
    )

    p["left_pass"] = p["geometry_pass"] & (p["type"] == "LEFT_MISSING")

    p["right_pass"] = p["geometry_pass"] & (p["type"] == "RIGHT_MISSING")

    edge_level = p.groupby(
        [
            "dataset",
            "prefix",
            "source_id",
            "target_id",
        ],
        as_index=False,
    ).agg(
        left_pass=(
            "left_pass",
            "max",
        ),
        right_pass=(
            "right_pass",
            "max",
        ),
    )

    policy = edge_level[edge_level["left_pass"] & edge_level["right_pass"]][
        [
            "dataset",
            "prefix",
            "source_id",
            "target_id",
        ]
    ].copy()

    return policy


def build_strict_two_edge_policy(
    isolated_candidates,
    residual_um: float | None = None,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
):
    # Verbatim body migrated from 09_gap_recovery.ipynb cell 14 (build_strict_two_edge_policy_0914), with the
    # candidate table and frozen limits bound from arguments.
    if residual_um is None:
        residual_um = config.recovery_residual_max_um

    p = isolated_candidates[
        [
            "dataset",
            "prefix",
            "a_id",
            "b_id",
            "c_id",
            "cv_residual_um",
            "max_step_um",
            "step_balance",
            "turn_angle_deg",
            "ambiguity_margin_um",
            "pair_count",
        ]
    ].copy()

    for col in [
        "a_id",
        "b_id",
        "c_id",
    ]:
        p[col] = p[col].astype(int)

    # --------------------------------------------------------
    # Geometry first.
    #
    # Only residual_um changes across 09.14.
    # --------------------------------------------------------

    p = p[
        (p["cv_residual_um"] <= residual_um)
        & (p["max_step_um"] <= config.recovery_max_step_um)
        & (p["step_balance"] <= config.recovery_step_balance_max)
        & (p["turn_angle_deg"] <= config.recovery_turn_angle_max_deg)
        & (p["ambiguity_margin_um"] >= config.recovery_ambiguity_margin_min_um)
    ].copy()

    # --------------------------------------------------------
    # Recompute endpoint conflicts INSIDE this radius.
    #
    # This preserves the exact 09.13 rule at r=4 while allowing
    # conflicts to disappear naturally for stricter radii.
    # --------------------------------------------------------

    a_usage = p.groupby(
        [
            "dataset",
            "a_id",
        ]
    )["b_id"].transform("nunique")

    c_usage = p.groupby(
        [
            "dataset",
            "c_id",
        ]
    )["b_id"].transform("nunique")

    p["conflict_free"] = (a_usage == 1) & (c_usage == 1)

    # --------------------------------------------------------
    # Strict unique-pair policy
    # --------------------------------------------------------

    p = p[p["conflict_free"] & (p["pair_count"] == config.two_edge_required_pair_count)].copy()

    return p


def add_available_repairs(
    graph,
    candidate_edges,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
) -> dict:
    # Verbatim body migrated from 09_gap_recovery.ipynb cell 6 (add_available_repairs_0907).
    """
    Add a repair only when both endpoints still exist and
    source/target remain a valid track-end -> track-start pair.
    """

    node_table = graph.node_attrs(unpack=True).to_pandas().set_index("node_id")

    stats = {
        "candidate_edges": len(candidate_edges),
        "added": 0,
        "missing_endpoint": 0,
        "degree_conflict": 0,
        "already_present": 0,
    }

    for row in candidate_edges.itertuples(index=False):
        source_id = int(row.source_id)

        target_id = int(row.target_id)

        if not graph.has_node(source_id) or not graph.has_node(target_id):
            stats["missing_endpoint"] += 1
            continue

        if graph.has_edge(
            source_id,
            target_id,
        ):
            stats["already_present"] += 1
            continue

        if graph.out_degree(source_id) != 0 or graph.in_degree(target_id) != 0:
            stats["degree_conflict"] += 1
            continue

        t_source = int(
            node_table.loc[
                source_id,
                "t",
            ]
        )

        t_target = int(
            node_table.loc[
                target_id,
                "t",
            ]
        )

        assert t_target == t_source + 1

        edge_dist = recovered_edge_dist(
            node_table,
            source_id,
            target_id,
            config,
        )

        graph.add_edge(
            source_id,
            target_id,
            {
                "edge_dist": edge_dist,
                "edge_prob": 0.0,
            },
        )

        stats["added"] += 1

    return stats


def add_two_edge_bridges(
    graph,
    candidate_bridges,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
) -> dict:
    # Verbatim body migrated from 09_gap_recovery.ipynb cell 12 (add_two_edge_bridges_0912).
    """
    Add A->B and B->C only when the bridge remains structurally
    valid after frozen single-edge repair.

    Required current topology:
        out_degree(A) = 0
        in_degree(B)  = 0
        out_degree(B) = 0
        in_degree(C)  = 0

    No forced rewiring.
    """

    node_table = graph.node_attrs(unpack=True).to_pandas().set_index("node_id")

    stats = {
        "requested_bridges": len(candidate_bridges),
        "added_bridges": 0,
        "edges_added": 0,
        "missing_endpoint": 0,
        "degree_conflict": 0,
        "existing_edge_conflict": 0,
    }

    for row in candidate_bridges.itertuples(index=False):
        a_id = int(row.a_id)
        b_id = int(row.b_id)
        c_id = int(row.c_id)

        # ----------------------------------------
        # All three existing nodes must survive
        # G2 + K9 and single-edge repair stage.
        # ----------------------------------------

        if not (graph.has_node(a_id) and graph.has_node(b_id) and graph.has_node(c_id)):
            stats["missing_endpoint"] += 1
            continue

        # ----------------------------------------
        # Do not duplicate edges.
        # ----------------------------------------

        if graph.has_edge(
            a_id,
            b_id,
        ) or graph.has_edge(
            b_id,
            c_id,
        ):
            stats["existing_edge_conflict"] += 1
            continue

        # ----------------------------------------
        # Strict topology gate after interaction
        # with frozen single-edge repair.
        # ----------------------------------------

        if not (
            graph.out_degree(a_id) == 0
            and graph.in_degree(b_id) == 0
            and graph.out_degree(b_id) == 0
            and graph.in_degree(c_id) == 0
        ):
            stats["degree_conflict"] += 1
            continue

        # ----------------------------------------
        # Temporal consistency
        # ----------------------------------------

        ta = int(
            node_table.loc[
                a_id,
                "t",
            ]
        )

        tb = int(
            node_table.loc[
                b_id,
                "t",
            ]
        )

        tc = int(
            node_table.loc[
                c_id,
                "t",
            ]
        )

        assert tb == ta + 1 and tc == tb + 1

        # ----------------------------------------
        # Verified model-grid edge_dist
        # ----------------------------------------

        dist_ab = recovered_edge_dist(
            node_table,
            a_id,
            b_id,
            config,
        )

        dist_bc = recovered_edge_dist(
            node_table,
            b_id,
            c_id,
            config,
        )

        graph.add_edge(
            a_id,
            b_id,
            {
                "edge_dist": dist_ab,
                "edge_prob": 0.0,
            },
        )

        graph.add_edge(
            b_id,
            c_id,
            {
                "edge_dist": dist_bc,
                "edge_prob": 0.0,
            },
        )

        stats["added_bridges"] += 1

        stats["edges_added"] += 2

    return stats


# ============================================================
# Public API
#
# Everything below is orchestration over the migrated frozen
# functions. It adds no new scientific policy.
# ============================================================


@dataclass(frozen=True)
class FrozenV9Policies:
    """Candidate censuses and frozen policies for one prediction set.

    Always regenerated from the prediction set's own node IDs.
    """

    gap_candidates: pd.DataFrame
    single_policy: pd.DataFrame
    isolated_candidates: pd.DataFrame
    isolated_scope: pd.DataFrame
    strict_two_policy: pd.DataFrame
    config: FrozenV9Config = FROZEN_V9_CONFIG

    @property
    def single_opportunities(self) -> int:
        return len(self.gap_candidates)

    @property
    def single_policy_count(self) -> int:
        return len(self.single_policy)

    @property
    def isolated_nodes(self) -> int:
        return int(self.isolated_scope["isolated_nodes"].sum())

    @property
    def isolated_with_local_pair(self) -> int:
        return int(self.isolated_scope["isolated_with_local_pair"].sum())

    @property
    def strict_two_policy_count(self) -> int:
        return len(self.strict_two_policy)

    def counts(self) -> dict[str, int]:
        """Candidate/policy counts, in the order notebooks report them."""
        return {
            "single_opportunities": self.single_opportunities,
            "single_policy": self.single_policy_count,
            "isolated_nodes": self.isolated_nodes,
            "isolated_with_local_pair": self.isolated_with_local_pair,
            "strict_two_policy": self.strict_two_policy_count,
        }


@dataclass
class FrozenV9Application:
    """Per-dataset intervention statistics from one frozen V9 application."""

    dataset: str
    single_added: int
    two_requested: int
    two_added: int
    two_edges_added: int
    two_missing_endpoint: int
    two_degree_conflict: int
    two_existing_edge_conflict: int
    cleanup_removed: int
    cleanup_singleton_components: int
    cleanup_long_two_node_components: int

    @property
    def associations_added(self) -> int:
        """Associations added by the strict two-edge stage (2 per bridge).

        Matches the locked Stage10 "two associations added" statistic. For the
        combined count across both recovery stages use :attr:`total_edges_added`.
        """
        return int(self.two_edges_added)

    @property
    def total_edges_added(self) -> int:
        """Edges added by both recovery stages combined."""
        return int(self.single_added + self.two_edges_added)

    def to_row(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "single_added": self.single_added,
            "two_requested": self.two_requested,
            "two_added": self.two_added,
            "two_edges_added": self.two_edges_added,
            "associations_added": self.associations_added,
            "total_edges_added": self.total_edges_added,
            "two_missing_endpoint": self.two_missing_endpoint,
            "two_degree_conflict": self.two_degree_conflict,
            "two_existing_edge_conflict": self.two_existing_edge_conflict,
            "cleanup_removed": self.cleanup_removed,
        }


@dataclass
class FrozenV9Evaluation:
    """Structured result of a frozen V9 evaluation run."""

    summary: dict[str, Any]
    records: list[dict[str, Any]]
    dataset_df: pd.DataFrame
    interventions: dict[str, int]
    policies: FrozenV9Policies
    applications: list[FrozenV9Application] = field(default_factory=list)


def _prefix_for(name: str, prefix_map: dict[str, str] | None) -> str:
    """Dataset prefix, from an explicit map when given, else the name stem."""
    if prefix_map is not None and name in prefix_map:
        return str(prefix_map[name])
    return str(name).split("_")[0]


def build_frozen_v9_policies(
    dataset_names: list[str],
    pred_995_dir: Path,
    pred_9975_dir: Path,
    train_dir: Path,
    *,
    prefix_map: dict[str, str] | None = None,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
    progress: bool = False,
) -> FrozenV9Policies:
    """Regenerate candidate censuses and frozen policies for one prediction set.

    Candidates are derived from THIS prediction set's node IDs: a census is run
    per dataset against the supplied det=.995 / det=.9975 directories. Stage09
    candidate tables are never reused.
    """
    pred_995_dir = Path(pred_995_dir)
    pred_9975_dir = Path(pred_9975_dir)
    train_dir = Path(train_dir)

    single_parts: list[pd.DataFrame] = []
    isolated_parts: list[pd.DataFrame] = []
    isolated_scope_rows: list[dict[str, Any]] = []

    total = len(dataset_names)

    for i, name in enumerate(dataset_names, start=1):
        prefix = _prefix_for(name, prefix_map)

        single_df = census_single_edge_gaps(
            name,
            pred_995_dir,
            pred_9975_dir,
            train_dir,
            prefix,
            config,
        )

        if len(single_df):
            single_parts.append(single_df)

        isolated_df, scope = census_isolated_bridges(
            name,
            pred_995_dir,
            pred_9975_dir,
            train_dir,
            prefix,
            config,
        )

        if len(isolated_df):
            isolated_parts.append(isolated_df)

        isolated_scope_rows.append(scope)

        if progress and (i == 1 or i % 10 == 0 or i == total):
            print(f"  candidates {i:02d}/{total}", flush=True)

    if not single_parts:
        raise ValueError("No single-edge candidates were generated.")

    if not isolated_parts:
        raise ValueError("No isolated-middle candidates were generated.")

    gap_candidates = pd.concat(single_parts, ignore_index=True)
    isolated_candidates = pd.concat(isolated_parts, ignore_index=True)
    isolated_scope = pd.DataFrame(isolated_scope_rows)

    single_policy = build_mutual_policy(
        gap_candidates,
        config.recovery_residual_max_um,
        config,
    )

    strict_two_policy = build_strict_two_edge_policy(
        isolated_candidates,
        config.recovery_residual_max_um,
        config,
    )

    return FrozenV9Policies(
        gap_candidates=gap_candidates,
        single_policy=single_policy,
        isolated_candidates=isolated_candidates,
        isolated_scope=isolated_scope,
        strict_two_policy=strict_two_policy,
        config=config,
    )


def apply_frozen_v9(
    dataset_name: str,
    pred_995_dir: Path,
    pred_9975_dir: Path,
    policies: FrozenV9Policies,
    train_dir: Path,
    *,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
    strict_single_policy: bool = True,
):
    """Build a fresh graph and apply the frozen V9 order to it.

    Order is fixed: G2 -> K9 -> single-edge recovery -> strict two-edge
    recovery -> cleanup. Returns ``(graph, scale, FrozenV9Application)``.

    The graph is built in memory from the prediction GEFFs; the GEFFs on disk
    are never modified.
    """
    graph, scale = prepare_precleanup_graph(
        dataset_name,
        pred_995_dir,
        pred_9975_dir,
        train_dir,
        config,
    )

    single_sub = policies.single_policy[policies.single_policy["dataset"] == dataset_name].copy()

    single_stats = add_available_repairs(graph, single_sub, config)

    if strict_single_policy and int(single_stats["added"]) != len(single_sub):
        raise AssertionError(f"{dataset_name}: single-edge policy unexpectedly not fully addable.\n{single_stats}")

    two_sub = policies.strict_two_policy[policies.strict_two_policy["dataset"] == dataset_name].copy()

    two_stats = add_two_edge_bridges(graph, two_sub, config)

    cleanup_stats = apply_frozen_component_cleanup(graph, scale, config)

    stats = FrozenV9Application(
        dataset=dataset_name,
        single_added=int(single_stats["added"]),
        two_requested=int(two_stats["requested_bridges"]),
        two_added=int(two_stats["added_bridges"]),
        two_edges_added=int(two_stats["edges_added"]),
        two_missing_endpoint=int(two_stats["missing_endpoint"]),
        two_degree_conflict=int(two_stats["degree_conflict"]),
        two_existing_edge_conflict=int(two_stats["existing_edge_conflict"]),
        cleanup_removed=int(cleanup_stats["nodes_removed"]),
        cleanup_singleton_components=int(cleanup_stats["singleton_components"]),
        cleanup_long_two_node_components=int(cleanup_stats["long_two_node_components"]),
    )

    return graph, scale, stats


def evaluate_graph(
    graph,
    gt_graph,
    scale,
    estimated_nodes: float,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
) -> dict[str, Any]:
    """Locked official metric wrapper.

    Exact Stage10 flow (10.15B / 10.16): ``official_evaluate`` -> ``node_recall``
    -> ``per_sample_metrics``. The competition metric itself is not
    reimplemented; it comes from the source-locked ``external/official`` package.
    """
    er = official_evaluate(
        graph,
        gt_graph,
        scale=tuple(scale),
        max_distance=config.metric_max_distance,
    )

    recall = node_recall(graph, gt_graph)

    return per_sample_metrics(er, float(estimated_nodes), recall)


def evaluate_frozen_v9(
    dataset_names: list[str],
    pred_995_dir: Path,
    pred_9975_dir: Path,
    gt_dir: Path,
    estimated_nodes: dict[str, float],
    *,
    train_dir: Path | None = None,
    prefix_map: dict[str, str] | None = None,
    policies: FrozenV9Policies | None = None,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
    progress: bool = False,
) -> FrozenV9Evaluation:
    """Regenerate policies, process every graph, and run the locked metric.

    Read-only with respect to every artifact on disk: prediction GEFFs are read,
    never written, and no prediction GEFF is produced.

    ``train_dir`` supplies the physical voxel scale and defaults to ``gt_dir``,
    which is where the competition keeps both GT graphs and dataset metadata.
    """
    pred_995_dir = Path(pred_995_dir)
    pred_9975_dir = Path(pred_9975_dir)
    gt_dir = Path(gt_dir)
    train_dir = gt_dir if train_dir is None else Path(train_dir)

    missing = [name for name in dataset_names if name not in estimated_nodes]
    if missing:
        raise KeyError(f"estimated_nodes is missing {len(missing)} dataset(s): {missing[:5]}")

    if policies is None:
        policies = build_frozen_v9_policies(
            dataset_names,
            pred_995_dir,
            pred_9975_dir,
            train_dir,
            prefix_map=prefix_map,
            config=config,
            progress=progress,
        )

    records: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    applications: list[FrozenV9Application] = []

    total = len(dataset_names)

    for i, name in enumerate(dataset_names, start=1):
        graph, scale, stats = apply_frozen_v9(
            name,
            pred_995_dir,
            pred_9975_dir,
            policies,
            train_dir,
            config=config,
        )

        gt_graph = load_graph(gt_dir / f"{name}.geff")

        metrics = evaluate_graph(
            graph,
            gt_graph,
            scale,
            estimated_nodes[name],
            config,
        )

        records.append(metrics)
        applications.append(stats)

        dataset_rows.append(
            {
                "dataset": name,
                "prefix": _prefix_for(name, prefix_map),
                "edge_jaccard": float(metrics["edge_jaccard"]),
                "adj_edge_jaccard": float(metrics["adj_edge_jaccard"]),
                "node_recall": float(metrics["node_recall"]),
                "total_node_ratio": float(metrics["total_node_ratio"]),
                "single_added": stats.single_added,
                "two_requested": stats.two_requested,
                "two_added": stats.two_added,
                "cleanup_removed": stats.cleanup_removed,
            }
        )

        if progress and (i == 1 or i % 10 == 0 or i == total):
            print(f"  evaluated {i:02d}/{total}", flush=True)

    interventions = {
        "single_added": sum(a.single_added for a in applications),
        "two_requested": sum(a.two_requested for a in applications),
        "two_added": sum(a.two_added for a in applications),
        "two_edges_added": sum(a.two_edges_added for a in applications),
        "associations_added": sum(a.associations_added for a in applications),
        "total_edges_added": sum(a.total_edges_added for a in applications),
        "two_missing_endpoint": sum(a.two_missing_endpoint for a in applications),
        "two_degree_conflict": sum(a.two_degree_conflict for a in applications),
        "two_existing_edge_conflict": sum(a.two_existing_edge_conflict for a in applications),
        "cleanup_removed": sum(a.cleanup_removed for a in applications),
    }

    return FrozenV9Evaluation(
        summary=summarise(records),
        records=records,
        dataset_df=pd.DataFrame(dataset_rows),
        interventions=interventions,
        policies=policies,
        applications=applications,
    )


def summarize_by_prefix(
    evaluation: FrozenV9Evaluation,
    *,
    prefix_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Summarise already-computed per-dataset metrics grouped by prefix.

    Reporting only. This introduces no prefix-specific graph policy: it never
    touches a graph, and the records it groups were produced by one global
    frozen policy. ``summarise`` is applied per prefix group so each row is
    computed exactly the way the overall summary is.
    """
    records = evaluation.records
    names = list(evaluation.dataset_df["dataset"])

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for name, record in zip(names, records, strict=True):
        groups[_prefix_for(name, prefix_map)].append(record)

    rows = []

    for prefix in sorted(groups):
        group_summary = summarise(groups[prefix])
        rows.append({"prefix": prefix, "n": len(groups[prefix]), **group_summary})

    return pd.DataFrame(rows)
