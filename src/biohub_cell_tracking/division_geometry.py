"""Stage 11.03 — ground-truth division geometry census and threshold sensitivity.

Diagnostic only. This module measures the geometry of *annotated* divisions
across the whole labelled training set and sweeps counterfactual geometry gates
over it. It promotes nothing: :mod:`biohub_cell_tracking.frozen_v9` remains the
only definition of the frozen policy, and every gate produced here is a
candidate for later validation, never a new policy.

Geometry is not redefined. :func:`biohub_cell_tracking.division_audit.fork_geometry`
supplies the measurement — the same arithmetic ``frozen_v9.apply_frozen_g2``
uses, including the physical voxel scaling the anisotropic z axis requires — and
:class:`GeometryGate` only carries thresholds to compare it against.

Two quantities are kept strictly apart throughout:

``GT geometry recall``
    fraction of annotated divisions whose *ground-truth* coordinates satisfy a
    gate. Model independent: it says whether the gate is compatible with real
    biology at all.

``observed / counterfactual prediction counts``
    what a gate would admit from a saved prediction set. Model dependent, and
    only meaningful for the fold whose predictions were measured.

Nothing here trains, runs inference, or mutates a graph, GEFF, or config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tracking_cellmot.io import open_dataset

from .division_audit import fork_geometry
from .frozen_v9 import FROZEN_V9_CONFIG, FrozenV9Config, load_graph

__all__ = [
    "GEOMETRY_VARIABLES",
    "QUANTILES",
    "GeometryGate",
    "census_gt_divisions",
    "enumerate_geometry_grid",
    "evaluate_geometry_gate",
    "geometry_pass_rates",
    "grid_gt_recall",
    "quantile_summary",
]


#: Geometry columns the census reports and the sweep summarises.
GEOMETRY_VARIABLES: tuple[str, ...] = (
    "daughter_separation_um",
    "division_angle_deg",
    "parent_daughter_1_um",
    "parent_daughter_2_um",
    "max_parent_daughter_um",
    "distance_balance",
)

#: Quantiles reported for every geometry variable.
QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


# ============================================================
# Geometry gate
# ============================================================


@dataclass(frozen=True)
class GeometryGate:
    """One candidate division-acceptance gate.

    Field names deliberately mirror :class:`frozen_v9.FrozenV9Config` so a gate
    can be read straight off the frozen policy and compared against relaxations
    of it. ``p2_min`` is carried alongside the geometry because the two are only
    interpretable together, but the primary sweep holds it fixed.
    """

    daughter_sep_min_um: float
    angle_min_deg: float
    max_parent_daughter_um: float
    distance_balance_max: float
    p2_min: float = FROZEN_V9_CONFIG.division_p2_min

    @classmethod
    def frozen(cls, config: FrozenV9Config = FROZEN_V9_CONFIG) -> GeometryGate:
        """The gate Frozen V9 currently enforces."""
        return cls(
            daughter_sep_min_um=config.division_daughter_sep_min_um,
            angle_min_deg=config.division_angle_min_deg,
            max_parent_daughter_um=config.division_max_parent_daughter_um,
            distance_balance_max=config.division_distance_balance_max,
            p2_min=config.division_p2_min,
        )

    def to_config(self, config: FrozenV9Config = FROZEN_V9_CONFIG) -> FrozenV9Config:
        """A :class:`FrozenV9Config` identical to *config* but for this gate.

        Lets a counterfactual run the real pipeline through the frozen code
        path without editing ``frozen_v9.py``.
        """
        return FrozenV9Config(
            **{
                **{
                    field: getattr(config, field)
                    for field in config.__dataclass_fields__
                    if field
                    not in {
                        "division_daughter_sep_min_um",
                        "division_angle_min_deg",
                        "division_max_parent_daughter_um",
                        "division_distance_balance_max",
                        "division_p2_min",
                    }
                },
                "division_daughter_sep_min_um": self.daughter_sep_min_um,
                "division_angle_min_deg": self.angle_min_deg,
                "division_max_parent_daughter_um": self.max_parent_daughter_um,
                "division_distance_balance_max": self.distance_balance_max,
                "division_p2_min": self.p2_min,
            }
        )

    @property
    def label(self) -> str:
        """Compact, stable identifier used as a grid key."""
        return (
            f"sep>={self.daughter_sep_min_um:g}"
            f"|ang>={self.angle_min_deg:g}"
            f"|maxd<={self.max_parent_daughter_um:g}"
            f"|bal<={self.distance_balance_max:g}"
            f"|p2>={self.p2_min:g}"
        )

    @property
    def is_frozen_geometry(self) -> bool:
        return self.geometry_key == GeometryGate.frozen().geometry_key

    @property
    def geometry_key(self) -> tuple[float, float, float, float]:
        return (
            float(self.daughter_sep_min_um),
            float(self.angle_min_deg),
            float(self.max_parent_daughter_um),
            float(self.distance_balance_max),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.label,
            "sep_min_um": self.daughter_sep_min_um,
            "angle_min_deg": self.angle_min_deg,
            "max_parent_daughter_um": self.max_parent_daughter_um,
            "balance_max": self.distance_balance_max,
            "p2_min": self.p2_min,
        }


# ============================================================
# Ground-truth census
# ============================================================


def census_gt_divisions(
    dataset_names: list[str],
    gt_dir: Path,
    *,
    fold_map: dict[str, int] | None = None,
    prefix_map: dict[str, str] | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Measure the geometry of every annotated division in *dataset_names*.

    Ground-truth graphs are read and never modified. A GT node with three or
    more successors is recorded with ``gt_daughter_count > 2`` and measured on
    its two lowest-id daughters, matching how the frozen pipeline would see the
    first two branches; such rows are flagged so they can be excluded.
    """
    gt_dir = Path(gt_dir)
    rows: list[dict[str, Any]] = []
    total = len(dataset_names)

    for i, name in enumerate(dataset_names, start=1):
        graph = load_graph(gt_dir / f"{name}.geff")
        dividing = sorted(int(n) for n in graph.dividing_nodes())

        if dividing:
            scale = tuple(float(s) for s in open_dataset(gt_dir / name, load_image=False).scale)
            nodes = graph.node_attrs(unpack=True).to_pandas().set_index("node_id")

        for parent in dividing:
            daughters = sorted(int(c) for c in graph.successors(parent))
            d1, d2 = daughters[0], daughters[1]

            p_row = nodes.loc[parent]
            d1_row = nodes.loc[d1]
            d2_row = nodes.loc[d2]

            geometry = fork_geometry(
                (p_row["z"], p_row["y"], p_row["x"]),
                (d1_row["z"], d1_row["y"], d1_row["x"]),
                (d2_row["z"], d2_row["y"], d2_row["x"]),
                scale,
            )

            rows.append(
                {
                    "dataset": name,
                    "fold": None if fold_map is None else int(fold_map[name]),
                    "prefix": (str(name).split("_")[0] if prefix_map is None else str(prefix_map[name])),
                    "parent": parent,
                    "daughter1": d1,
                    "daughter2": d2,
                    "t": int(p_row["t"]),
                    "daughter1_t": int(d1_row["t"]),
                    "daughter2_t": int(d2_row["t"]),
                    "gt_daughter_count": len(daughters),
                    "daughter_separation_um": geometry.daughter_separation_um,
                    "division_angle_deg": geometry.division_angle_deg,
                    "parent_daughter_1_um": geometry.d1_um,
                    "parent_daughter_2_um": geometry.d2_um,
                    "max_parent_daughter_um": geometry.max_parent_daughter_um,
                    "distance_balance": geometry.distance_balance,
                    "scale_z": scale[0],
                    "scale_y": scale[1],
                    "scale_x": scale[2],
                    # Degenerate geometry the frozen gate cannot score meaningfully.
                    "degenerate_zero_step": bool(geometry.d1_um == 0.0 or geometry.d2_um == 0.0),
                    "degenerate_angle_nan": bool(not np.isfinite(geometry.division_angle_deg)),
                    "extra_daughters": len(daughters) > 2,
                    "daughter_frame_gap_ok": bool(
                        int(d1_row["t"]) == int(p_row["t"]) + 1 and int(d2_row["t"]) == int(p_row["t"]) + 1
                    ),
                }
            )

        if progress and (i == 1 or i % 50 == 0 or i == total):
            print(f"  census {i:3d}/{total}", flush=True)

    return pd.DataFrame(rows)


# ============================================================
# Gate evaluation
# ============================================================


def evaluate_geometry_gate(frame: pd.DataFrame, gate: GeometryGate) -> pd.DataFrame:
    """Per-row pass flags for *gate*, as a frame aligned to *frame*'s index.

    Mirrors the frozen ``apply_frozen_g2`` comparisons exactly, including
    treating a non-finite angle as a failure.
    """
    angle = pd.to_numeric(frame["division_angle_deg"], errors="coerce")

    passes_separation = pd.to_numeric(frame["daughter_separation_um"], errors="coerce") >= gate.daughter_sep_min_um
    passes_angle = np.isfinite(angle) & (angle >= gate.angle_min_deg)
    passes_distance = pd.to_numeric(frame["max_parent_daughter_um"], errors="coerce") <= gate.max_parent_daughter_um
    passes_balance = pd.to_numeric(frame["distance_balance"], errors="coerce") <= gate.distance_balance_max

    result = pd.DataFrame(
        {
            "passes_separation": passes_separation.fillna(False).astype(bool),
            "passes_angle": pd.Series(passes_angle, index=frame.index).fillna(False).astype(bool),
            "passes_distance": passes_distance.fillna(False).astype(bool),
            "passes_balance": passes_balance.fillna(False).astype(bool),
        },
        index=frame.index,
    )

    result["passes_geometry"] = (
        result["passes_separation"] & result["passes_angle"] & result["passes_distance"] & result["passes_balance"]
    )

    if "p2" in frame.columns:
        p2 = pd.to_numeric(frame["p2"], errors="coerce")
        result["passes_p2"] = (p2 >= gate.p2_min).fillna(False).astype(bool)
        result["passes_all"] = result["passes_geometry"] & result["passes_p2"]
    else:
        result["passes_all"] = result["passes_geometry"]

    return result


def geometry_pass_rates(
    frame: pd.DataFrame,
    gate: GeometryGate,
    *,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Per-criterion and full-gate pass rates, optionally grouped."""
    flags = evaluate_geometry_gate(frame, gate)
    joined = pd.concat([frame[group_cols] if group_cols else frame[[]], flags], axis=1)

    flag_cols = list(flags.columns)

    if not group_cols:
        row = {"n": len(joined)}
        row.update({col: float(joined[col].mean()) if len(joined) else float("nan") for col in flag_cols})
        row.update({f"{col}_n": int(joined[col].sum()) for col in flag_cols})
        return pd.DataFrame([row])

    grouped = joined.groupby(group_cols, dropna=False)
    out = grouped[flag_cols].mean()
    out.insert(0, "n", grouped.size())
    for col in flag_cols:
        out[f"{col}_n"] = grouped[col].sum().astype(int)

    return out.reset_index()


# ============================================================
# Distribution summary
# ============================================================


def quantile_summary(
    frame: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
    variables: tuple[str, ...] = GEOMETRY_VARIABLES,
) -> pd.DataFrame:
    """n / min / q05 / q10 / q25 / median / q75 / q90 / q95 / max per variable."""

    def describe(block: pd.DataFrame, key: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for variable in variables:
            values = pd.to_numeric(block[variable], errors="coerce").dropna()
            row: dict[str, Any] = {**key, "variable": variable, "n": len(values)}
            if len(values):
                row["min"] = float(values.min())
                for q in QUANTILES:
                    row[f"q{round(q * 100):02d}"] = float(values.quantile(q))
                row["max"] = float(values.max())
            else:
                row["min"] = float("nan")
                for q in QUANTILES:
                    row[f"q{round(q * 100):02d}"] = float("nan")
                row["max"] = float("nan")
            rows.append(row)
        return rows

    if not group_cols:
        return pd.DataFrame(describe(frame, {"group": "ALL"}))

    rows: list[dict[str, Any]] = []
    for key, block in frame.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        rows.extend(describe(block, dict(zip(group_cols, key_tuple, strict=True))))

    return pd.DataFrame(rows)


# ============================================================
# Threshold grid
# ============================================================


def enumerate_geometry_grid(
    separations: list[float],
    angles: list[float],
    distances: list[float],
    balances: list[float],
    p2_values: list[float] | None = None,
) -> tuple[GeometryGate, ...]:
    """Deterministic product of the predeclared threshold levels.

    Ordering is fixed (separation, angle, distance, balance, p2) so grid row
    order is reproducible across runs and machines.
    """
    if p2_values is None:
        p2_values = [FROZEN_V9_CONFIG.division_p2_min]

    return tuple(
        GeometryGate(
            daughter_sep_min_um=float(sep),
            angle_min_deg=float(angle),
            max_parent_daughter_um=float(distance),
            distance_balance_max=float(balance),
            p2_min=float(p2),
        )
        for sep in separations
        for angle in angles
        for distance in distances
        for balance in balances
        for p2 in p2_values
    )


def grid_gt_recall(
    census: pd.DataFrame,
    gates: tuple[GeometryGate, ...],
    *,
    fold_col: str = "fold",
    prefix_col: str = "prefix",
) -> pd.DataFrame:
    """GT geometry recall of every gate, overall and per fold and prefix.

    ``p2`` never enters: ground truth carries no association probability, so a
    gate's GT recall is a pure geometry statement.
    """
    folds = sorted(census[fold_col].dropna().unique()) if fold_col in census else []
    prefixes = sorted(census[prefix_col].dropna().unique()) if prefix_col in census else []

    rows = []

    for gate in gates:
        flags = evaluate_geometry_gate(census, gate)["passes_geometry"]

        row: dict[str, Any] = {
            **gate.as_dict(),
            "is_frozen_geometry": gate.is_frozen_geometry,
            "n_gt": len(census),
            "gt_pass": int(flags.sum()),
            "gt_recall": float(flags.mean()) if len(census) else float("nan"),
        }

        fold_recalls = []
        for fold in folds:
            mask = census[fold_col] == fold
            recall = float(flags[mask].mean()) if int(mask.sum()) else float("nan")
            row[f"gt_recall_fold{int(fold)}"] = recall
            fold_recalls.append(recall)

        if fold_recalls:
            row["gt_recall_fold_min"] = float(np.nanmin(fold_recalls))
            row["gt_recall_fold_max"] = float(np.nanmax(fold_recalls))
            row["gt_recall_fold_spread"] = row["gt_recall_fold_max"] - row["gt_recall_fold_min"]

        for prefix in prefixes:
            mask = census[prefix_col] == prefix
            row[f"gt_recall_{prefix}"] = float(flags[mask].mean()) if int(mask.sum()) else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)
