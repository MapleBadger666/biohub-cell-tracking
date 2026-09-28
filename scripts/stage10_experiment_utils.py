"""Reusable experiment plumbing for Stage10 notebooks.

This module exists only to remove repetitive subprocess/manifest/verification
boilerplate from notebook cells. It is deliberately *infrastructure only*.

Explicitly OUT OF SCOPE for this module (kept independently auditable):

- domain-aware detection loss implementation
- model architecture
- training logic
- inference logic
- frozen V9 graph policies
- competition metric logic
- dataset scientific logic

Conventions assumed by the helpers:

- ``project`` is the absolute Biohub project root. Paths are treated
  lexically (never resolved), so callers should pass an absolute root.
- The project interpreter is ``<project>/.venv/bin/python``. ``CONDA_PREFIX``
  is never consulted.
- Prediction outputs are ``*.geff`` entries (directories in this project).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

__all__ = [
    "CommandResult",
    "ExperimentRun",
    "SplitPayload",
    "build_project_env",
    "ensure_no_existing_outputs",
    "load_split_payload",
    "project_python",
    "read_log_tail",
    "run_logged",
    "sha256_file",
    "verify_prediction_set",
    "write_subset_split",
]


# Streaming chunk for checkpoint-sized files; never read a checkpoint whole.
_HASH_CHUNK_BYTES = 1024 * 1024

# Keys under which this project has stored a fold list inside a dict payload.
_FOLD_KEYS = ("folds", "splits")

_GEFF_SUFFIX = ".geff"

_LOG_TAIL_LINES = 100
_LOG_TAIL_MAX_BYTES = 256 * 1024


# ============================================================
# Hashing
# ============================================================


def sha256_file(path: Path, *, chunk_size: int = _HASH_CHUNK_BYTES) -> str:
    """Return the SHA256 hex digest of ``path``, streamed in chunks.

    Large checkpoints are never loaded into memory whole.
    """
    path = Path(path)
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# Subprocess environment / interpreter
# ============================================================


def build_project_env(
    project: Path,
    *,
    include_existing_pythonpath: bool = True,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a subprocess environment with a project-first ``PYTHONPATH``.

    ``PYTHONPATH`` begins with, in order:

    1. ``project / "scripts"``
    2. ``project / "external/official/scripts"``

    Any pre-existing ``PYTHONPATH`` is preserved *after* those entries when
    ``include_existing_pythonpath`` is true.

    ``CONDA_PREFIX`` is never used to resolve Python; see :func:`project_python`.
    """
    project = Path(project)

    env = dict(os.environ if base_env is None else base_env)

    parts = [
        str(project / "scripts"),
        str(project / "external/official/scripts"),
    ]

    if include_existing_pythonpath:
        existing = env.get("PYTHONPATH", "")
        if existing:
            parts.append(existing)

    env["PYTHONPATH"] = os.pathsep.join(parts)

    return env


def project_python(project: Path) -> Path:
    """Return ``project / ".venv/bin/python"``, failing closed if absent.

    ``CONDA_PREFIX`` is deliberately ignored: experiments must run under the
    project virtualenv interpreter or not at all.
    """
    candidate = Path(project) / ".venv/bin/python"

    if not candidate.exists():
        raise FileNotFoundError(f"Project interpreter missing: {candidate}")

    return candidate


# ============================================================
# Dataset split manifests
# ============================================================


class SplitPayload(NamedTuple):
    """Parsed ``dataset_splits.json``: the original payload and its fold list."""

    payload: Any
    folds: list[dict[str, Any]]


def _extract_folds(payload: Any, *, source: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Return ``(folds, dict_key)``; ``dict_key`` is ``None`` for list payloads."""
    if isinstance(payload, list):
        return payload, None

    if isinstance(payload, dict):
        for key in _FOLD_KEYS:
            if key in payload:
                folds = payload[key]
                if not isinstance(folds, list):
                    raise ValueError(f"Split payload field {key!r} is not a list in {source}")
                return folds, key

        raise ValueError(f"Split payload dict has none of {_FOLD_KEYS} in {source}")

    raise ValueError(f"Unsupported dataset_splits.json structure {type(payload).__name__} in {source}")


def load_split_payload(path: Path) -> SplitPayload:
    """Load ``dataset_splits.json`` without mutating the file on disk.

    Supports both structures encountered by this project: a dict wrapping the
    folds (``{"folds": [...]}``, also ``{"splits": [...]}``) and a bare list of
    fold dicts. Returns the original payload alongside the fold list.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds, _ = _extract_folds(payload, source=path)
    return SplitPayload(payload=payload, folds=folds)


def write_subset_split(
    source_split: Path,
    output_split: Path,
    *,
    fold: int,
    train_names: list[str] | None = None,
    test_names: list[str] | None = None,
) -> Path:
    """Write a copy of ``source_split`` with one fold's train/test overridden.

    Only the requested fold's ``train``/``test`` lists are replaced, and only
    when the corresponding argument is provided. Every other part of the JSON
    structure is preserved verbatim. The written manifest is re-read and
    verified before returning.
    """
    source_split = Path(source_split)
    output_split = Path(output_split)

    if source_split == output_split:
        raise ValueError(f"Refusing to overwrite the source split in place: {source_split}")

    original = json.loads(source_split.read_text(encoding="utf-8"))

    payload = copy.deepcopy(original)
    folds, _ = _extract_folds(payload, source=source_split)

    if not -len(folds) <= fold < len(folds):
        raise IndexError(f"Fold {fold} out of range: {source_split} has {len(folds)} folds")

    entry = folds[fold]
    if not isinstance(entry, dict):
        raise ValueError(f"Fold {fold} is not a dict in {source_split}")

    if train_names is not None:
        entry["train"] = list(train_names)

    if test_names is not None:
        entry["test"] = list(test_names)

    output_split.parent.mkdir(parents=True, exist_ok=True)
    output_split.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    written = json.loads(output_split.read_text(encoding="utf-8"))
    if written != payload:
        raise RuntimeError(f"Subset split verification failed after write: {output_split}")

    return output_split


# ============================================================
# Fail-closed output guards
# ============================================================


def ensure_no_existing_outputs(
    *,
    log_path: Path | None = None,
    checkpoint_path: Path | None = None,
    prediction_dir: Path | None = None,
) -> None:
    """Fail closed rather than silently overwrite existing experiment evidence.

    ``prediction_dir`` is rejected only when it already contains ``*.geff``
    entries; a missing or empty directory is accepted.
    """
    if log_path is not None and Path(log_path).exists():
        raise FileExistsError(f"Refusing to overwrite existing log: {log_path}")

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {checkpoint_path}")

    if prediction_dir is not None:
        prediction_dir = Path(prediction_dir)
        if prediction_dir.exists():
            existing = sorted(prediction_dir.glob(f"*{_GEFF_SUFFIX}"))
            if existing:
                raise FileExistsError(
                    f"Refusing to overwrite {len(existing)} existing prediction(s) in {prediction_dir}: "
                    f"{[p.name for p in existing[:5]]}"
                )


# ============================================================
# Logged subprocess execution
# ============================================================


@dataclass(frozen=True)
class CommandResult:
    """Outcome of a single logged subprocess run."""

    command: list[str]
    return_code: int
    elapsed_seconds: float
    log_path: Path
    cwd: Path

    @property
    def ok(self) -> bool:
        return self.return_code == 0


def read_log_tail(path: Path, *, max_lines: int = _LOG_TAIL_LINES, max_bytes: int = _LOG_TAIL_MAX_BYTES) -> str:
    """Return roughly the last ``max_lines`` lines of ``path``.

    Reads backwards in blocks so a multi-gigabyte progress-bar log is never
    loaded into memory.
    """
    path = Path(path)
    if not path.exists():
        return ""

    block = 8192
    data = b""

    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()

        while position > 0 and data.count(b"\n") <= max_lines and len(data) < max_bytes:
            step = min(block, position)
            position -= step
            handle.seek(position)
            data = handle.read(step) + data

    lines = data.decode("utf-8", errors="replace").splitlines()

    # A backwards read can clip the first line mid-way; drop it when truncated.
    if position > 0 and lines:
        lines = lines[1:]

    return "\n".join(lines[-max_lines:])


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    check: bool = True,
) -> CommandResult:
    """Run ``command``, streaming stdout+stderr straight into ``log_path``.

    Notebook-facing code never receives tqdm/progress-bar spam: nothing is
    printed and the full log is never returned. On failure only the last
    ~100 log lines are read and attached to the raised ``RuntimeError``.
    """
    command = [str(part) for part in command]
    cwd = Path(cwd)
    log_path = Path(log_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()

    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    elapsed_seconds = time.monotonic() - start

    result = CommandResult(
        command=command,
        return_code=completed.returncode,
        elapsed_seconds=elapsed_seconds,
        log_path=log_path,
        cwd=cwd,
    )

    if check and completed.returncode != 0:
        tail = read_log_tail(log_path)
        raise RuntimeError(
            f"Command failed with return code {completed.returncode} after {elapsed_seconds:.1f}s.\n"
            f"Command: {command}\n"
            f"Log: {log_path}\n"
            f"--- last {_LOG_TAIL_LINES} log lines ---\n{tail}"
        )

    return result


# ============================================================
# Prediction set verification
# ============================================================


def verify_prediction_set(prediction_dir: Path, expected_names: list[str]) -> list[Path]:
    """Require exactly one ``.geff`` per expected dataset.

    Dataset stems are compared as sets; missing, extra, or duplicate outputs
    all fail closed. Returns the sorted GEFF paths.
    """
    prediction_dir = Path(prediction_dir)

    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory missing: {prediction_dir}")

    wanted = [str(name).removesuffix(_GEFF_SUFFIX) for name in expected_names]

    duplicate_expected = sorted({name for name in wanted if wanted.count(name) > 1})
    if duplicate_expected:
        raise ValueError(f"Duplicate expected dataset names: {duplicate_expected}")

    found_paths = sorted(prediction_dir.glob(f"*{_GEFF_SUFFIX}"))
    found = [path.name.removesuffix(_GEFF_SUFFIX) for path in found_paths]

    duplicate_found = sorted({name for name in found if found.count(name) > 1})
    if duplicate_found:
        raise ValueError(f"Duplicate predictions in {prediction_dir}: {duplicate_found}")

    missing = sorted(set(wanted) - set(found))
    extra = sorted(set(found) - set(wanted))

    if missing or extra:
        raise ValueError(
            f"Prediction set mismatch in {prediction_dir}: "
            f"expected {len(wanted)}, found {len(found)}; "
            f"missing={missing}; extra={extra}"
        )

    return found_paths


# ============================================================
# Notebook result reporting
# ============================================================


@dataclass
class ExperimentRun:
    """Neutral container for reporting one notebook experiment run.

    Intentionally policy-free: it records what was run and where the evidence
    landed. Any interpretation of ``metrics`` belongs in the notebook or in a
    dedicated, independently auditable module.
    """

    name: str
    command_result: CommandResult | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Flatten into a single row suitable for a notebook summary table."""
        row: dict[str, Any] = {"name": self.name}

        if self.command_result is not None:
            row["return_code"] = self.command_result.return_code
            row["elapsed_seconds"] = round(self.command_result.elapsed_seconds, 3)
            row["log_path"] = str(self.command_result.log_path)

        row.update({key: str(value) for key, value in self.artifacts.items()})
        row.update(self.metrics)
        row.update(self.notes)

        return row
