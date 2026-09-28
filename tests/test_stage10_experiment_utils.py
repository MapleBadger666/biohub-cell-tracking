"""Focused tests for scripts/stage10_experiment_utils.py.

No training, no inference, no GT evaluation, no large data reads.
Everything runs against tmp_path fixtures.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stage10_experiment_utils import (
    CommandResult,
    ExperimentRun,
    build_project_env,
    ensure_no_existing_outputs,
    load_split_payload,
    project_python,
    read_log_tail,
    run_logged,
    sha256_file,
    verify_prediction_set,
    write_subset_split,
)

FOLD_PAYLOAD = {
    "meta": {"created": "stage10", "n_folds": 2},
    "folds": [
        {"split": 0, "train": ["a1", "a2", "a3"], "test": ["b1", "b2"]},
        {"split": 1, "train": ["c1"], "test": ["d1"]},
    ],
}

LIST_PAYLOAD = [
    {"split": 0, "train": ["a1", "a2", "a3"], "test": ["b1", "b2"]},
    {"split": 1, "train": ["c1"], "test": ["d1"]},
]


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _make_geff(directory: Path, stem: str) -> Path:
    """Create a .geff entry; in this project GEFFs are directories."""
    geff = directory / f"{stem}.geff"
    geff.mkdir(parents=True, exist_ok=True)
    (geff / "zarr.json").write_text("{}", encoding="utf-8")
    return geff


# ============================================================
# sha256_file
# ============================================================


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    blob = b"stage10-checkpoint-bytes" * 5000
    target = tmp_path / "weights.pt"
    target.write_bytes(blob)

    assert sha256_file(target) == hashlib.sha256(blob).hexdigest()


def test_sha256_file_chunking_is_size_independent(tmp_path: Path) -> None:
    blob = os.urandom(300_000)
    target = tmp_path / "big.pt"
    target.write_bytes(blob)

    expected = hashlib.sha256(blob).hexdigest()

    assert sha256_file(target, chunk_size=7) == expected
    assert sha256_file(target, chunk_size=1 << 20) == expected


def test_sha256_file_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "empty.bin"
    target.write_bytes(b"")

    assert sha256_file(target) == hashlib.sha256(b"").hexdigest()


# ============================================================
# project_python
# ============================================================


def test_project_python_returns_venv_interpreter(tmp_path: Path) -> None:
    interpreter = tmp_path / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")

    assert project_python(tmp_path) == tmp_path / ".venv/bin/python"


def test_project_python_fails_closed_when_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Project interpreter missing"):
        project_python(tmp_path)


def test_project_python_ignores_conda_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDA_PREFIX", "/opt/miniconda3/envs/other")

    interpreter = tmp_path / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")

    assert project_python(tmp_path) == tmp_path / ".venv/bin/python"


# ============================================================
# build_project_env
# ============================================================


def test_build_project_env_ordering_with_existing_pythonpath(tmp_path: Path) -> None:
    env = build_project_env(tmp_path, base_env={"PYTHONPATH": "/pre/existing"})

    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(tmp_path / "scripts"),
        str(tmp_path / "external/official/scripts"),
        "/pre/existing",
    ]


def test_build_project_env_without_existing_pythonpath(tmp_path: Path) -> None:
    env = build_project_env(tmp_path, base_env={})

    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(tmp_path / "scripts"),
        str(tmp_path / "external/official/scripts"),
    ]


def test_build_project_env_can_drop_existing_pythonpath(tmp_path: Path) -> None:
    env = build_project_env(
        tmp_path,
        include_existing_pythonpath=False,
        base_env={"PYTHONPATH": "/pre/existing"},
    )

    assert "/pre/existing" not in env["PYTHONPATH"]
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(tmp_path / "scripts"),
        str(tmp_path / "external/official/scripts"),
    ]


def test_build_project_env_preserves_other_vars_and_does_not_mutate_base(tmp_path: Path) -> None:
    base = {"HOME": "/Users/example", "PYTHONPATH": "/pre/existing"}

    env = build_project_env(tmp_path, base_env=base)

    assert env["HOME"] == "/Users/example"
    assert base["PYTHONPATH"] == "/pre/existing"


def test_build_project_env_defaults_to_os_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/from/os/environ")
    monkeypatch.setenv("STAGE10_MARKER", "kept")

    env = build_project_env(tmp_path)

    assert env["STAGE10_MARKER"] == "kept"
    assert env["PYTHONPATH"].split(os.pathsep)[-1] == "/from/os/environ"


# ============================================================
# load_split_payload
# ============================================================


def test_load_split_payload_folds_dict(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "dataset_splits.json", FOLD_PAYLOAD)

    payload, folds = load_split_payload(path)

    assert payload == FOLD_PAYLOAD
    assert payload["meta"]["n_folds"] == 2
    assert len(folds) == 2
    assert folds[0]["train"] == ["a1", "a2", "a3"]


def test_load_split_payload_direct_list(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "dataset_splits.json", LIST_PAYLOAD)

    payload, folds = load_split_payload(path)

    assert payload == LIST_PAYLOAD
    assert folds is payload
    assert folds[1]["test"] == ["d1"]


def test_load_split_payload_accepts_splits_key(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "dataset_splits.json", {"splits": LIST_PAYLOAD})

    _, folds = load_split_payload(path)

    assert folds[0]["train"] == ["a1", "a2", "a3"]


def test_load_split_payload_does_not_mutate_file(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "dataset_splits.json", FOLD_PAYLOAD)
    before = path.read_bytes()

    _, folds = load_split_payload(path)
    folds[0]["train"] = ["mutated-in-memory"]

    assert path.read_bytes() == before


def test_load_split_payload_rejects_unknown_structure(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "dataset_splits.json", {"unexpected": []})

    with pytest.raises(ValueError, match="none of"):
        load_split_payload(path)


# ============================================================
# write_subset_split
# ============================================================


def test_write_subset_split_changes_only_requested_fields(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "configs/dataset_splits.json", FOLD_PAYLOAD)
    output = tmp_path / "nested/out/subset.json"

    returned = write_subset_split(source, output, fold=0, train_names=["a1"], test_names=["b2"])

    assert returned == output
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["folds"][0]["train"] == ["a1"]
    assert written["folds"][0]["test"] == ["b2"]
    assert written["folds"][0]["split"] == 0
    # Untouched fold and sibling metadata survive verbatim.
    assert written["folds"][1] == FOLD_PAYLOAD["folds"][1]
    assert written["meta"] == FOLD_PAYLOAD["meta"]


def test_write_subset_split_only_train_leaves_test_intact(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "dataset_splits.json", FOLD_PAYLOAD)
    output = tmp_path / "subset.json"

    write_subset_split(source, output, fold=0, train_names=["a2"])

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["folds"][0]["train"] == ["a2"]
    assert written["folds"][0]["test"] == FOLD_PAYLOAD["folds"][0]["test"]


def test_write_subset_split_only_test_leaves_train_intact(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "dataset_splits.json", FOLD_PAYLOAD)
    output = tmp_path / "subset.json"

    write_subset_split(source, output, fold=1, test_names=["d1", "d2"])

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["folds"][1]["test"] == ["d1", "d2"]
    assert written["folds"][1]["train"] == FOLD_PAYLOAD["folds"][1]["train"]


def test_write_subset_split_direct_list_payload(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "dataset_splits.json", LIST_PAYLOAD)
    output = tmp_path / "subset.json"

    write_subset_split(source, output, fold=0, train_names=["a3"])

    written = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(written, list)
    assert written[0]["train"] == ["a3"]
    assert written[1] == LIST_PAYLOAD[1]


def test_write_subset_split_leaves_source_file_unchanged(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "dataset_splits.json", FOLD_PAYLOAD)
    before_bytes = source.read_bytes()
    before_sha = sha256_file(source)

    write_subset_split(source, tmp_path / "subset.json", fold=0, train_names=["a1"], test_names=["b1"])

    assert source.read_bytes() == before_bytes
    assert sha256_file(source) == before_sha


def test_write_subset_split_creates_parent_directories(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "dataset_splits.json", FOLD_PAYLOAD)
    output = tmp_path / "deeply/nested/dirs/subset.json"

    write_subset_split(source, output, fold=0, train_names=["a1"])

    assert output.exists()


def test_write_subset_split_rejects_out_of_range_fold(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "dataset_splits.json", FOLD_PAYLOAD)

    with pytest.raises(IndexError, match="out of range"):
        write_subset_split(source, tmp_path / "subset.json", fold=7, train_names=["a1"])


def test_write_subset_split_refuses_in_place_overwrite(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "dataset_splits.json", FOLD_PAYLOAD)

    with pytest.raises(ValueError, match="in place"):
        write_subset_split(source, source, fold=0, train_names=["a1"])


# ============================================================
# ensure_no_existing_outputs
# ============================================================


def test_ensure_no_existing_outputs_passes_on_clean_targets(tmp_path: Path) -> None:
    empty_dir = tmp_path / "predictions"
    empty_dir.mkdir()

    ensure_no_existing_outputs(
        log_path=tmp_path / "run.log",
        checkpoint_path=tmp_path / "model.pt",
        prediction_dir=empty_dir,
    )


def test_ensure_no_existing_outputs_rejects_existing_log(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text("previous evidence\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="existing log"):
        ensure_no_existing_outputs(log_path=log_path)


def test_ensure_no_existing_outputs_rejects_existing_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")

    with pytest.raises(FileExistsError, match="existing checkpoint"):
        ensure_no_existing_outputs(checkpoint_path=checkpoint)


def test_ensure_no_existing_outputs_rejects_existing_geff(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    _make_geff(prediction_dir, "44b6_0113de3b")

    with pytest.raises(FileExistsError, match="existing prediction"):
        ensure_no_existing_outputs(prediction_dir=prediction_dir)


def test_ensure_no_existing_outputs_allows_missing_prediction_dir(tmp_path: Path) -> None:
    ensure_no_existing_outputs(prediction_dir=tmp_path / "not_created_yet")


def test_ensure_no_existing_outputs_ignores_non_geff_entries(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    (prediction_dir / "notes.txt").write_text("scratch", encoding="utf-8")

    ensure_no_existing_outputs(prediction_dir=prediction_dir)


def test_ensure_no_existing_outputs_accepts_all_none() -> None:
    ensure_no_existing_outputs()


# ============================================================
# run_logged
# ============================================================


def test_run_logged_success_writes_log_and_returns_result(tmp_path: Path) -> None:
    log_path = tmp_path / "logs/run.log"

    result = run_logged(
        [sys.executable, "-c", "print('hello stage10')"],
        cwd=tmp_path,
        env=build_project_env(tmp_path, base_env=dict(os.environ)),
        log_path=log_path,
    )

    assert isinstance(result, CommandResult)
    assert result.return_code == 0
    assert result.ok
    assert result.log_path == log_path
    assert result.command[-1] == "print('hello stage10')"
    assert result.elapsed_seconds >= 0.0
    assert "hello stage10" in log_path.read_text(encoding="utf-8")


def test_run_logged_captures_stderr_into_same_log(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"

    run_logged(
        [sys.executable, "-c", "import sys; sys.stderr.write('progress-bar-spam\\n')"],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=log_path,
    )

    assert "progress-bar-spam" in log_path.read_text(encoding="utf-8")


def test_run_logged_does_not_print_to_notebook(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    run_logged(
        [sys.executable, "-c", "print('x' * 100)"],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=tmp_path / "run.log",
    )

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_run_logged_failure_raises_with_log_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"

    program = "import sys\nfor i in range(500):\n    print(f'line {i}')\nsys.exit(3)\n"

    with pytest.raises(RuntimeError) as excinfo:
        run_logged(
            [sys.executable, "-c", program],
            cwd=tmp_path,
            env=dict(os.environ),
            log_path=log_path,
        )

    message = str(excinfo.value)

    assert "return code 3" in message
    assert str(log_path) in message
    # Tail is present ...
    assert "line 499" in message
    # ... and the full log is not dumped.
    assert "line 0\n" not in message
    assert "line 100" not in message


def test_run_logged_check_false_returns_failure_result(tmp_path: Path) -> None:
    result = run_logged(
        [sys.executable, "-c", "import sys; sys.exit(5)"],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=tmp_path / "run.log",
        check=False,
    )

    assert result.return_code == 5
    assert not result.ok


def test_run_logged_uses_supplied_cwd_and_env(tmp_path: Path) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    log_path = tmp_path / "run.log"

    env = dict(os.environ)
    env["STAGE10_TEST_MARKER"] = "marker-value"

    run_logged(
        [sys.executable, "-c", "import os; print(os.getcwd()); print(os.environ['STAGE10_TEST_MARKER'])"],
        cwd=workdir,
        env=env,
        log_path=log_path,
    )

    contents = log_path.read_text(encoding="utf-8")
    assert str(workdir.resolve()) in contents
    assert "marker-value" in contents


def test_read_log_tail_limits_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "big.log"
    log_path.write_text("".join(f"line {i}\n" for i in range(5000)), encoding="utf-8")

    tail = read_log_tail(log_path, max_lines=10)

    assert tail.splitlines() == [f"line {i}" for i in range(4990, 5000)]


def test_read_log_tail_short_file_returns_everything(tmp_path: Path) -> None:
    log_path = tmp_path / "small.log"
    log_path.write_text("only line\n", encoding="utf-8")

    assert read_log_tail(log_path, max_lines=100) == "only line"


def test_read_log_tail_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_log_tail(tmp_path / "absent.log") == ""


# ============================================================
# verify_prediction_set
# ============================================================


def test_verify_prediction_set_exact_match(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "split_0"
    for stem in ("44b6_18ced818", "44b6_1d530831", "44b6_24264f12"):
        _make_geff(prediction_dir, stem)

    paths = verify_prediction_set(prediction_dir, ["44b6_24264f12", "44b6_18ced818", "44b6_1d530831"])

    assert [p.name for p in paths] == [
        "44b6_18ced818.geff",
        "44b6_1d530831.geff",
        "44b6_24264f12.geff",
    ]
    assert paths == sorted(paths)


def test_verify_prediction_set_accepts_names_with_geff_suffix(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "split_0"
    _make_geff(prediction_dir, "a1")

    paths = verify_prediction_set(prediction_dir, ["a1.geff"])

    assert [p.name for p in paths] == ["a1.geff"]


def test_verify_prediction_set_rejects_missing_prediction(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "split_0"
    _make_geff(prediction_dir, "a1")

    with pytest.raises(ValueError, match=r"missing=\['a2'\]"):
        verify_prediction_set(prediction_dir, ["a1", "a2"])


def test_verify_prediction_set_rejects_extra_prediction(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "split_0"
    _make_geff(prediction_dir, "a1")
    _make_geff(prediction_dir, "a2")

    with pytest.raises(ValueError, match=r"extra=\['a2'\]"):
        verify_prediction_set(prediction_dir, ["a1"])


def test_verify_prediction_set_rejects_duplicate_expected_names(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "split_0"
    _make_geff(prediction_dir, "a1")

    with pytest.raises(ValueError, match="Duplicate expected"):
        verify_prediction_set(prediction_dir, ["a1", "a1"])


def test_verify_prediction_set_ignores_non_geff_siblings(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "split_0"
    _make_geff(prediction_dir, "a1")
    (prediction_dir / "run.log").write_text("noise", encoding="utf-8")

    assert [p.name for p in verify_prediction_set(prediction_dir, ["a1"])] == ["a1.geff"]


def test_verify_prediction_set_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Prediction directory missing"):
        verify_prediction_set(tmp_path / "absent", ["a1"])


def test_verify_prediction_set_empty_expectation_requires_empty_dir(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "split_0"
    prediction_dir.mkdir()

    assert verify_prediction_set(prediction_dir, []) == []


# ============================================================
# ExperimentRun reporting dataclass
# ============================================================


def test_experiment_run_to_row_flattens_fields(tmp_path: Path) -> None:
    result = CommandResult(
        command=["python", "train.py"],
        return_code=0,
        elapsed_seconds=12.3456,
        log_path=tmp_path / "run.log",
        cwd=tmp_path,
    )

    run = ExperimentRun(
        name="SPARSE_9975",
        command_result=result,
        artifacts={"checkpoint": tmp_path / "model.pt"},
        metrics={"final_loss": 0.25},
        notes={"ignore_prob": "0.9975"},
    )

    row = run.to_row()

    assert row["name"] == "SPARSE_9975"
    assert row["return_code"] == 0
    assert row["elapsed_seconds"] == 12.346
    assert row["log_path"] == str(tmp_path / "run.log")
    assert row["checkpoint"] == str(tmp_path / "model.pt")
    assert row["final_loss"] == 0.25
    assert row["ignore_prob"] == "0.9975"


def test_experiment_run_defaults_are_independent() -> None:
    first = ExperimentRun(name="a")
    second = ExperimentRun(name="b")

    first.metrics["x"] = 1.0

    assert second.metrics == {}
    assert first.to_row() == {"name": "a", "x": 1.0}


# ============================================================
# Source lock (the frozen scripts must not be touched by this module)
# ============================================================


def test_frozen_training_and_predictor_scripts_unchanged() -> None:
    project = Path(__file__).resolve().parents[1]

    expected = {
        "scripts/train_unet_transformer_mps.py": "c6cd92b3562d8a8ad42d40944a545122ca23b42e85eacd04690abae8ee00cbc1",
        "scripts/predict_unet_transformer_mps.py": "23b76f7c90ffe698e955efee2b4333277a07f5b58b7eff6e580f2bb2ff49bc55",
    }

    for relative, digest in expected.items():
        assert sha256_file(project / relative) == digest, f"Source lock violated: {relative}"


def test_helper_module_compiles() -> None:
    module = Path(__file__).resolve().parents[1] / "scripts/stage10_experiment_utils.py"

    subprocess.run(
        [sys.executable, "-m", "py_compile", str(module)],
        check=True,
        capture_output=True,
    )
