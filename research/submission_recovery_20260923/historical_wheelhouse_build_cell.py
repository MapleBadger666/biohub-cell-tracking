
# ============================================================
# Notebook 14 / 14.15D-1
# Build a Kaggle Linux / Python 3.12 offline dependency overlay.
#
# Downloads pinned wheels and builds tracksdata from the exact
# source commit. Does NOT install, import project code, infer,
# track, evaluate, or read GT.
# ============================================================

import sys
import json
import hashlib
import subprocess
import shutil
import zipfile
from pathlib import Path
from importlib import metadata as importlib_metadata

assert sys.platform == "linux"
assert sys.version_info[:2] == (3, 12), (
    f"Expected Kaggle Python 3.12, got {sys.version}"
)

WORKING = Path("/kaggle/working")
assert WORKING.is_dir()

BUNDLE = Path(
    "/kaggle/input/datasets/markdd520/"
    "biohub-frozen-v9-warm2-bundle/bundle"
)
assert (BUNDLE / "bundle_manifest.json").is_file()

OUT = WORKING / "biohub_wheelhouse_py312_v1"
WHEELS = OUT / "wheels"

ZIP_PATH = WORKING / "biohub_wheelhouse_py312_v1.zip"

# Never overwrite an earlier build.
assert not OUT.exists(), f"Output already exists: {OUT}"
assert not ZIP_PATH.exists(), f"ZIP already exists: {ZIP_PATH}"

TRACKSDATA_COMMIT = (
    "7bfeaf845ceb951226f19b72fe5b80e01601018a"
)

TRACKSDATA_URL = (
    "git+https://github.com/royerlab/tracksdata@"
    + TRACKSDATA_COMMIT
)

# Packages actually added or aligned during the successful
# Kaggle import-repair sequence.
# Base packages such as torch/numpy/pandas are intentionally
# not downloaded here; they will be checked in a clean runtime.

PINNED = [
    "zarr==3.1.6",
    "numcodecs==0.15.1",
    "donfig==0.8.1.post1",
    "geff==1.3.0.1.2",
    "geff-spec==1.2.0",
    "polars==1.43.2",
    "polars-runtime-32==1.43.2",
    "bidict==0.23.1",
    "ilpy==0.6.0",
    "pyscipopt==6.2.1",
    "rustworkx==0.18.1",
    "imagecodecs==2026.3.6",
]

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()

def run_checked(command, label):
    print(f"\n========== {label} ==========", flush=True)
    print("COMMAND =", command, flush=True)

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )

    print(result.stdout[-20000:], flush=True)
    print("EXIT_CODE =", result.returncode, flush=True)

    assert result.returncode == 0, (
        f"{label} failed. Stop; preserve the partial build."
    )

OUT.mkdir(parents=False, exist_ok=False)
WHEELS.mkdir(parents=False, exist_ok=False)

# ------------------------------------------------------------
# 1. Download known pinned Linux-compatible binary wheels
# ------------------------------------------------------------

run_checked(
    [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--no-deps",
        "--only-binary=:all:",
        "--dest",
        str(WHEELS),
        *PINNED,
    ],
    "DOWNLOAD PINNED WHEELS",
)

# ------------------------------------------------------------
# 2. Build the exact source-pinned tracksdata wheel
# ------------------------------------------------------------

run_checked(
    [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--wheel-dir",
        str(WHEELS),
        TRACKSDATA_URL,
    ],
    "BUILD TRACKSDATA",
)

# ------------------------------------------------------------
# 3. Verify wheel inventory and tracksdata metadata
# ------------------------------------------------------------

wheel_files = sorted(WHEELS.glob("*.whl"))

assert len(wheel_files) == len(PINNED) + 1, (
    f"Expected {len(PINNED) + 1} wheels, "
    f"found {len(wheel_files)}"
)

tracksdata_files = [
    path
    for path in wheel_files
    if path.name.lower().startswith("tracksdata-")
]

assert len(tracksdata_files) == 1, (
    "Expected exactly one tracksdata wheel"
)

tracksdata_wheel = tracksdata_files[0]

with zipfile.ZipFile(tracksdata_wheel) as archive:
    metadata_names = [
        name for name in archive.namelist()
        if name.endswith(".dist-info/METADATA")
    ]

    assert len(metadata_names) == 1

    metadata_text = archive.read(
        metadata_names[0]
    ).decode("utf-8")

expected_version = "0.1.0rc9.dev4+g7bfeaf845"

assert f"Version: {expected_version}" in metadata_text, (
    "tracksdata wheel version does not match the frozen commit"
)

# Record the newly produced wheel identity.
# Do not substitute either historical SHA.
tracksdata_sha = sha256_file(tracksdata_wheel)

print("\n========== TRACKSDATA IDENTITY ==========")
print("GIT_URL =", TRACKSDATA_URL)
print("WHEEL =", tracksdata_wheel.name)
print("VERSION =", expected_version)
print("NEW_WHEEL_SHA256 =", tracksdata_sha)

print("HISTORICAL_KAGGLE_WHEEL_SHA256 =",
      "9e5ecf8cc4e297c55515e836109c34eaee7103a27b73563909db06e0b0bea8c4")

print("LOCAL_CODEX_WHEEL_SHA256 =",
      "a12b65a10ab50f1449c532de72c8bf0c1cf0495088915dc93d4a30db93071df8")

# ------------------------------------------------------------
# 4. Produce an auditable manifest
# ------------------------------------------------------------

wheel_records = []

for path in wheel_files:
    record = {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    wheel_records.append(record)

    print(
        "WHEEL",
        record["filename"],
        "BYTES",
        record["size_bytes"],
        "SHA256",
        record["sha256"],
    )

manifest = {
    "stage": "14.15D-1",
    "purpose": "kaggle_py312_offline_dependency_overlay",
    "python": "3.12",
    "platform": "linux_x86_64",
    "source_bundle_manifest_sha256":
        "615c3550cba46a5032344549d2769201af22c0425beb9349726874669ae9c865",
    "tracksdata_git_url": TRACKSDATA_URL,
    "tracksdata_commit": TRACKSDATA_COMMIT,
    "tracksdata_wheel_sha256": tracksdata_sha,
    "pinned_requirements": PINNED,
    "wheels": wheel_records,
    "torch_numpy_pandas_expected_from_base": {
        "torch": "2.10.0+cu128",
        "numpy": "2.0.2",
        "pandas": "2.3.3",
    },
    "status": "BUILT_NOT_OFFLINE_TESTED",
}

manifest_path = OUT / "wheelhouse_manifest.json"

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

# ------------------------------------------------------------
# 5. Create one portable ZIP for private Kaggle Dataset upload
# ------------------------------------------------------------

created = shutil.make_archive(
    str(ZIP_PATH.with_suffix("")),
    "zip",
    root_dir=OUT,
)

assert Path(created) == ZIP_PATH
assert ZIP_PATH.is_file()

print("\n========== 14.15D-1 RESULT ==========")
print("WHEEL_COUNT =", len(wheel_files))
print("MANIFEST =", manifest_path)
print("MANIFEST_SHA256 =", sha256_file(manifest_path))
print("ZIP =", ZIP_PATH)
print("ZIP_BYTES =", ZIP_PATH.stat().st_size)
print("ZIP_SHA256 =", sha256_file(ZIP_PATH))

print("\n14.15D1_WHEELHOUSE_BUILT")
print("OFFLINE_INSTALL_VERIFIED=NO")
print("GT_READ=NO")
print("INFERENCE_RUN=NO")
print("TRACKING_RUN=NO")