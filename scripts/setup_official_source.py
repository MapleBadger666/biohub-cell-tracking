"""Fetch the locked official evaluator without changing an existing checkout."""

from pathlib import Path
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL = PROJECT / "external" / "official"
COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
URL = "https://github.com/royerlab/kaggle-cell-tracking-competition.git"


def main() -> None:
    if not OFFICIAL.exists():
        subprocess.run(["git", "clone", URL, str(OFFICIAL)], check=True)
        subprocess.run(["git", "-C", str(OFFICIAL), "checkout", "--detach", COMMIT], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "status", "--porcelain"], text=True
    ).strip()
    if actual != COMMIT or dirty:
        raise SystemExit(
            "Existing official checkout is not clean at the locked commit. "
            "No existing source was modified; inspect external/official before proceeding."
        )
    print(f"Official source verified: {COMMIT}")


if __name__ == "__main__":
    main()
