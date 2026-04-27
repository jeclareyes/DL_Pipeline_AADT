"""Run minimal phase-5 CI checks in a stable order.

This runner is intentionally Python-based so it can be reused both locally and in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[CI-MIN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    python_bin = sys.executable

    commands = [
        [python_bin, str(repo_root / "src/test/model_contract_regression.py")],
        [python_bin, str(repo_root / "tests/vi_trainer_controls_regression.py")],
        [python_bin, str(repo_root / "tests/vi_gradient_flow_regression.py")],
        [python_bin, str(repo_root / "tests/vi_grouped_params_unknown_init_regression.py")],
    ]

    for cmd in commands:
        _run(cmd)

    print("[CI-MIN] All checks passed")


if __name__ == "__main__":
    main()
