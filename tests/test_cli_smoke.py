from __future__ import annotations

import subprocess
import sys


def test_cli_smoke():
    cmd = [
        sys.executable,
        "-m",
        "kvbench.cli",
        "bench-kv-scaling",
        "--config",
        "configs/bench/kv_scaling.yaml",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "kv_mib" in result.stdout
