"""End-to-end smoke test: `python run.py --smoke` on synthetic data in a temp dir (< 3 min)."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_end_to_end(tmp_path):
    ov = [f"paths.results_dir={tmp_path}/results", f"paths.cache_dir={tmp_path}/cache",
          f"paths.split_root={tmp_path}/splits", f"data.split_dir={tmp_path}/splits/synthetic2",
          f"data.root={tmp_path}/data/synthetic"]
    t0 = time.time()
    r = subprocess.run([sys.executable, "run.py", "--smoke", "--override", *ov], cwd=ROOT,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert time.time() - t0 < 180
    res = tmp_path / "results"
    for exp in ("exp1_borb", "exp2_noborb"):
        assert (res / exp / "mobilenetv3_large" / "seed0" / "metrics.json").exists()
    assert (res / "exp3_robustness" / "robustness_summary.csv").exists()
    assert (res / "summary.md").exists() and (res / "ablation.md").exists()
