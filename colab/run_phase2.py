#!/usr/bin/env python3
"""Phase 2 on a Colab T4: smoke test first, then SFT vs SFT+DPO for seeds 0, 1, 2.

Cells to paste into a fresh Colab notebook (Runtime > T4 GPU), after the Phase 1 setup cell:

    !python colab/run_phase2.py --smoke        # 2 steps in each stage; checks the whole path
    !python colab/run_phase2.py --matrix --drive-dir /content/drive/MyDrive/codejudge_phase2

Each seed trains the SFT model (n=117, Phase 1 recipe), scores it, then trains DPO on top and
scores that, so every seed gives a paired SFT-only / SFT+DPO result. Resumable: a seed whose
summary JSON already exists is skipped. Summaries land in experiments/phase2/.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trainer import dpo_train  # noqa: E402

SEEDS = (0, 1, 2)


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def matrix(cfg: dict) -> list[dict]:
    return [{"seed": s, "run_id": f"{cfg['experiment']}_s{s}"} for s in SEEDS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--config", default="trainer/configs/dpo_v4.json")
    ap.add_argument("--drive-dir", type=Path)
    args = ap.parse_args()
    if not (args.smoke or args.matrix):
        ap.error("pass --smoke or --matrix")

    run(["nvidia-smi"])
    cfg = dpo_train.load_config(ROOT / args.config)
    base = [sys.executable, "trainer/dpo_train.py", "--config", args.config]
    if args.smoke:
        rc = run(base + ["--seed", "0", "--max-steps", "2"])
        print("SMOKE", "PASSED" if rc == 0 else "FAILED", "(summary in runs/smoke/)")
        return rc

    summary_dir = ROOT / cfg["summary_dir"]
    failures = 0
    for cell in matrix(cfg):
        if (summary_dir / f"{cell['run_id']}.json").exists():
            print(f"skip {cell['run_id']} (summary exists)")
            continue
        if run(base + ["--seed", str(cell["seed"])]) != 0:
            failures += 1
            print(f"FAILED {cell['run_id']}; continuing with the rest")
            continue
        if args.drive_dir:
            args.drive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(summary_dir, args.drive_dir / "summaries", dirs_exist_ok=True)
            if (ROOT / "mlflow.db").exists():
                shutil.copy2(ROOT / "mlflow.db", args.drive_dir / "mlflow.db")
    zip_path = shutil.make_archive("/content/phase2_summaries" if Path("/content").exists() else
                                   str(ROOT / "phase2_summaries"), "zip", summary_dir)
    print(f"{failures} failed run(s). Summaries zipped at {zip_path}: download it into experiments/phase2/.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
