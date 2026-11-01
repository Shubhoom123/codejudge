#!/usr/bin/env python3
"""Phase 1 on a Colab T4: smoke test first, then the full ablation matrix.

Cells to paste into a fresh Colab notebook (Runtime > T4 GPU):

    !git clone https://github.com/Shubhoom123/codejudge.git && cd codejudge && pip -q install -r requirements-train.txt
    %cd codejudge
    !python colab/run_phase1.py --smoke            # 2 optimizer steps, full path incl. eval; check it works
    !python colab/run_phase1.py --matrix --drive-dir /content/drive/MyDrive/codejudge_phase1   # after mounting Drive

(The repo must contain the commit you want to run: Colab clones from GitHub, so
push first. Use `--ref <sha>` to check out an exact commit.)

The matrix is resumable: a run whose summary JSON already exists is skipped,
so a disconnected session can be restarted with the same command. With
--drive-dir, summaries and the MLflow store are copied to Drive after every
run. Download experiments/phase1/*.json afterwards and put them in the repo:
tools/build_ledger.py ingests them as `recorded` entries.

Matrix: n in {20, 50, 117} x seeds {0, 1, 2} (the data-scaling ablation, fixed
epochs), plus a supplementary `steps_matched` arm (n=20, 50 trained for as many
optimizer steps as n=117) so "less data" can be told apart from "fewer updates".
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trainer import sft_train  # noqa: E402

SIZES = (117, 50, 20)  # biggest first: earliest signal on the written-number question
SEEDS = (0, 1, 2)


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def matrix(cfg: dict) -> list[dict]:
    t = cfg["train"]
    target = sft_train.steps_per_epoch(max(SIZES), t["per_device_batch_size"], t["grad_accum"]) * t["epochs"]
    cells = [{"arm": "scaling", "n": n, "seed": s} for s in SEEDS for n in SIZES]
    cells += [{"arm": "steps_matched", "n": n, "seed": s, "target": target} for s in SEEDS for n in (20, 50)]
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--config", default="trainer/configs/sft_v4.json")
    ap.add_argument("--drive-dir", type=Path)
    ap.add_argument("--ref", help="git ref to check out before running")
    ap.add_argument("--no-supplementary", action="store_true", help="skip the steps_matched arm")
    args = ap.parse_args()
    if not (args.smoke or args.matrix):
        ap.error("pass --smoke or --matrix")

    if args.ref:
        run(["git", "checkout", args.ref])
    run(["nvidia-smi"])
    cfg = sft_train.load_config(ROOT / args.config)
    base = [sys.executable, "trainer/sft_train.py", "--config", args.config]

    if args.smoke:
        rc = run(base + ["--n", "20", "--seed", "0", "--max-steps", "2"])
        print("SMOKE", "PASSED" if rc == 0 else "FAILED", "(summary in runs/smoke/)")
        return rc

    summary_dir = ROOT / cfg["summary_dir"]
    mlflow_uri = f"file:{args.drive_dir}/mlruns" if args.drive_dir else None
    failures = 0
    for cell in matrix(cfg):
        if cell["arm"] == "steps_matched" and args.no_supplementary:
            continue
        run_id = f"{cfg['experiment']}_{cell['arm']}_n{cell['n']}_s{cell['seed']}"
        if (summary_dir / f"{run_id}.json").exists():
            print(f"skip {run_id} (summary exists)")
            continue
        cmd = base + ["--n", str(cell["n"]), "--seed", str(cell["seed"]), "--arm", cell["arm"]]
        if cell["arm"] == "steps_matched":
            cmd += ["--target-steps", str(cell["target"])]
        if mlflow_uri:
            cmd += ["--mlflow-uri", mlflow_uri]
        if run(cmd) != 0:
            failures += 1
            print(f"FAILED {run_id}; continuing with the rest")
            continue
        if args.drive_dir:
            args.drive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(summary_dir, args.drive_dir / "summaries", dirs_exist_ok=True)
    zip_path = shutil.make_archive("/content/phase1_summaries" if Path("/content").exists() else
                                   str(ROOT / "phase1_summaries"), "zip", summary_dir)
    print(f"{failures} failed run(s). Summaries zipped at {zip_path}: download it into experiments/phase1/.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
