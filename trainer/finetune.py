import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config

BASE_MODEL = "google/gemma-3-4b-it"


def prepare_mlx_data(run_dir: Path) -> int:
    input_file = run_dir / "filtered_data.json"
    mlx_data_file = run_dir / "mlx_train.jsonl"

    if not input_file.exists():
        print(f"No filtered data at {input_file}. Run trainer/reward.py first.")
        sys.exit(1)

    examples = json.loads(input_file.read_text())
    print(f"Loaded {len(examples)} training examples")

    with open(mlx_data_file, "w") as f:
        for ex in examples:
            record = {
                "text": (
                    f"<instruction>{ex['instruction']}</instruction>\n"
                    f"<task>{ex['input']}</task>\n"
                    f"<solution>{ex['output']}</solution>"
                )
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(examples)} examples to {mlx_data_file}")
    return len(examples), mlx_data_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=config.DEFAULT_TASK)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    run_dir = config.DATA_DIR / "training_runs" / args.task
    checkpoint_dir = config.DATA_DIR / "checkpoints" / f"{args.task}-codejudge"

    n, mlx_data_file = prepare_mlx_data(run_dir)
    if n < 10:
        print("Not enough training examples.")
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", args.base_model,
        "--train",
        "--data", str(mlx_data_file.parent),
        "--adapter-path", str(checkpoint_dir),
        "--num-layers", str(args.num_layers),
        "--batch-size", "1",
        "--iters", str(args.iters),
        "--learning-rate", str(args.learning_rate),
        "--steps-per-report", "20",
        "--save-every", "100",
    ]

    print("\nStarting fine-tuning with mlx-lm...")
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"\nAdapter saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
