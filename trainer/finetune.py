import json
import subprocess
import sys
from pathlib import Path

INPUT_FILE = Path("data/training_runs/filtered_data.json")
CHECKPOINT_DIR = Path("data/checkpoints/gemma3-codejudge")
BASE_MODEL = "google/gemma-3-4b-it"
MLX_DATA_FILE = Path("data/training_runs/train.jsonl")


def prepare_mlx_data():
    examples = json.loads(INPUT_FILE.read_text())
    print(f"Loaded {len(examples)} training examples")

    MLX_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MLX_DATA_FILE, "w") as f:
        for ex in examples:
            record = {
                "text": (
                    f"<instruction>{ex['instruction']}</instruction>\n"
                    f"<task>{ex['input']}</task>\n"
                    f"<solution>{ex['output']}</solution>"
                )
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(examples)} examples to {MLX_DATA_FILE}")
    return len(examples)


def main():
    n = prepare_mlx_data()
    if n < 10:
        print("Not enough training examples.")
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", BASE_MODEL,
        "--train",  
        "--data", str(MLX_DATA_FILE.parent),
        "--adapter-path", str(CHECKPOINT_DIR),
        "--num-layers", "2",
        "--batch-size", "1",
        "--iters", "100",
        "--learning-rate", "1e-4",
        "--steps-per-report", "20",
        "--save-every", "100",
    ]

    print(f"\nStarting fine-tuning with mlx-lm...")
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"\nAdapter saved to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()