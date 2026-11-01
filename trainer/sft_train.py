#!/usr/bin/env python3
"""Reproducible SFT run: one command, one (n, seed) cell of the ablation.

    python trainer/sft_train.py --config trainer/configs/sft_v4.json --n 117 --seed 0

Needs a CUDA GPU and `pip install -r requirements-train.txt` (Colab T4 is the
target). `--dry-run` needs neither: it loads the config and data, selects the
subset, and prints the plan (steps, data hash, prompt, lengths).

What a run does
  1. loads the perfect teacher attempts from training_data.json (the same
     reward.score_to_reward filter as the rest of the pipeline);
  2. picks n of them with a seeded shuffle (nested: for one seed, the 20 are
     a subset of the 50 are a subset of the 117);
  3. fine-tunes Qwen2-1.5B-Instruct with LoRA via TRL's SFTTrainer
     (Unsloth if available, plain HF+PEFT otherwise; the backend actually
     used is logged), loss on the completion only;
  4. greedy-decodes one solution with the SAME prompt format the evaluators
     use (config.build_eval_prompt + config.format_chatml);
  5. scores it on the frozen stratified eval set and on the 24-case grader;
  6. logs params / loss / metrics to MLflow and writes a small summary JSON
     to experiments/ (the ledger's input) plus per-case results to runs/.

Design notes worth knowing
  * Every training example has the identical prompt (one task, one spec), so
    "more data" here means more distinct completions for one prompt.
  * Fixed epochs means steps grow with n (n=20 is ~3 optimizer steps per
    epoch, n=117 is 15). The `steps_matched` arm trains a small n for as many
    steps as n=117 gets, to separate "less data" from "fewer updates".
  * Nothing in this file has been executed against a GPU in the repo's CI;
    the first Colab step is a 2-step smoke run (see colab/run_phase1.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cj_config  # noqa: E402
import evaluation as ev  # noqa: E402
from trainer.reward import score_to_reward  # noqa: E402

REQUIRED_EOS = "<|im_end|>"


# ------------------------------------------------------------------ config
def load_config(path: Path, overrides: dict | None = None) -> dict:
    cfg = json.loads(Path(path).read_text())
    for dotted, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    for key in ("experiment", "task", "base_model", "backend", "lora", "train", "data", "eval"):
        if key not in cfg:
            raise ValueError(f"config is missing '{key}'")
    if cfg["backend"] not in {"unsloth", "hf"}:
        raise ValueError("backend must be 'unsloth' or 'hf'")
    return cfg


def config_sha256(cfg: dict) -> str:
    """Identity of the training recipe. Output locations don't change the recipe."""
    recipe = {k: v for k, v in cfg.items() if k not in {"mlflow", "summary_dir", "runs_dir"}}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


# -------------------------------------------------------------------- data
def load_perfect_examples(cfg: dict) -> list[dict]:
    """Teacher attempts that clear the reward threshold, with their original index."""
    raw = json.loads((ROOT / cfg["data"]["training_data"]).read_text())
    problem = cj_config.load_task_prompt(cfg["task"]).strip()
    out = []
    for idx, ex in enumerate(raw):
        if ex["task"] != cfg["task"] or score_to_reward(ex["score"], cfg["data"]["min_score"]) <= 0:
            continue
        if ex["prompt"].strip() != problem:
            raise ValueError(f"attempt {idx} was recorded against a different prompt than problem.md "
                             "(train/eval prompt mismatch)")
        out.append({"id": idx, "solution": ex["solution"].strip(), "score": ex["score"]})
    return out


def select_examples(examples: list[dict], n: int, seed: int) -> list[dict]:
    """Seeded shuffle, first n. Nested across n for a fixed seed."""
    if not 0 < n <= len(examples):
        raise ValueError(f"n={n} but only {len(examples)} examples available")
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    return [examples[i] for i in order[:n]]


def data_sha256(examples: list[dict]) -> str:
    canon = [(e["id"], hashlib.sha256(e["solution"].encode()).hexdigest()) for e in examples]
    return hashlib.sha256(json.dumps(sorted(canon)).encode()).hexdigest()


def build_records(examples: list[dict], task: str) -> list[dict]:
    """prompt/completion pairs. The prompt is built by the evaluators' own function."""
    prompt = cj_config.format_chatml(cj_config.build_eval_prompt(task))
    return [{"prompt": prompt, "completion": e["solution"]} for e in examples]


def steps_per_epoch(n: int, batch: int, accum: int) -> int:
    return math.ceil(n / (batch * accum))


def plan_epochs(n: int, cfg: dict, arm: str, target_steps: int | None) -> int:
    t = cfg["train"]
    if arm == "steps_matched":
        if not target_steps:
            raise ValueError("--arm steps_matched needs --target-steps")
        return math.ceil(target_steps / steps_per_epoch(n, t["per_device_batch_size"], t["grad_accum"]))
    return int(t["epochs"])


def data_diagnostics(examples: list[dict], records: list[dict]) -> dict:
    sols = [e["solution"] for e in examples]
    return {
        "n_examples": len(examples),
        "n_unique_prompts": len({r["prompt"] for r in records}),
        "n_unique_completions": len(set(sols)),
        "solution_chars": {"min": min(map(len, sols)), "max": max(map(len, sols)),
                           "mean": round(sum(map(len, sols)) / len(sols), 1)},
        "n_mentioning_written_number_words": sum(bool(ev.WRITTEN_LOGIC_RE.search(s)) for s in sols),
    }


def make_plan(cfg: dict, n: int, seed: int, arm: str, target_steps: int | None) -> dict:
    examples = select_examples(load_perfect_examples(cfg), n, seed)
    records = build_records(examples, cfg["task"])
    t = cfg["train"]
    epochs = plan_epochs(n, cfg, arm, target_steps)
    spe = steps_per_epoch(n, t["per_device_batch_size"], t["grad_accum"])
    return {
        "examples": examples, "records": records, "epochs": epochs,
        "steps_per_epoch": spe, "planned_optimizer_steps": spe * epochs,
        "effective_batch": t["per_device_batch_size"] * t["grad_accum"],
        "data_sha256": data_sha256(examples), "config_sha256": config_sha256(cfg),
        "data_diagnostics": data_diagnostics(examples, records),
    }


# ---------------------------------------------------------------- training
def _versions() -> dict:
    import importlib.metadata as md
    out = {}
    for pkg in ("torch", "transformers", "trl", "peft", "unsloth", "datasets", "mlflow", "mlflow-skinny", "accelerate"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def train_and_eval(cfg: dict, plan: dict, *, n: int, seed: int, arm: str, max_steps: int | None,
                   skip_eval: bool, summary_dir: Path, runs_dir: Path) -> dict:
    import dataclasses
    import inspect

    FastLanguageModel = None
    backend_actual = "hf"
    if cfg["backend"] == "unsloth":
        try:
            from unsloth import FastLanguageModel  # noqa: F811  (must be imported before transformers/trl/peft)
            backend_actual = "unsloth"
        except Exception as e:  # noqa: BLE001
            print(f"[warn] unsloth unavailable ({type(e).__name__}: {e}); falling back to HF+PEFT")

    import mlflow
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(seed)
    run_id = f"{cfg['experiment']}_{arm}_n{n}_s{seed}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    t, lo = cfg["train"], cfg["lora"]
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if backend_actual == "unsloth":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["base_model"], max_seq_length=cfg["max_seq_length"], dtype=None,
            load_in_4bit=cfg["load_in_4bit"])
        model = FastLanguageModel.get_peft_model(
            model, r=lo["r"], target_modules=lo["target_modules"], lora_alpha=lo["alpha"],
            lora_dropout=lo["dropout"], bias="none", use_gradient_checkpointing="unsloth", random_state=seed)
    else:
        from peft import LoraConfig, get_peft_model
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"], torch_dtype=torch.bfloat16 if use_bf16 else torch.float16, device_map={"": 0})
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"], target_modules=lo["target_modules"],
            bias="none", task_type="CAUSAL_LM"))
        for prm in model.parameters():  # fp16 AMP needs fp32 trainable params
            if prm.requires_grad:
                prm.data = prm.data.float()

    if tokenizer.eos_token != REQUIRED_EOS:
        raise RuntimeError(f"tokenizer.eos_token is {tokenizer.eos_token!r}, expected {REQUIRED_EOS!r}: the trained "
                           "model would not emit the stop token Ollama/ChatML expects")

    records = plan["records"]
    lengths = [len(tokenizer(r["prompt"] + r["completion"] + REQUIRED_EOS)["input_ids"]) for r in records]
    max_len = cfg["max_seq_length"]
    token_stats = {"max": max(lengths), "mean": round(sum(lengths) / len(lengths), 1),
                   "n_over_max_seq_length": sum(x > max_len for x in lengths), "max_seq_length": max_len}
    if token_stats["n_over_max_seq_length"]:
        print(f"[warn] {token_stats['n_over_max_seq_length']} examples exceed max_seq_length and will be truncated")

    sft_fields = {f.name for f in dataclasses.fields(SFTConfig)}
    if "completion_only_loss" not in sft_fields:
        raise RuntimeError("this TRL has no prompt/completion support (completion_only_loss); upgrade trl")
    kwargs = dict(
        output_dir=str(run_dir / "trainer"), num_train_epochs=plan["epochs"],
        per_device_train_batch_size=t["per_device_batch_size"], gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["learning_rate"], lr_scheduler_type=t["lr_scheduler"], warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"], optim=t["optim"], logging_steps=t["logging_steps"], seed=seed,
        data_seed=seed, fp16=not use_bf16, bf16=use_bf16, save_strategy="no", report_to="none",
        completion_only_loss=True, packing=False)
    kwargs["max_length" if "max_length" in sft_fields else "max_seq_length"] = max_len
    if max_steps:
        kwargs["max_steps"] = max_steps
    trainer_kwargs = {"processing_class": tokenizer} if "processing_class" in inspect.signature(
        SFTTrainer.__init__).parameters else {"tokenizer": tokenizer}
    trainer = SFTTrainer(model=model, args=SFTConfig(**kwargs), train_dataset=Dataset.from_list(records),
                         **trainer_kwargs)

    prov = ev.provenance()
    mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(cfg["experiment"])
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({
            "n_examples": n, "seed": seed, "arm": arm, "epochs": plan["epochs"], "backend_requested": cfg["backend"],
            "backend_actual": backend_actual, "base_model": cfg["base_model"], "data_sha256": plan["data_sha256"],
            "config_sha256": plan["config_sha256"], "planned_optimizer_steps": plan["planned_optimizer_steps"],
            "effective_batch": plan["effective_batch"], "code_commit": prov["code_commit"], "dirty": prov["dirty"],
            **{f"lora_{k}": v for k, v in lo.items() if k != "target_modules"},
            **{f"train_{k}": v for k, v in t.items()}})
        trainer.train()
        log_history = trainer.state.log_history
        for h in log_history:
            for key in ("loss", "grad_norm", "learning_rate"):
                if key in h:
                    mlflow.log_metric(f"train_{key}", h[key], step=h["step"])
        loss = ev.loss_summary(log_history)
        training = {"optimizer_steps": trainer.state.global_step, "loss": loss, "token_stats": token_stats,
                    "log_history": [{k: round(v, 5) if isinstance(v, float) else v for k, v in h.items()}
                                    for h in log_history if "loss" in h]}

        summary = {
            "schema": 1, "kind": "sft_run", "arm": arm, "run_id": run_id, "n_examples": n, "seed": seed,
            "model_tag": f"{cfg['base_model']}+lora:{run_id}", "base_model": cfg["base_model"],
            "backend_actual": backend_actual, "config": cfg, "config_sha256": plan["config_sha256"],
            "data_sha256": plan["data_sha256"], "epochs": plan["epochs"], **prov,
            "env": {**_versions(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "training": training, "data_diagnostics": plan["data_diagnostics"],
        }

        if not skip_eval:
            if backend_actual == "unsloth":
                FastLanguageModel.for_inference(model)
            else:
                model.eval()
            max_new = cfg["eval"]["max_new_tokens"]
            prompt_ids = tokenizer(cj_config.format_chatml(cj_config.build_eval_prompt(cfg["task"])),
                                   return_tensors="pt").to(model.device)
            stop_ids = [tokenizer.convert_tokens_to_ids(REQUIRED_EOS), tokenizer.eos_token_id]
            with torch.no_grad():
                out = model.generate(**prompt_ids, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids)
            new_ids = out[0][prompt_ids["input_ids"].shape[1]:]
            raw = tokenizer.decode(new_ids, skip_special_tokens=True)
            finish = "eos" if int(new_ids[-1]) in stop_ids else "length"

            from code_cleaning import clean_llm_code
            from evaluator import run_solution_against_grader
            cleaned = clean_llm_code(raw)
            cases, meta = ev.load_eval_set(ROOT / cfg["eval"]["eval_set"])
            results = ev.score_solution(cleaned, cases, workers=cfg["eval"]["workers"])
            scores = run_solution_against_grader(cleaned, cj_config.load_task_grader(cfg["task"]))
            eval_summary = ev.summarize(results, meta["corpus_counts"], n_boot=cfg["eval"]["bootstrap_resamples"])
            summary["eval_set_sha256"] = meta["cases_sha256"]
            summary["metrics"] = {"eval": eval_summary, "hand_grader": {
                "passed": scores.get("passed"), "total": scores.get("total"),
                "failed_tests": [x["name"] for x in scores.get("tests", []) if not x["passed"]]}}
            summary["diagnostics"] = {"generation": ev.diagnose_generation(
                raw, cleaned, n_new_tokens=len(new_ids), max_new_tokens=max_new, finish_reason=finish)}
            summary["solution"] = cleaned
            (run_dir / "eval_results.json").write_text(json.dumps(results))
            mlflow.log_metric("eval_weighted_accuracy", eval_summary["weighted"]["accuracy"])
            for c, v in eval_summary["per_category"].items():
                mlflow.log_metric(f"eval_acc_{c}", v["accuracy"])
            mlflow.log_metric("hand_grader_passed", scores.get("passed") or 0)

        model.save_pretrained(str(run_dir / "adapter"))
        summary_dir.mkdir(parents=True, exist_ok=True)
        out_path = summary_dir / f"{run_id}.json"
        out_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_artifact(str(out_path))
    return summary


# -------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "trainer" / "configs" / "sft_v4.json")
    ap.add_argument("--n", type=int, required=True, help="number of perfect teacher attempts to train on")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", choices=["scaling", "steps_matched"], default="scaling")
    ap.add_argument("--target-steps", type=int, help="optimizer steps to match (steps_matched arm)")
    ap.add_argument("--max-steps", type=int, help="cap steps (smoke test); the run is NOT a valid ablation cell")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="plan only: no GPU, no heavy imports")
    ap.add_argument("--summary-dir", type=Path)
    ap.add_argument("--mlflow-uri", help="override mlflow.tracking_uri (e.g. a Google Drive path on Colab)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, {"mlflow.tracking_uri": args.mlflow_uri} if args.mlflow_uri else None)
    plan = make_plan(cfg, args.n, args.seed, args.arm, args.target_steps)
    d = plan["data_diagnostics"]
    print(f"run: {cfg['experiment']} arm={args.arm} n={args.n} seed={args.seed}")
    print(f"  data sha256 {plan['data_sha256'][:12]}  config sha256 {plan['config_sha256'][:12]}")
    print(f"  {plan['steps_per_epoch']} steps/epoch x {plan['epochs']} epochs = {plan['planned_optimizer_steps']} "
          f"optimizer steps (effective batch {plan['effective_batch']})")
    print(f"  unique prompts {d['n_unique_prompts']}, unique completions {d['n_unique_completions']}, "
          f"completion chars {d['solution_chars']}")
    if args.dry_run:
        return 0
    summary_dir = ROOT / (args.summary_dir or cfg["summary_dir"])
    if args.max_steps:
        summary_dir = ROOT / "runs" / "smoke"
    summary = train_and_eval(cfg, plan, n=args.n, seed=args.seed, arm=args.arm, max_steps=args.max_steps,
                             skip_eval=args.skip_eval, summary_dir=summary_dir, runs_dir=ROOT / cfg["runs_dir"])
    w = summary.get("metrics", {}).get("eval", {}).get("weighted")
    if w:
        print(f"  weighted accuracy {w['accuracy'] * 100:.1f}%  (hand grader "
              f"{summary['metrics']['hand_grader']['passed']}/{summary['metrics']['hand_grader']['total']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
