#!/usr/bin/env python3
"""SFT -> DPO for one seed: a paired SFT-only vs SFT+DPO comparison.

    python trainer/dpo_train.py --config trainer/configs/dpo_v4.json --seed 0

Needs a CUDA GPU and `pip install -r requirements-train.txt` (Colab T4).
`--dry-run` needs neither: it builds the preference pairs and prints the plan.

What a run does
  1. builds the preference pairs (trainer/dpo_pairs.py) and checks their sha256
     against the committed data/dpo/*.meta.json;
  2. stage 1: trains the SFT model (sft_train.py in a subprocess: n=117, this
     seed, the Phase 1 recipe) and scores it on the frozen eval set. This is the
     SFT-only arm, measured in the same run so the comparison is paired;
  3. stage 2: merges the SFT adapter into the base weights, puts a fresh LoRA on
     top and trains it with TRL's DPOTrainer. The reference model is the merged
     SFT model (the DPO adapter switched off), not the raw base model;
  4. scores the DPO model on the same eval set and 24-case grader and writes
     experiments/phase2/phase2_dpo_s<seed>.json.

Stage 2 deliberately uses plain HF+PEFT, not Unsloth: it runs in a process that
has never imported Unsloth, and a merged model is a normal model.
Not done here: a QLoRA arm.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cj_config  # noqa: E402
import evaluation as ev  # noqa: E402
from trainer import dpo_pairs  # noqa: E402
from trainer import sft_train  # noqa: E402

SFT_EXPERIMENT = "phase2_sft"


def load_config(path: Path, overrides: dict | None = None) -> dict:
    cfg = json.loads(Path(path).read_text())
    for dotted, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    for key in ("experiment", "task", "sft_config", "sft_n", "pairs", "dpo"):
        if key not in cfg:
            raise ValueError(f"config is missing '{key}'")
    return cfg


def config_sha256(cfg: dict) -> str:
    recipe = {k: v for k, v in cfg.items() if k not in {"mlflow", "summary_dir", "runs_dir"}}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def sft_stage_config(cfg: dict) -> dict:
    """The Phase 1 SFT recipe, renamed so its run ids cannot collide with Phase 1's."""
    sft = sft_train.load_config(ROOT / cfg["sft_config"])
    sft["experiment"] = SFT_EXPERIMENT
    if sft["task"] != cfg["task"]:
        raise ValueError("sft_config and dpo config disagree on the task")
    return sft


def plan(cfg: dict, n_pairs: int) -> dict:
    d = cfg["dpo"]
    spe = math.ceil(n_pairs / (d["per_device_batch_size"] * d["grad_accum"]))
    return {"n_pairs": n_pairs, "steps_per_epoch": spe, "epochs": d["epochs"],
            "planned_optimizer_steps": spe * d["epochs"], "effective_batch": d["per_device_batch_size"] * d["grad_accum"]}


def warmup_kwargs(config_fields: set, ratio: float) -> dict:
    """transformers 5 renamed warmup_ratio to warmup_steps (a float below 1 is read as a ratio)."""
    if "warmup_ratio" in config_fields:
        return {"warmup_ratio": ratio}
    if "warmup_steps" in config_fields:
        return {"warmup_steps": ratio}
    raise RuntimeError("this transformers has neither warmup_ratio nor warmup_steps")


def build_pair_records(pairs: list[dict], task: str) -> list[dict]:
    """prompt/chosen/rejected strings; the prompt is the evaluators' own prompt function."""
    prompt = cj_config.format_chatml(cj_config.build_eval_prompt(task))
    return [{"prompt": prompt, "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]


def check_pairs_against_meta(meta: dict, seed: int, task: str) -> str:
    path = ROOT / "data" / "dpo" / f"{task}_pairs_seed{seed}.meta.json"
    if not path.exists():
        return "no committed meta to compare against"
    committed = json.loads(path.read_text())
    if committed["pairs_sha256"] != meta["pairs_sha256"]:
        raise RuntimeError(f"pairs differ from {path.name}: rebuilt {meta['pairs_sha256'][:12]}, "
                           f"committed {committed['pairs_sha256'][:12]}. Re-run trainer/dpo_pairs.py and commit.")
    return "matches committed meta"


# ----------------------------------------------------------------- stage 1
def run_sft_stage(cfg: dict, seed: int, max_steps: int | None) -> tuple[Path, dict]:
    sft = sft_stage_config(cfg)
    run_id = f"{SFT_EXPERIMENT}_scaling_n{cfg['sft_n']}_s{seed}"
    sft_cfg_path = ROOT / cfg["runs_dir"] / "phase2" / f"sft_stage_config_s{seed}.json"
    sft_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    sft_cfg_path.write_text(json.dumps(sft, indent=2))
    summary_rel = Path("runs") / ("smoke" if max_steps else "phase2/sft_stage")
    summary_path = ROOT / summary_rel / f"{run_id}.json"
    adapter = ROOT / sft["runs_dir"] / run_id / "adapter"
    if summary_path.exists() and adapter.exists():
        print(f"[stage 1] reusing {summary_path.name}")
    else:
        cmd = [sys.executable, "trainer/sft_train.py", "--config", str(sft_cfg_path), "--n", str(cfg["sft_n"]),
               "--seed", str(seed), "--summary-dir", str(summary_rel)]
        if max_steps:
            cmd += ["--max-steps", str(max_steps)]
        print("+", " ".join(cmd), flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            raise RuntimeError("SFT stage failed")
    return adapter, json.loads(summary_path.read_text())


# ----------------------------------------------------------------- stage 2
def _generate_and_score(model, tokenizer, sft_cfg: dict, run_dir: Path) -> dict:
    import torch
    from code_cleaning import clean_llm_code
    from evaluator import run_solution_against_grader

    model.eval()
    model.config.use_cache = True
    max_new = sft_cfg["eval"]["max_new_tokens"]
    prompt_ids = tokenizer(cj_config.format_chatml(cj_config.build_eval_prompt(sft_cfg["task"])),
                           return_tensors="pt").to(model.device)
    stop_ids = [tokenizer.convert_tokens_to_ids(sft_train.REQUIRED_EOS), tokenizer.eos_token_id]
    with torch.no_grad():
        out = model.generate(**prompt_ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids)
    new_ids = out[0][prompt_ids["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    finish = "empty" if len(new_ids) == 0 else ("eos" if int(new_ids[-1]) in stop_ids else "length")
    cleaned = clean_llm_code(raw)
    cases, meta = ev.load_eval_set(ROOT / sft_cfg["eval"]["eval_set"])
    results = ev.score_solution(cleaned, cases, workers=sft_cfg["eval"]["workers"])
    hand = run_solution_against_grader(cleaned, cj_config.load_task_grader(sft_cfg["task"]))
    (run_dir / "eval_results.json").write_text(json.dumps(results))
    return {
        "eval_set_sha256": meta["cases_sha256"],
        "eval": ev.summarize(results, meta["corpus_counts"], n_boot=sft_cfg["eval"]["bootstrap_resamples"]),
        "hand_grader": {"passed": hand.get("passed"), "total": hand.get("total"),
                        "failed_tests": [x["name"] for x in hand.get("tests", []) if not x["passed"]]},
        "generation": ev.diagnose_generation(raw, cleaned, n_new_tokens=len(new_ids), max_new_tokens=max_new,
                                             finish_reason=finish),
        "solution": cleaned,
    }


def run_dpo_stage(cfg: dict, sft_cfg: dict, records: list[dict], adapter: Path, *, seed: int,
                  max_steps: int | None, skip_eval: bool, run_dir: Path) -> dict:
    import mlflow
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(seed)
    d, lo = cfg["dpo"], cfg["dpo"]["lora"]
    base = sft_cfg["base_model"]
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.eos_token != sft_train.REQUIRED_EOS:
        raise RuntimeError(f"tokenizer.eos_token is {tokenizer.eos_token!r}, expected {sft_train.REQUIRED_EOS!r}")
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map={"": 0})
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, device_map={"": 0})
    # The SFT model becomes the base of stage 2; DPO's reference = this model with the new adapter off.
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"], target_modules=lo["target_modules"],
        bias="none", task_type="CAUSAL_LM"))
    for prm in model.parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()

    fields = {f.name for f in dataclasses.fields(DPOConfig)}
    kwargs = dict(
        output_dir=str(run_dir / "trainer"), beta=d["beta"], learning_rate=d["learning_rate"],
        num_train_epochs=d["epochs"], per_device_train_batch_size=d["per_device_batch_size"],
        gradient_accumulation_steps=d["grad_accum"], lr_scheduler_type=d["lr_scheduler"],
        logging_steps=d["logging_steps"], seed=seed, data_seed=seed,
        fp16=not use_bf16, bf16=use_bf16, save_strategy="no", report_to="none",
        **warmup_kwargs(fields, d["warmup_ratio"]))
    if "max_length" in fields:
        kwargs["max_length"] = d["max_length"]
    if "max_prompt_length" in fields:
        kwargs["max_prompt_length"] = 1024
    if max_steps:
        kwargs["max_steps"] = max_steps
    tk = {"processing_class": tokenizer} if "processing_class" in inspect.signature(
        DPOTrainer.__init__).parameters else {"tokenizer": tokenizer}
    trainer = DPOTrainer(model=model, ref_model=None, args=DPOConfig(**kwargs),
                         train_dataset=Dataset.from_list(records), **tk)

    prov = ev.provenance()
    mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(cfg["experiment"])
    with mlflow.start_run(run_name=f"{cfg['experiment']}_s{seed}"):
        mlflow.log_params({"seed": seed, "n_pairs": len(records), "code_commit": prov["code_commit"],
                           "dirty": prov["dirty"], **{f"dpo_{k}": v for k, v in d.items() if k != "lora"},
                           **{f"lora_{k}": v for k, v in lo.items() if k != "target_modules"}})
        trainer.train()
        log = trainer.state.log_history
        for h in log:
            for key, val in h.items():
                if isinstance(val, (int, float)) and key not in {"epoch", "step"} and "step" in h:
                    mlflow.log_metric(key.replace("/", "_"), val, step=h["step"])
        result = {"optimizer_steps": trainer.state.global_step,
                  "log_history": [{k: round(v, 5) if isinstance(v, float) else v for k, v in h.items()}
                                  for h in log if "loss" in h],
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if not skip_eval:
            model.gradient_checkpointing_disable()
            result["scored"] = _generate_and_score(model, tokenizer, sft_cfg, run_dir)
            mlflow.log_metric("eval_weighted_accuracy", result["scored"]["eval"]["weighted"]["accuracy"])
        model.save_pretrained(str(run_dir / "dpo_adapter"))
    return result


# -------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "trainer" / "configs" / "dpo_v4.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, help="cap steps in BOTH stages (smoke test); NOT a valid ablation cell")
    ap.add_argument("--skip-eval", action="store_true", help="skip the DPO-stage evaluation")
    ap.add_argument("--dry-run", action="store_true", help="plan only: no GPU, no heavy imports")
    ap.add_argument("--summary-dir", type=Path)
    ap.add_argument("--mlflow-uri")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, {"mlflow.tracking_uri": args.mlflow_uri} if args.mlflow_uri else None)
    sft_cfg = sft_stage_config(cfg)
    pairs, meta = dpo_pairs.build(ROOT / sft_cfg["data"]["training_data"], cfg["task"], seed=cfg["pairs"]["seed"],
                                  per_chosen=cfg["pairs"]["per_chosen"])
    meta_status = check_pairs_against_meta(meta, cfg["pairs"]["seed"], cfg["task"])
    p = plan(cfg, len(pairs))
    dg = meta["diagnostics"]
    print(f"run: {cfg['experiment']} seed={args.seed}  (stage 1: SFT n={cfg['sft_n']}, stage 2: DPO)")
    print(f"  pairs {dg['n_pairs']} sha {meta['pairs_sha256'][:12]} ({meta_status}); tiers {dg['tier_counts']}")
    print(f"  {p['steps_per_epoch']} steps/epoch x {p['epochs']} epochs = {p['planned_optimizer_steps']} DPO optimizer "
          f"steps (effective batch {p['effective_batch']}), beta {cfg['dpo']['beta']}, lr {cfg['dpo']['learning_rate']}")
    if args.dry_run:
        return 0

    run_id = f"{cfg['experiment']}_s{args.seed}"
    run_dir = ROOT / cfg["runs_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter, sft_summary = run_sft_stage(cfg, args.seed, args.max_steps)
    dpo = run_dpo_stage(cfg, sft_cfg, build_pair_records(pairs, cfg["task"]), adapter, seed=args.seed,
                        max_steps=args.max_steps, skip_eval=args.skip_eval, run_dir=run_dir)
    prov = ev.provenance()
    sft_metrics = sft_summary.get("metrics", {})
    summary = {
        "schema": 1, "kind": "dpo_run", "run_id": run_id, "seed": args.seed,
        "model_tag": f"{sft_cfg['base_model']}+sft:{sft_summary['run_id']}+dpo", "base_model": sft_cfg["base_model"],
        "config": cfg, "config_sha256": config_sha256(cfg), "sft_config_sha256": sft_summary["config_sha256"],
        "pairs": {"sha256": meta["pairs_sha256"], "meta_check": meta_status, "diagnostics": dg},
        "plan": p, **prov,
        "sft_stage": {"run_id": sft_summary["run_id"], "data_sha256": sft_summary["data_sha256"],
                      "optimizer_steps": sft_summary["training"]["optimizer_steps"],
                      "backend_actual": sft_summary["backend_actual"], "metrics": sft_metrics,
                      "diagnostics": sft_summary.get("diagnostics")},
        "dpo_stage": {k: v for k, v in dpo.items() if k != "scored"},
        "metrics": {"eval": dpo["scored"]["eval"], "hand_grader": dpo["scored"]["hand_grader"],
                    "eval_set_sha256": dpo["scored"]["eval_set_sha256"]} if "scored" in dpo else {},
        "diagnostics": {"generation": dpo["scored"]["generation"]} if "scored" in dpo else {},
        "solution": dpo["scored"]["solution"] if "scored" in dpo else None,
    }
    out_dir = ROOT / (args.summary_dir or cfg["summary_dir"])
    if args.max_steps:
        out_dir = ROOT / "runs" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)) + "\n")
    s_w = sft_metrics.get("eval", {}).get("weighted", {}).get("accuracy")
    d_w = summary["metrics"].get("eval", {}).get("weighted", {}).get("accuracy")
    if s_w is not None and d_w is not None:
        print(f"  weighted accuracy: SFT-only {s_w * 100:.1f}%  ->  SFT+DPO {d_w * 100:.1f}%")
    print(f"summary: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
