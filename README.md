# CodeJudge

CodeJudge is an evaluation and fine-tuning attempt for LLM-generated code. `evaluator.py` reads a task's `problem.md` then prepends a fixed system prompt which tells the model to output only the raw Python, and POSTs that to either `Ollama` or the `OpenAI API`. The model's raw text response gets cleaned up, `code_cleaning.py` strips markdown fences and any trailing self-test junk. That code is then treated as untrusted and it's written to a temp file which runs as a separate subprocess with 5 basic security features (CPU-time cap, a memory cap, a thread/process-count cap, a wall-clock timeout, and a stripped environment with no API keys in it). Basically emulating a sandbox environment.

It started as a way to answer one question: when a small local model (`qwen2:1.5b`) fails at a coding task, is that failure fixable with a modest amount of targeted training data, or does it need a bigger model? The harness exists to answer that question repeatably, instead of by eyeballing a handful of outputs.

```bash
python evaluator.py --model qwen2:1.5b --task ambigous_spec
```

---

## Contents

- [What it does](#what-it-does)
- [Tasks](#tasks)
- [Architecture](#architecture)
- [Sandboxing untrusted code](#sandboxing-untrusted-code)
- [Cleaning model output](#cleaning-model-output)
- [Test-case generation at scale](#test-case-generation-at-scale)
- [Fine-tuning](#fine-tuning)
- [REST API](#rest-api)
- [Testing and CI](#testing-and-ci)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Usage](#usage)
- [Known limitations](#known-limitations)
- [License](#license)

---

## What it does

The pipeline has five stages. Each one produces a file the next stage consumes, so any step can be rerun on its own without redoing the ones before it.

1. **Evaluate.** Send a task's spec to a model (via Ollama or the OpenAI API), capture the code it returns, and run that code against a hand-written grader.
2. **Benchmark at scale.** Sample thousands of programmatically generated test cases from a multi-million-case corpus and report pass rate by category, not just an aggregate score.
3. **Generate training data.** Sample a stronger "teacher" model against a task repeatedly, keep only the attempts that score well against the grader, and write them out as instruction/output pairs.
4. **Fine-tune.** Train a weaker "student" model on that data, either via LoRA on Colab (Unsloth/TRL, exported to GGUF and loaded into Ollama) or locally on Apple Silicon (`mlx-lm`).
5. **Re-evaluate.** Run the fine-tuned model through the same benchmark, so "did this help" is a diff between two JSON files instead of a guess.

None of this is hardcoded to one task. Everything is driven by `config.py` and a `tasks/<name>/` directory, so adding a fourth task means adding a `problem.md` and a `grader.py`. Nothing else in the pipeline needs to change.

---

## Tasks

Three tasks currently live under `tasks/`, and each one is chosen to expose a different, specific way LLM-generated code tends to fail.

| Task | Function contract | What it's testing for |
|---|---|---|
| `ambigous_spec` | `parse_amount(text: str) -> float` | Whether the model actually handles the input formats it's told about (`"$12.50"`, `"100 dollars"`, `"fifty"`, `"$1,000"`, negatives, whitespace), instead of just handling the one or two examples shown in the prompt. |
| `edge_case_trap` | `most_frequent(lst: list) -> any` | Whether the model's solution silently breaks on edge cases the spec doesn't spell out: `None` in the list, `True` vs. `1` colliding as dict keys, ties, large inputs. |
| `race_condition` | `Counter` class with `increment()` / `value` | Whether the model can spot and fix a real concurrency bug (a non-atomic `self.count += 1` under `threading`), rather than just leaving the starter code's race condition in place. |

`ambigous_spec/hint.md` exists alongside `problem.md` for a reason. `problem.md` is the deliberately underspecified prompt a model actually sees during evaluation, while `hint.md` is a fully enumerated version used only when generating teacher training data. The teacher needs to see the whole contract to produce consistent training examples, but the student being evaluated shouldn't get the answer key baked into its prompt.

Each `grader.py` runs a fixed set of named sub-tests and prints a single JSON object: `{"tests": [...], "passed": N, "total": M, "score": N/M}`. That's the only contract a grader has to satisfy, and it's what lets `evaluator.py` stay task-agnostic.

---

## Architecture

```
 task spec (problem.md)
        │
        ▼
 evaluator.py ──► Ollama / OpenAI ──► raw model output
        │
        ▼
 code_cleaning.py   (strip markdown fences, drop trailing side effects)
        │
        ▼
 sandbox.py::run_isolated()   (resource-limited subprocess)
        │
        ▼
 tasks/<task>/grader.py   (runs inside the sandbox, prints JSON)
        │
        ▼
 results/run_<timestamp>.json ──► analyze.py (pass/fail breakdown by test name)
```

`run_isolated()` is the only place model-generated code actually executes, whether it's called from `evaluator.py` (a single grader run), `tools/evaluate_at_scale.py` (thousands of sampled cases against one solution), or `api/main.py` (the same path over HTTP). There's exactly one execution boundary in the codebase. That's on purpose: a second, slightly different sandboxing path is usually how these things quietly stop being sandboxed.

---

## Sandboxing untrusted code

`solution_code` in `run_solution_against_grader()` is whatever the model wrote. It is never `import`ed or `exec()`'d in-process. It's always written to a temp file and run as a separate subprocess with:

- a hard wall-clock timeout (`subprocess.run(..., timeout=...)`)
- a CPU-time limit via `resource.setrlimit(RLIMIT_CPU, ...)`
- a memory ceiling via `RLIMIT_AS`
- a process/thread-count ceiling via `RLIMIT_NPROC`
- a stripped environment (no inherited API keys or secrets, just `PATH` and two encoding-related variables)
- its own throwaway temp directory as `cwd`

This is defense-in-depth, not a container. It doesn't restrict filesystem access or network egress: a solution that calls `requests.get(...)` or `open("/etc/passwd")` can still do it. A harder boundary than this would mean running the harness itself inside a container or VM with a read-only filesystem and no network.

### A real bug this caught, twice, from two different directions

The `RLIMIT_AS` memory limit is enforced reliably by the Linux kernel but not reliably by macOS for large anonymous allocations. A `bytearray(...)` big enough to get killed on Linux can run to completion on macOS. `tests/test_sandbox.py` skips that specific assertion on Darwin, with an explanation, rather than pretending the guarantee holds everywhere. CI runs on Ubuntu specifically so the memory limit is actually verified somewhere for real.

The process-count limit had the opposite kind of surprise. It was originally set to 16, which seemed generous for stopping a fork bomb. Then `race_condition`'s grader, which legitimately spawns 50 real threads to test lock contention, started failing in CI with `can't start new thread`, even for a *correct* `Counter` implementation. It turns out `RLIMIT_NPROC` on Linux isn't scoped to the sandboxed subprocess tree. It's a per-real-UID limit that counts every process and thread the CI runner's user already owns. 16 wasn't a fork-bomb guard at that point, it was an accidental cap on legitimate concurrency, and it only showed up on Linux because macOS doesn't enforce the limit the same way. The fix in `sandbox.py` raised the limit to 256 and left a comment explaining why the number isn't arbitrary. Both of these are documented in `sandbox.py`'s docstring instead of being left for someone else to rediscover.

---

## Cleaning model output

Models don't reliably follow "output only raw Python, no markdown." Two failure modes showed up often enough to need a shared fix in `code_cleaning.py`.

The first is markdown fences: ```` ```python ... ``` ```` wrapped around otherwise correct code. The second is trailing self-test code. A model finishes a correct function, then appends something like a `check_solution()` call to "prove" it works, and that appended code references a name that was never defined (an unquoted `fifty` instead of `"fifty"`, for instance). That crashes the whole module on `import` with a `NameError` before a single real grader test runs, and scores an otherwise-correct solution as a total failure.

`strip_top_level_side_effects()` fixes the second case by parsing the code with `ast.parse()` and keeping only `import`/`def`/`class` nodes at module level. That's real parsing, not a regex or a line-prefix heuristic, so it isn't fooled by unusual formatting. If the code doesn't parse as valid Python at all, it's returned unchanged; a genuine syntax error should surface naturally when the grader tries to import it, which is a more honest failure mode than silently rewriting unparseable input.

This used to be three separate, slightly different implementations of "strip the fences" scattered across `evaluator.py`, `tools/evaluate_at_scale.py`, and `tools/debug_model.py`, and only one of the three had any protection against the trailing-code problem. `tests/test_code_cleaning.py` includes a regression test built from the exact case that surfaced this: a `qwen2-codejudge` output where `parse_amount` was correct but a trailing `check_parse_amount()` referencing bare `fifty` crashed every single test in the grader.

---

## Test-case generation at scale

`ambigous_spec/grader.py` has 24 hand-written test cases. That's enough to catch obvious breakage, not enough to be confident a model handles the full format space the spec implies. `tools/generate_test_cases_bulk.py` generates a much larger, sharded corpus instead of hand-writing more cases one at a time.

The scaling lever is combinatorial, not arbitrary. Every positive integer in a chosen range has exactly one canonical written-number phrase (`482613` becomes `"four hundred eighty two thousand six hundred thirteen"`), generated by `number_to_words()`, which is the literal inverse of the word-parsing logic in the reference `parse_amount()`. Every generated phrase is round-tripped back through that same reference solution before being kept: if the reference can't parse its own phrase back to the original number, the case is dropped rather than trusted. Format variants (`$`-prefixed, comma-thousands, `"N dollars"`, whitespace padding, negative, decimal cents) are layered on top of that same integer sweep using modulo gates (`n % 7`, `n % 11`, and so on) so no single category dominates the corpus.

A handful of edge cases aren't naturally combinatorial: things like `"banana"` or `"$$100"` that are supposed to be invalid, or `"zero"` / `"$0.00"` that are supposed to parse to zero. Those are validated the same way as everything else. `build_fixed_edge_cases()` runs each hand-picked candidate through the reference solution and only keeps it if it actually behaves the way its category claims. `"$$100"` was originally assumed to raise `ValueError`. It doesn't, because the reference just strips both `$` characters, and this validation step is what caught that before it became a silently wrong test case.

Generation runs across a `multiprocessing.Pool`, and each worker streams cases straight to disk as newline-delimited JSON shards (`test_case_store.py`) instead of building one giant list in memory. A multi-million-case JSON array would need to be loaded whole just to read one record, and it doesn't belong in git anyway. Sampling from a corpus that size uses reservoir sampling (`reservoir_sample()`, Algorithm R), so pulling *k* cases for an evaluation run costs O(k) memory regardless of how large the corpus is.

A real run on a 2020 MacBook, 4 worker processes:

```
Total cases: 4,233,458
Generation time: 5.11s  (827,894 cases/sec)

By category:
  written_number:   2,000,000
  whitespace:          769,230
  negative:            307,692
  dollar_sign:         285,715
  word_dollars:        285,715
  plain_number:        285,715
  comma_thousands:     181,728
  decimal:             117,648
  invalid:                   9
  zero:                      6
```

That manifest (`data/test_cases_bulk/manifest.json`) is committed as evidence of what an actual generation run produced; the shard files themselves are gitignored, since they aren't something a git repo should hold. One limitation is called out directly in the manifest rather than glossed over: deduplication happens per worker, not globally, since workers operate on disjoint integer ranges. Exact duplicates across workers are possible in principle, just rare in practice. That's fine for a sampled evaluation corpus, not a claim of a perfectly unique set.

---

## Fine-tuning

Two separate fine-tuning paths exist, aimed at two different constraints.

### Colab / Unsloth (GPU, cloud)

```bash
python trainer/generate_training_data.py --task ambigous_spec --teacher-model gemma3:12b --attempts 100
python trainer/reward.py --task ambigous_spec
```

`generate_training_data.py` samples the teacher model repeatedly at `temperature=0.8` (diversity on purpose: more varied attempts means more good ones to choose from) against a task's grader, and records every attempt's score to `training_data.json`. It doesn't filter anything itself. `--seed` is passed through to Ollama's `options.seed` where the backend supports it, which makes runs more reproducible, not perfectly reproducible. That's a real constraint of sampling at temperature 0.8 against a quantized local model, not a bug, and it's documented as such in the module docstring rather than implied to be solved.

`reward.py` is the one place filtering happens, and by default it's strict: `--min-score` defaults to `1.0`, so only solutions that pass *every* grader test make it into `filtered_data.json` / `train.jsonl`. That default used to be `0.8` (split across two scripts that silently disagreed with each other, no less), which is exactly what produced the written-number gap documented below: a solution that fails an entire test category but nails everything else still clears 0.8, and 77 of the 83 examples that trained the current `qwen2-codejudge` scored between 0.83 and 0.96, not 1.0. `train.jsonl` is ready to upload to Colab for LoRA fine-tuning with Unsloth and TRL, then export to GGUF and load into Ollama as a new model tag (`qwen2-codejudge`, for example).

### Local / mlx-lm (Apple Silicon, no cloud dependency)

```bash
python trainer/finetune.py --task ambigous_spec --base-model google/gemma-3-4b-it --num-layers 2 --iters 100
```

This reuses the same `filtered_data.json` from `reward.py`, reformats it for `mlx-lm`, and runs LoRA fine-tuning directly on device via `python -m mlx_lm.lora`. Adapters are written to `data/checkpoints/<task>-codejudge/` and are gitignored, since checkpoint weights don't belong in version control any more than the test-case shards do.

### Where things actually stand

This section is a chronological, honest account of three fine-tuning attempts on `ambigous_spec`, because the process of finding out *why* each one failed turned up more real bugs than the final number did.

**v1** (`qwen2-codejudge`, fine-tuned via Colab/Unsloth on 83 examples): 20/24 (83%) on the hand-written grader, every format passes except written numbers (`"fifty"`, `"one hundred"`, `"one thousand"`, `"one million"` all fail). That 83% understated the real gap. Sampling 2,000 cases from the 30-million-case corpus instead of 24 hand-picked ones gave a very different picture:

```
Overall: 1037/2000 = 51.8%

written_number      0/963     0.0%
whitespace         357/357  100.0%
dollar_sign        161/161  100.0%
negative           140/140  100.0%
plain_number       137/137  100.0%
word_dollars       127/127  100.0%
comma_thousands     69/69   100.0%
decimal             46/46   100.0%
```

Every non-written-number category is a clean 100%. `written_number` is a flat 0%, not "mostly fails", all of it. The 24-case grader only has 4 written-number cases (about 17%), so it undersold how much of the real input space that category represents; at true combinatorial scale it's closer to 47% of the corpus, since every integer contributes exactly one written-number phrase versus format variants that only fire on modulo gates.

**Bug #1, found:** the training data behind v1 was mostly near-misses. Of the 83 examples, only 6 (7%) had a perfect grader score; the other 77 scored between 0.83 and 0.96, meaning each failed at least one category, and `trainer/reward.py`'s filtering threshold at the time (`score >= 0.8`) let all of them through as "high quality" anyway. A solution that aces every format except written numbers lands at exactly 20/24 = 0.833, clears 0.8 easily, and teaches the model that skipping written numbers is acceptable. Fixed: `reward.py` now defaults to `--min-score 1.0` (perfect solutions only), and a second, disagreeing filter that used to live inside `generate_training_data.py` (also defaulting to 0.8, and the one that actually produced v1's `train.jsonl`) was removed, so there's exactly one filtering decision in the pipeline now.

**v2** (retrained with bug #1 fixed, 33 perfect examples this time): 20/24 (83%), the identical four written-number tests failing, byte-for-byte the same result as v1.

**Bug #2, found:** the training examples were labeled with the wrong prompt. `generate_training_data.py` queries the teacher model with `hint.md` (a fully-enumerated spec, needed so the teacher reliably writes correct code) but was also storing that same `hint.md` text as the training example's `"input"` field. `evaluator.py` evaluates the student against `problem.md`, a shorter, differently-worded spec. A LoRA fine-tune on a few dozen examples generalizes narrowly enough that training against one prompt and evaluating against a differently-worded one can look identical to no training at all, which is exactly what v2's result was. Fixed: the teacher is still queried with `hint.md`, but the recorded training example now uses `problem.md`, matching what evaluation actually sends. (Grading itself doesn't care which prompt text is attached to an example, `run_solution_against_grader()` only sees the solution code, so decoupling these two was safe.)

**v3** (retrained with bugs #1 and #2 both fixed, 20 perfect examples): the first evaluation attempt returned `Could not import parse_amount` with a suspiciously fast sub-second response.

**Bug #3, found:** `evaluator.py`'s Ollama call included `"stop": ["```"]`, intended to trim a trailing closing markdown fence. Ollama's stop tokens match anywhere in the output stream, not just at the end. `qwen2-codejudge-v3` opens its responses with `` ```python ``, an opening fence, so generation was being cut off after the first three backtick characters, before any real code came out, on every call. `tools/debug_model.py` and `tools/evaluate_at_scale.py` never set this stop token, which is why they worked against the same model when `evaluator.py` didn't. Fixed: the stop token was removed from both `evaluator.py` and `generate_training_data.py` (same bug existed there too, and is a plausible explanation for some of the `score=0.00` attempts seen during teacher sampling in earlier runs). `code_cleaning.py` already strips fences correctly after the full response is captured, so no stop token was needed for that in the first place.

With all three bugs fixed, v3's real result: 20/24 (83%), the same four written-number tests still failing. This time the failure mode was actually inspected directly (`tools/debug_model.py`, full untruncated output), and the generated `parse_amount()` has no written-number handling whatsoever, it strips `$`/`,`/dollar suffixes, handles the negative sign, and calls `float()` on whatever's left. Not broken logic. No attempt at all.

That's a meaningfully different finding than "the pipeline is buggy." All 20 of v3's training examples scored a perfect 24/24, meaning every one of them contained correct, working written-number parsing code, and none of it transferred to the fine-tuned model's default output. With `per_device_train_batch_size=2` and `gradient_accumulation_steps=4` (effective batch size 8) over 20 examples across 3 epochs, that's under 10 total optimizer steps, plausibly too little signal to shift a 1.5B model's greedy decoding toward reliably reproducing a genuinely novel ~20-30 line code pattern that its base pretraining doesn't already give it strong latent competence in. This wasn't tested further (a v4 with meaningfully more perfect training examples, likely requiring 1,000+ teacher attempts instead of 300 to collect enough given the ~5-12% perfect-score hit rate observed across runs, would be the next real experiment, not another pipeline audit).

Net result: three real, verifiable bugs found and fixed in the fine-tuning pipeline (reward-filtering threshold, train/eval prompt mismatch, and a stop-token that was silently truncating generation). The written-number gap itself is still open, and the evidence now points specifically at training-data volume rather than at anything left broken in the harness.

---

## REST API

`api/main.py` is a thin FastAPI wrapper over the same evaluation logic, for driving the harness from something other than a CLI.

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness check |
| `GET /tasks` | List available task names |
| `GET /tasks/{task_name}` | Get a task's prompt |
| `POST /evaluate` | Run a model against a task, end-to-end |
| `GET /results` | List saved result files |
| `GET /results/latest` | Fetch the most recent result |
| `GET /results/{filename}` | Fetch a specific result file |

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload
# docs at http://localhost:8000/docs
```

`GET /results/{filename}` takes a filename straight from the URL, which is exactly the kind of input a path-traversal attempt (`../../etc/passwd`) targets. It's validated against the exact shape the harness generates (`run_<timestamp>.json`, no `/`, no `..`) before touching the filesystem, rather than trusted and passed straight to `Path()`.

`POST /evaluate` triggers real model inference and sandboxed code execution. That makes it an expensive, code-executing endpoint, and there's no authentication or rate limiting in front of it, because it's built to run locally next to Ollama, not to be exposed publicly. If that ever changes, auth and rate limiting need to go in front of this endpoint specifically.

---

## Testing and CI

```bash
python -m pytest
```

52 tests across seven files:

- `test_code_cleaning.py`, including the exact `check_parse_amount()` / undefined-`fifty` regression that motivated the module
- `test_sandbox.py`, checking that CPU, memory, and process limits actually get enforced (the memory test is skipped on macOS, with the platform gap documented rather than hidden)
- `test_shard_store.py`, covering `ShardWriter`, `iter_shards`, reservoir-sampling determinism (same seed produces the same sample), and edge cases like a sample size larger than the corpus or an empty corpus
- `test_generate_test_cases_bulk.py`, including a regression test proving fixed edge-case candidates are validated against the reference solution rather than just assumed correct
- `test_evaluator.py`, `test_graders.py`, and `test_reward.py`, covering the harness logic and each task's reference solution actually passing its own grader

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request to `main`: install dependencies (both `requirements.txt` and `api/requirements.txt`, the latter being what `test_api.py` needs `fastapi` and `httpx` for), run the full pytest suite, then `ruff check . --select=E9,F` to catch syntax errors and undefined names without being opinionated about style.

---

## Project layout

```
codejudge/
├── evaluator.py              # CLI entry point: run a model against a task
├── analyze.py                # Summarize a results/run_*.json file
├── config.py                 # Single source of truth: paths, models, prompts, task loading
├── sandbox.py                # The one place model-generated code actually executes
├── code_cleaning.py          # Fence-stripping + AST-based side-effect removal
├── test_case_store.py        # Sharded JSONL storage + reservoir sampling
├── requirements.txt
├── pytest.ini
│
├── tasks/
│   ├── ambigous_spec/        # parse_amount: format handling
│   │   ├── problem.md        #   underspecified prompt shown to the model under test
│   │   ├── hint.md           #   fully-enumerated version used only for teacher generation
│   │   └── grader.py
│   ├── edge_case_trap/       # most_frequent: silent edge-case bugs
│   │   ├── problem.md
│   │   └── grader.py
│   └── race_condition/       # Counter: real concurrency bug
│       ├── problem.md
│       └── grader.py
│
├── trainer/
│   ├── generate_training_data.py   # Teacher sampling + grading (Colab/Unsloth path)
│   ├── reward.py                   # The single filtering step: raw attempts -> filtered_data.json + train.jsonl
│   └── finetune.py                 # Local LoRA fine-tuning via mlx-lm (Apple Silicon)
│
├── tools/
│   ├── generate_test_cases.py        # Original single-file, ~56k-case generator
│   ├── generate_test_cases_bulk.py   # Sharded, multi-million-case, multiprocessing generator
│   ├── evaluate_at_scale.py          # Sample N cases from a corpus, score by category
│   └── debug_model.py                # Inspect raw vs. cleaned model output for one input
│
├── api/
│   ├── main.py                # FastAPI wrapper
│   └── requirements.txt
│
├── benchmark/
│   ├── run_before.py
│   ├── run_after.py
│   └── compare.py
│
├── tests/                     # pytest suite (52 tests, see above)
├── results/                   # evaluator.py run_<timestamp>.json output (gitignored)
├── data/                      # training runs, checkpoints, corpora (gitignored except manifests)
└── .github/workflows/ci.yml
```

---

## Setup

Requirements:
- Python 3.12+ (developed and CI-tested against 3.12; also runs on 3.10)
- [Ollama](https://ollama.com) running locally for the Ollama-backed models
- An `OPENAI_API_KEY` environment variable if you want to evaluate an OpenAI model

```bash
git clone git@github.com:Shubhoom123/codejudge.git
cd codejudge

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -r api/requirements.txt   # only needed for the API and its tests

ollama serve                          # in a separate terminal, if not already running
ollama pull gemma3:12b                # teacher model
ollama pull qwen2:1.5b                # student model
```

---

## Usage

**Evaluate one model on one task:**
```bash
python evaluator.py --model qwen2:1.5b --task ambigous_spec
python evaluator.py --model gemma3:12b                        # runs every task in tasks/
```

**Summarize the latest run:**
```bash
python analyze.py
```

**Generate a large test-case corpus and benchmark against it:**
```bash
python tools/generate_test_cases_bulk.py --target 2000000 --workers 4
python tools/evaluate_at_scale.py --model qwen2:1.5b --task ambigous_spec --samples 500 --output data/baseline.json
```

**Generate teacher training data, filter it, fine-tune:**
```bash
python trainer/generate_training_data.py --task ambigous_spec --teacher-model gemma3:12b --attempts 100
python trainer/reward.py --task ambigous_spec
python trainer/finetune.py --task ambigous_spec --base-model google/gemma-3-4b-it     # local, mlx-lm
# or upload data/training_runs/ambigous_spec/train.jsonl to Colab for Unsloth/TRL
```

**Debug what a specific model actually outputs:**
```bash
python tools/debug_model.py --model qwen2-codejudge --task ambigous_spec --input "\$44.28"
```

**Run the API:**
```bash
uvicorn api.main:app --reload
```

**Add a new task:**
1. `tasks/your_task/problem.md`, the spec
2. `tasks/your_task/grader.py`, which must print `{"tests": [...], "passed": N, "total": M, "score": N/M}`
3. `python evaluator.py --task your_task --model qwen2:1.5b`

If the task's entry point is a single-argument function returning a plain value, add it to `SCALE_TEST_ENTRY_POINT` in `config.py` to make it eligible for `tools/evaluate_at_scale.py` as well.

---

## Known limitations

Written down here on purpose, instead of left for someone else to discover.

The sandbox doesn't restrict filesystem or network access. It stops runaway CPU, memory, and thread usage, not a model that writes `requests.get(...)` or reads local files. A real trust boundary needs a container or VM around the whole harness.

`RLIMIT_AS`, the memory limit, isn't enforced by macOS the way it is by Linux. The CPU-time, process-count, and wall-clock limits all still hold on macOS; the memory guarantee specifically only holds on Linux, which is why CI runs on Ubuntu.

Test-case deduplication is per worker, not global. Rare exact duplicates across workers are possible in the bulk corpus and aren't corrected for.

The written-number gap is real and, as of this writing, still unresolved: three separate fine-tuning attempts (`qwen2-codejudge`, `-v2`, `-v3`) all land at 20/24 (83%) on the grader with zero written-number handling in the generated code (see "Where things actually stand" above for the full account). Three real bugs in the fine-tuning pipeline were found and fixed along the way (reward-filtering threshold, train/eval prompt mismatch, and an Ollama stop-token silently truncating generation), but fixing all three did not close the gap. The current evidence points at training-data volume, not a remaining pipeline bug: `-v3`'s training set was 20 examples that each scored a perfect 24/24, and none of that written-number-handling code transferred to the fine-tuned model's default output, with an effective batch size of 8 over 20 examples across 3 epochs producing well under 10 total gradient updates. Closing this would mean collecting a meaningfully larger perfect-example set (likely 1,000+ teacher attempts given the ~5-12% perfect-score hit rate observed) before another fine-tuning attempt, not another audit of the harness itself.

The API has no authentication or rate limiting. It's built to run locally next to Ollama. `POST /evaluate` triggers real inference and code execution and shouldn't be exposed publicly as-is.

Reproducibility of teacher sampling is best-effort, not guaranteed. `--seed` is passed to Ollama where the backend supports it, but temperature-0.8 sampling against a quantized model isn't guaranteed to be bit-identical across runs or Ollama versions.

---

## License

MIT. See [LICENSE](LICENSE).
