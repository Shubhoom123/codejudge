# CodeJudge

CodeJudge is an evaluation and fine-tuning harness for LLM-generated code. It sends a task spec to a model, runs whatever code comes back inside a resource-limited sandbox, scores it against a grader, and (optionally) uses the results to build a training set for fine-tuning a smaller model to close the gap.

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
python trainer/generate_training_data.py --task ambigous_spec --teacher-model gemma3:12b --attempts 100 --min-score 0.8
python trainer/reward.py --task ambigous_spec
```

`generate_training_data.py` samples the teacher model repeatedly at `temperature=0.8` (diversity on purpose: more varied attempts means more good ones to choose from) against a task's grader, and keeps whichever attempts score at or above `--min-score`. `--seed` is passed through to Ollama's `options.seed` where the backend supports it, which makes runs more reproducible, not perfectly reproducible. That's a real constraint of sampling at temperature 0.8 against a quantized local model, not a bug, and it's documented as such in the module docstring rather than implied to be solved. `reward.py` then filters that raw output down to a `filtered_data.json` / `train.jsonl` pair using a simple reward function (`1.0` for a perfect grader score, `0.5` for 0.8 or above, `0.0` otherwise), ready to upload to Colab for LoRA fine-tuning with Unsloth and TRL, then exported to GGUF and loaded into Ollama as a new model tag (`qwen2-codejudge`, for example).

### Local / mlx-lm (Apple Silicon, no cloud dependency)

```bash
python trainer/finetune.py --task ambigous_spec --base-model google/gemma-3-4b-it --num-layers 2 --iters 100
```

This reuses the same `filtered_data.json` from `reward.py`, reformats it for `mlx-lm`, and runs LoRA fine-tuning directly on device via `python -m mlx_lm.lora`. Adapters are written to `data/checkpoints/<task>-codejudge/` and are gitignored, since checkpoint weights don't belong in version control any more than the test-case shards do.

### Where things actually stand

Measured against the 24-case hand-written grader (not the sharded corpus), for the `qwen2-codejudge` model currently loaded in Ollama:

```
Score: 20/24 (83%)

Passing: dollar signs, comma-thousands, whitespace (including tabs),
         negatives, zero, decimals, leading zeroes, all four
         ValueError cases.

Failing: "fifty", "one hundred", "one thousand", "one million"
         (the written-number category, specifically compound
         and multi-word phrases)
```

That's a current result, not a cherry-picked one. The model closed the format-handling gaps from the original `qwen2:1.5b` baseline, which only handled plain numbers, but the written-number path still needs more or better training examples. The `written_number` category in the bulk-generated corpus (2 million cases, all round-trip validated) exists specifically to drive the next training pass on that gap. It hasn't been run yet.

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
│   ├── reward.py                   # Score-to-reward filtering
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

The written-number gap in `qwen2-codejudge` is real and currently unresolved. Compound written numbers like `"one hundred"` and `"one million"` still fail. The training data to fix it exists (the `written_number` category of the bulk corpus); the next fine-tuning pass using it hasn't been run.

The API has no authentication or rate limiting. It's built to run locally next to Ollama. `POST /evaluate` triggers real inference and code execution and shouldn't be exposed publicly as-is.

Reproducibility of teacher sampling is best-effort, not guaranteed. `--seed` is passed to Ollama where the backend supports it, but temperature-0.8 sampling against a quantized model isn't guaranteed to be bit-identical across runs or Ollama versions.

---

## License

MIT. See [LICENSE](LICENSE).
