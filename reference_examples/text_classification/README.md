# Text Classification

Minimal text classification reference experiment for Meta-Harness. The outer loop writes candidate memory systems to `agents/`; the inner loop evaluates them on the datasets in `config.yaml`.

## Quick Start

Install:

```bash
cd reference_examples/text_classification
uv sync
```

Run one evolve iteration:

```bash
uv run python meta_harness.py --iterations 1
```

Run one memory system on one dataset:

```bash
PYTHONPATH=.. uv run python -m text_classification.inner_loop \
  --memory fewshot_all \
  --dataset Symptom2Disease
```

By default this uses the model in `config.yaml` (`openrouter/openai/gpt-oss-120b`). To target another provider or any OpenAI-compatible endpoint, override `--model` and optionally `--api-base`.

Print the benchmark summary:

```bash
uv run python benchmark.py --results
```

## Layout Notes

- `agents/`: the kept baselines plus the write target for generated candidates.
- `.claude/skills/meta-harness/SKILL.md`: main proposer prior used by `meta_harness.py`.

## Runtime And Cost

The release default uses OpenRouter (`openrouter/openai/gpt-oss-120b`). If you want a different provider or your own OpenAI-compatible endpoint, pass `--model` and optionally `--api-base`, or change `config.yaml`. The paper experiments used a local `vllm` deployment of `gpt-oss-120b`, MXFP4 quantized, with `max-model-len=32768`. API-backed runs may differ in quality from that setup and may be better.

## Parallel Sandbox Benchmarking (Marimo)

By default `meta_harness.py` benchmarks candidates sequentially in-process. Set
`META_HARNESS_SANDBOX=1` to instead fan each candidate out into its own W&B
Serverless Sandbox and run them in parallel (see `sandbox_benchmark.py`). Each
sandbox runs `inner_loop` for one candidate and copies its `val.json` /
`memory.json` back into `logs/<run>/...`, so the frontier/leaderboard logic is
unchanged.

In a Marimo online notebook, two cells need updating:

1. Deps install cell — add the sandbox extra so `from wandb.sandbox import Sandbox` works:

```python
import subprocess
subprocess.run(
    ["pip", "install", "-q", "wandb[sandbox]", "openai-harmony",
     "litellm", "datasets", "tenacity>=8", "tqdm", "weave", "pyyaml"],
    check=True,
)
```

2. Run-evolution cell — turn on the sandbox path (the kernel must already have
`WANDB_API_KEY` set, since each sandbox needs it for W&B Inference auth):

```python
import os, sys, runpy

work_dir = "/marimo/meta-harness-main/reference_examples/text_classification"
os.chdir(work_dir)
for p in [work_dir, os.path.dirname(work_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["META_HARNESS_PROPOSER"] = "opencode"
os.environ["META_HARNESS_NO_UV"]   = "1"
os.environ["META_HARNESS_SANDBOX"] = "1"   # parallel sandboxes for benchmarking
os.environ["OPENCODE_BIN"]         = oc     # from the opencode-install cell
# os.environ["WANDB_API_KEY"] must already be set

sys.argv = ["meta_harness", "--iterations", "1"]
runpy.run_path("meta_harness.py", run_name="__main__")
```

The proposer still runs locally in the kernel; only the per-candidate
benchmark is offloaded to the parallel sandboxes.

## Release Notes

- `config.yaml` is the source of truth for datasets, models, and active memory systems.
- The public release includes the MCE paper datasets for this experiment under `data/`, so there is no runtime clone step.
- `inner_loop.py` still uses package-mode imports, so the single-candidate command above keeps `PYTHONPATH=..` when run from this directory.
- `benchmark.py` is the sweep/orchestration layer used by `meta_harness.py`; `inner_loop.py` is the single memory-system evaluator that `benchmark.py` dispatches.
