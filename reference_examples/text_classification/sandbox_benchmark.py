"""Parallel candidate benchmarking via W&B Serverless Sandboxes.

Instead of benchmarking candidates one-at-a-time in the local kernel
(meta_harness.py sequential loop), fan them out into N concurrent
``wandb.sandbox`` instances — one sandbox per candidate. Each sandbox:

  1. receives an in-memory gzip tarball of the text_classification package
     (mounted read-only), which already contains the freshly-written
     ``agents/<name>.py`` candidate files,
  2. extracts it, pip-installs the inner-loop deps,
  3. runs ``python -m text_classification.inner_loop`` for its candidate,
  4. and the resulting ``val.json`` / ``memory.json`` are copied back into the
     local logs tree at the exact path the rest of the harness expects
     (``run_dir(logs_dir, dataset, memory, model_short, seed)``).

Because results are written back to the canonical on-disk locations, the
downstream frontier/leaderboard logic (``benchmark.load_results`` /
``print_frontier`` / ``meta_harness.update_evolution_summary``) keeps working
unchanged — it just reads the val.json files the sandboxes produced.

Concurrency note: ``wandb.sandbox`` registers a process signal handler when a
sandbox starts, and Python only permits ``signal.signal`` on the **main
thread**. So we must NOT drive sandboxes from worker threads (doing so raises
"signal only works in main thread of the main interpreter"). Instead we create
all sandboxes on the main thread and rely on the SDK's own async ``exec`` —
``sandbox.exec(...)`` returns a ``Process`` immediately and ``.result()`` blocks
— launching each stage on every sandbox before awaiting it (a barrier per
stage). That keeps tar -> pip -> run ordered within a sandbox while overlapping
the slow inner_loop stage across all of them.

This module is import-light at module scope; ``wandb.sandbox`` is imported
lazily inside ``run_candidates`` so importing this file never fails when the
sandbox extra isn't installed (e.g. local runs that never set
META_HARNESS_SANDBOX).
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
from pathlib import Path

import benchmark
from benchmark import (
    DATASETS,
    MODELS,
    SEEDS,
    get_dataset_sizes,
    get_model_short_name,
    run_dir,
)

_PROJECT_DIR = Path(__file__).resolve().parent

# Package directories/files that should never be shipped to a sandbox: they are
# large, machine-specific, or irrelevant to running inner_loop. The candidate
# agents/*.py files ARE included (the tar is built at benchmark time).
_TAR_EXCLUDE_DIRS = {"logs", "results", ".venv", "__pycache__", ".git", ".pytest_cache"}

# Dependencies inner_loop needs inside the sandbox. Mirrors the verified Marimo
# snippet plus what inner_loop imports transitively.
_DEFAULT_DEPS = [
    "litellm",
    "openai-harmony",
    "datasets",
    "tenacity>=8",
    "tqdm",
    "weave",
    "pyyaml",
]


def _make_tarball(src_dir: Path) -> bytes:
    """Return a gzip tarball of ``src_dir`` with arcname ``text_classification``.

    Excludes large/irrelevant directories (see ``_TAR_EXCLUDE_DIRS``) so the
    mounted payload stays small. The package root is named ``text_classification``
    so ``python -m text_classification.inner_loop`` resolves inside the sandbox.
    """

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts  # e.g. ("text_classification", "logs", ...)
        if any(p in _TAR_EXCLUDE_DIRS for p in parts):
            return None
        if info.name.endswith(".tar.gz"):
            return None
        return info

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(src_dir), arcname="text_classification", filter=_filter)
    buf.seek(0)
    return buf.read()


def _build_inner_cmd(
    *,
    candidate: str,
    dataset: str,
    seed: int,
    model: str,
    api_base: str | None,
    wandb_key: str,
    mode: str,
    num_epochs: int,
    temperature: float | None,
) -> str:
    """Build the bash command that runs inner_loop for one candidate in-sandbox."""
    n_train, n_val, n_test = get_dataset_sizes(dataset)
    cmd = (
        f"export WANDB_API_KEY={wandb_key} && export PYTHONPATH=$(pwd) && "
        "python -m text_classification.inner_loop "
        f"--memory agents/{candidate}.py "
        f"--dataset {dataset} "
        f"--seed {seed} "
        f"--model '{model}' "
        f"--mode {mode} "
        f"--num-train {n_train} --num-val {n_val} --num-test {n_test} "
        "--val-output /tmp/val.json "
        "--save-memory /tmp/memory.json "
        "--log /tmp/log.jsonl"
    )
    if api_base:
        cmd += f" --api-base '{api_base}'"
    if mode == "offline" and num_epochs > 1:
        cmd += f" --num-epochs {num_epochs}"
    if temperature is not None:
        cmd += f" --temperature {temperature}"
    return cmd


def _await_all(procs: list) -> list:
    """Block on a list of sandbox exec Process handles, returning their results.

    Each exec was already launched (the SDK starts it eagerly), so awaiting them
    in sequence still lets the underlying commands run concurrently across
    sandboxes — this is a per-stage barrier, not serial execution.
    """
    return [p.result() for p in procs]


def run_candidates(
    candidate_names: list[str],
    logs_dir: Path,
    *,
    deps: list[str] | None = None,
) -> dict[str, bool]:
    """Benchmark each candidate in its own sandbox, concurrently.

    Builds one work item per (candidate, dataset, seed) using the harness config
    (MODELS[0], DATASETS, SEEDS, per-dataset sizes), creates one sandbox per work
    item on the MAIN thread (required for the SDK's signal handler), and drives
    them with the SDK's async exec using a barrier per stage (tar -> pip -> run
    -> read-back). Each val.json/memory.json is written to the canonical local
    path. Returns {label: ok}.
    """
    from wandb.sandbox import Sandbox

    logs_dir = Path(logs_dir)
    if not candidate_names:
        return {}

    if not MODELS:
        raise ValueError("No models configured in config.yaml ('models').")
    model_cfg = MODELS[0]
    model = model_cfg["model"]
    api_base = model_cfg.get("api_base")
    model_short = get_model_short_name(model)

    il = benchmark._CONFIG.get("inner_loop", {})
    mode = il.get("mode", "online")
    num_epochs = int(il.get("num_epochs", 1))
    temperature = il.get("temperature")

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        print(
            "  WARNING: WANDB_API_KEY not set in the kernel; sandbox inner_loop "
            "calls to W&B Inference will fail auth.",
            flush=True,
        )

    resolved_deps = deps if deps is not None else _DEFAULT_DEPS

    work = [
        (candidate, dataset, seed)
        for candidate in candidate_names
        for dataset in DATASETS
        for seed in SEEDS
    ]
    tar_bytes = _make_tarball(_PROJECT_DIR)
    mounted = [{"mount_path": "tc.tar.gz", "file_content": tar_bytes}]

    print(
        f"  sandbox: launching {len(work)} job(s) in parallel "
        f"(model={model_short})",
        flush=True,
    )

    results: dict[str, bool] = {}

    # Create every sandbox on the main thread (the SDK installs a signal handler
    # on start, which only works in the main thread).
    sandboxes = [Sandbox.run(mounted_files=mounted) for _ in work]
    try:
        # Stage 1: extract the package payload in every sandbox.
        _await_all([sb.exec(["tar", "-xzf", "tc.tar.gz"]) for sb in sandboxes])
        # Stage 2: install deps (parallel across sandboxes).
        _await_all([sb.exec(["pip", "install", "-q", *resolved_deps]) for sb in sandboxes])
        # Stage 3: the slow inner_loop, overlapped across all sandboxes.
        cmds = [
            _build_inner_cmd(
                candidate=candidate,
                dataset=dataset,
                seed=seed,
                model=model,
                api_base=api_base,
                wandb_key=wandb_key,
                mode=mode,
                num_epochs=num_epochs,
                temperature=temperature,
            )
            for (candidate, dataset, seed) in work
        ]
        runs = _await_all(
            [sb.exec(["bash", "-c", cmd]) for sb, cmd in zip(sandboxes, cmds)]
        )

        # Stage 4: read each result back and write it to the canonical path.
        for sb, run, (candidate, dataset, seed) in zip(sandboxes, runs, work):
            label = f"{dataset}/{candidate}/{model_short}/seed{seed}"
            val = sb.exec(
                ["bash", "-c", "cat /tmp/val.json 2>/dev/null || true"]
            ).result()
            val_text = (val.stdout or "").strip()
            if run.returncode != 0 or not val_text:
                results[label] = False
                print(f"    FAIL {label} exit={run.returncode}", flush=True)
                tail = (run.stderr or run.stdout or "no output").strip()
                for line in tail.splitlines()[-8:]:
                    print(f"      {line}", flush=True)
                continue
            try:
                json.loads(val_text)
            except json.JSONDecodeError as e:
                results[label] = False
                print(f"    FAIL {label} val.json not valid JSON: {e}", flush=True)
                continue

            rd = run_dir(logs_dir, dataset, candidate, model_short, seed)
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "val.json").write_text(val_text)

            mem = sb.exec(
                ["bash", "-c", "cat /tmp/memory.json 2>/dev/null || true"]
            ).result()
            mem_text = (mem.stdout or "").strip()
            if mem_text:
                (rd / "memory.json").write_text(mem_text)

            results[label] = True
            print(f"    OK   {label}", flush=True)
    finally:
        for sb in sandboxes:
            try:
                sb.stop()
            except Exception:
                pass

    succeeded = sum(1 for ok in results.values() if ok)
    print(f"  sandbox: completed {succeeded}/{len(results)}", flush=True)
    return results


if __name__ == "__main__":
    # Manual smoke test: python sandbox_benchmark.py <name> [<name> ...] <logs_dir>
    if len(sys.argv) < 3:
        print("usage: python sandbox_benchmark.py <candidate> [<candidate> ...] <logs_dir>")
        raise SystemExit(2)
    *names, _logs = sys.argv[1:]
    run_candidates(names, Path(sys.argv[-1]))
