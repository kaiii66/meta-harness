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

This module is import-light at module scope; ``wandb.sandbox`` is imported
lazily inside the worker so importing this file never fails when the sandbox
extra isn't installed (e.g. local runs that never set META_HARNESS_SANDBOX).
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _run_one(
    *,
    tar_bytes: bytes,
    candidate: str,
    dataset: str,
    seed: int,
    model: str,
    api_base: str | None,
    logs_dir: Path,
    wandb_key: str,
    deps: list[str],
    mode: str,
    num_epochs: int,
    temperature: float | None,
) -> tuple[str, bool, str]:
    """Run one candidate in its own sandbox; write val.json/memory.json back.

    Returns (label, ok, message). ``ok`` is True only when a non-empty val.json
    was retrieved and written locally.
    """
    from wandb.sandbox import Sandbox

    model_short = get_model_short_name(model)
    label = f"{dataset}/{candidate}/{model_short}/seed{seed}"
    n_train, n_val, n_test = get_dataset_sizes(dataset)

    inner_parts = [
        f"export WANDB_API_KEY={wandb_key}",
        "export PYTHONPATH=$(pwd)",
        (
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
        ),
    ]
    inner_cmd = inner_parts[0] + " && " + inner_parts[1] + " && " + inner_parts[2]
    if api_base:
        inner_cmd += f" --api-base '{api_base}'"
    if mode == "offline" and num_epochs > 1:
        inner_cmd += f" --num-epochs {num_epochs}"
    if temperature is not None:
        inner_cmd += f" --temperature {temperature}"

    mounted_files = [{"mount_path": "tc.tar.gz", "file_content": tar_bytes}]

    try:
        with Sandbox.run(mounted_files=mounted_files) as sandbox:
            sandbox.exec(["tar", "-xzf", "tc.tar.gz"], check=True).result()
            sandbox.exec(["pip", "install", "-q", *deps], check=True).result()

            run = sandbox.exec(["bash", "-c", inner_cmd]).result()
            val = sandbox.exec(
                ["bash", "-c", "cat /tmp/val.json 2>/dev/null || true"]
            ).result()
            mem = sandbox.exec(
                ["bash", "-c", "cat /tmp/memory.json 2>/dev/null || true"]
            ).result()

        val_text = (val.stdout or "").strip()
        if run.returncode != 0 or not val_text:
            tail = (run.stderr or run.stdout or "no output").strip()[-800:]
            return label, False, f"exit={run.returncode}\n{tail}"

        # Validate JSON before writing so we never persist a partial file.
        try:
            json.loads(val_text)
        except json.JSONDecodeError as e:
            return label, False, f"val.json not valid JSON: {e}"

        rd = run_dir(logs_dir, dataset, candidate, model_short, seed)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "val.json").write_text(val_text)

        mem_text = (mem.stdout or "").strip()
        if mem_text:
            (rd / "memory.json").write_text(mem_text)

        return label, True, str(rd / "val.json")
    except Exception:  # noqa: BLE001 — surface any sandbox/transport error per-candidate
        return label, False, traceback.format_exc()[-1200:]


def run_candidates(
    candidate_names: list[str],
    logs_dir: Path,
    *,
    max_parallel: int | None = None,
    deps: list[str] | None = None,
) -> dict[str, bool]:
    """Benchmark each candidate in its own sandbox, concurrently.

    Builds one work item per (candidate, dataset, seed) using the harness
    config (MODELS[0], DATASETS, SEEDS, per-dataset sizes), runs them via a
    thread pool (one sandbox per work item), and writes each val.json/memory.json
    back to the canonical local path. Returns {label: ok}.
    """
    logs_dir = Path(logs_dir)
    if not candidate_names:
        return {}

    if not MODELS:
        raise ValueError("No models configured in config.yaml ('models').")
    model_cfg = MODELS[0]
    model = model_cfg["model"]
    api_base = model_cfg.get("api_base")

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

    tar_bytes = _make_tarball(_PROJECT_DIR)

    work = [
        (candidate, dataset, seed)
        for candidate in candidate_names
        for dataset in DATASETS
        for seed in SEEDS
    ]
    workers = max_parallel if max_parallel is not None else len(work)
    workers = max(1, workers)

    print(
        f"  sandbox: launching {len(work)} job(s) across {workers} parallel "
        f"sandbox(es) (model={get_model_short_name(model)})",
        flush=True,
    )

    results: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_one,
                tar_bytes=tar_bytes,
                candidate=candidate,
                dataset=dataset,
                seed=seed,
                model=model,
                api_base=api_base,
                logs_dir=logs_dir,
                wandb_key=wandb_key,
                deps=resolved_deps,
                mode=mode,
                num_epochs=num_epochs,
                temperature=temperature,
            ): (candidate, dataset, seed)
            for (candidate, dataset, seed) in work
        }
        for fut in as_completed(futures):
            label, ok, message = fut.result()
            results[label] = ok
            if ok:
                print(f"    OK   {label}", flush=True)
            else:
                print(f"    FAIL {label}", flush=True)
                for line in message.splitlines()[-8:]:
                    print(f"      {line}", flush=True)

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
