"""Standalone Weave-eval teaching demo (FC workshop, hour 1).

Runs ONE declarative weave.Evaluation on the baseline (untrained) harness so
participants see the Eval object first-hand — Dataset + scorers + leaderboard —
before the auto-research loop starts emitting one per candidate automatically.

It reuses the exact Dataset builder + scorers the in-loop path uses
(text_classification.weave_eval), so "what you ran by hand" and "what the loop
runs for you" are the same machinery.

Usage (needs WANDB_PROJECT set + a solver model in config.yaml):
    cd reference_examples
    WANDB_PROJECT=<entity>/<project> \
        uv run --project text_classification python -m text_classification.weave_eval_demo \
        --dataset Symptom2Disease --num-val 50
"""

from __future__ import annotations

import argparse

from .data import load_dataset_splits_3way
from .inner_loop import load_config, load_memory_system
from .llm import LLM
from .weave_eval import evaluate_candidate


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Weave-eval teaching demo")
    parser.add_argument("--dataset", default="Symptom2Disease")
    parser.add_argument("--memory", default="agents/no_memory.py")
    parser.add_argument("--num-val", type=int, default=50)
    parser.add_argument("--model", default=None, help="Solver model (overrides config)")
    parser.add_argument("--seed", type=int, default=cfg["inner_loop"]["seed"])
    args = parser.parse_args()

    # A small val split is enough to see the Eval object; no train/test needed.
    _, val_examples, _, _ = load_dataset_splits_3way(
        args.dataset,
        num_train=0,
        num_val=args.num_val,
        num_test=0,
        shuffle_seed=args.seed,
    )

    model = args.model or cfg["models"][0]["model"]
    api_base = (
        None
        if model.startswith(("gemini/", "openrouter/"))
        else cfg["models"][0].get("api_base")
    )
    llm = LLM(model=model, api_base=api_base)
    memory = load_memory_system(args.memory, llm)  # baseline: no training

    result = evaluate_candidate(
        memory, val_examples, task=args.dataset, name=f"{args.dataset}-val-demo"
    )
    print("Baseline weave.Evaluation result:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
