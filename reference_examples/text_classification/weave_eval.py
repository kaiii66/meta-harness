"""Weave Evaluation layer for the meta-harness text-classification benchmark.

Wraps each candidate harness's post-training eval pass in a weave.Evaluation
(Dataset + scorers) so candidates become first-class, comparable Eval objects.
Rides the existing tracing plumbing in weave_tracing.py (gated by WANDB_PROJECT).
"""

from __future__ import annotations

from typing import Any

from .data import get_evaluator


def prediction_from_output(output: Any) -> str:
    """Pull the answer string out of a candidate's predict() output.

    predict() returns (answer, metadata); Weave may serialize the tuple to a
    list. A plain string is passed through unchanged.
    """
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def context_chars_from_output(output: Any) -> int:
    """Per-prediction injected-context size, recorded into predict() metadata
    by the model wrapper. Defaults to 0 when absent."""
    if (
        isinstance(output, (tuple, list))
        and len(output) > 1
        and isinstance(output[1], dict)
    ):
        return int(output[1].get("context_chars", 0) or 0)
    return 0


def build_weave_dataset(examples: list[dict[str, Any]], *, name: str) -> Any:
    """Project val/test examples to a weave.Dataset of exactly {input, target}.

    All candidates are scored against the same published dataset (so they're
    comparable on the leaderboard), and bulky fields are dropped to keep the
    logged payload small.
    """
    import weave

    rows = [{"input": ex["input"], "target": ex["target"]} for ex in examples]
    return weave.Dataset(name=name, rows=rows)


def score_correctness(task: str, prediction: str, target: str) -> dict[str, bool]:
    """Binary correctness via the harness's own task evaluator.

    Mirrors inner_loop._unpack_eval_result: get_evaluator(task) returns either a
    bool or {"was_correct": bool, "metrics": {...}}; normalize to {"correct": bool}.
    """
    raw = get_evaluator(task)(prediction, target)
    was_correct = raw["was_correct"] if isinstance(raw, dict) else raw
    return {"correct": bool(was_correct)}


def summarize_result(summary: dict[str, Any], total: int) -> dict[str, Any]:
    """Extract {accuracy, correct, total, context_chars_mean} from the dict
    weave.Evaluation.evaluate() returns. Shape (confirmed empirically):
        correctness -> correct -> {true_count, true_fraction}
        cost        -> context_chars -> {mean}
    """
    block = summary.get("correctness", {}).get("correct", {})
    cost_mean = summary.get("cost", {}).get("context_chars", {}).get("mean", 0.0)
    return {
        "accuracy": float(block.get("true_fraction", 0.0)),
        "correct": int(block.get("true_count", 0)),
        "total": total,
        "context_chars_mean": float(cost_mean),
    }


# ---------------------------------------------------------------------------
# Weave glue (imported lazily so the pure helpers above work without weave)
# ---------------------------------------------------------------------------


def make_correctness_scorer(task: str) -> Any:
    """Binary per-row scorer Weave calls with (target, output)."""
    import weave

    @weave.op(name="correctness")
    def correctness(target: str, output: Any) -> dict[str, bool]:
        return score_correctness(task, prediction_from_output(output), target)

    return correctness


def make_cost_scorer() -> Any:
    """Per-row context-cost scorer; mean is the Pareto second axis."""
    import weave

    @weave.op(name="cost")
    def cost(output: Any) -> dict[str, int]:
        return {"context_chars": context_chars_from_output(output)}

    return cost


def make_model(memory: Any) -> Any:
    """Adapt a trained MemorySystem to a Weave eval model: predict(input) calls
    memory.predict and records the harness's per-candidate injected-context size
    (memory_context_chars) into the output metadata for the cost scorer."""
    import weave

    @weave.op(name="predict")
    def predict(input: str):
        answer, meta = memory.predict(input)
        meta = dict(meta or {})
        try:
            meta["context_chars"] = int(memory.get_context_length())
        except Exception:
            meta.setdefault("context_chars", 0)
        return answer, meta

    return predict


def evaluate_candidate(
    memory: Any,
    examples: list[dict[str, Any]],
    *,
    task: str,
    name: str,
) -> dict[str, Any]:
    """Score one trained candidate harness as a weave.Evaluation over `examples`.

    Returns the same {accuracy, correct, total} the harness's val.json carries
    (plus context_chars_mean). Logging to Weave happens only when WANDB_PROJECT
    is set (via init_weave); otherwise the evaluation still runs and scores
    locally with no backend.
    """
    import asyncio

    import weave

    from .weave_tracing import init_weave

    init_weave()  # no-op unless Weave is enabled
    dataset = build_weave_dataset(examples, name=name)
    evaluation = weave.Evaluation(
        name=name,
        dataset=dataset,
        scorers=[make_correctness_scorer(task), make_cost_scorer()],
    )
    summary = asyncio.run(evaluation.evaluate(make_model(memory)))
    return summarize_result(summary, total=len(examples))


# ---------------------------------------------------------------------------
# In-loop path: log the harness's own predictions as a first-class Evaluation
# via the imperative EvaluationLogger (single pass, val.json preserved).
# ---------------------------------------------------------------------------


def prediction_log_payload(pred: dict[str, Any]) -> dict[str, Any]:
    """Map an evaluate_memory prediction dict to an EvaluationLogger payload of
    inputs + scores only.

    Deliberately excludes prompt_text/prompt_len: those are bulky and have
    errored the Weave UI, and they are not needed for the leaderboard.
    """
    return {
        "inputs": {"input": pred.get("input", "")},
        "output": pred.get("prediction", ""),
        "scores": {
            "correctness": bool(pred.get("was_correct", False)),
            "context_chars": int(pred.get("context_len", 0) or 0),
        },
    }


def log_weave_evaluation(
    *,
    model_name: str,
    dataset_name: str,
    predictions: list[dict[str, Any]],
    eval_name: str | None = None,
) -> str | None:
    """Log already-computed predictions+scores as a first-class weave Evaluation
    (one per candidate x split) using the imperative EvaluationLogger.

    No-op (returns None) unless Weave is enabled, so the harness path is
    unaffected when WANDB_PROJECT is unset. Logs inputs + scores only.
    """
    from .weave_tracing import weave_enabled

    if not weave_enabled():
        return None

    from weave import EvaluationLogger

    ev = EvaluationLogger(
        name=eval_name or dataset_name,
        model=model_name,
        dataset=dataset_name,
    )
    for pred in predictions:
        payload = prediction_log_payload(pred)
        score_logger = ev.log_prediction(
            inputs=payload["inputs"], output=payload["output"]
        )
        for scorer, score in payload["scores"].items():
            score_logger.log_score(scorer=scorer, score=score)
        score_logger.finish()
    ev.log_summary()

    url = getattr(ev, "ui_url", None)
    try:
        return url() if callable(url) else url
    except Exception:
        return None
