from __future__ import annotations

import unittest

from text_classification.weave_eval import (
    build_weave_dataset,
    context_chars_from_output,
    evaluate_candidate,
    log_weave_evaluation,
    prediction_from_output,
    prediction_log_payload,
    score_correctness,
    summarize_result,
)


class ScoreCorrectnessTests(unittest.TestCase):
    """The correctness scorer wraps the harness's own get_evaluator(task)
    and normalizes both its bool and {"was_correct": ...} return shapes
    into a binary {"correct": bool}."""

    def test_symptom_tagged_answer_is_correct(self) -> None:
        self.assertEqual(
            score_correctness(
                "Symptom2Disease", "[DIAGNOSIS]Diabetes[/DIAGNOSIS]", "diabetes"
            ),
            {"correct": True},
        )

    def test_symptom_wrong_answer_is_incorrect(self) -> None:
        self.assertEqual(
            score_correctness(
                "Symptom2Disease", "[DIAGNOSIS]flu[/DIAGNOSIS]", "diabetes"
            ),
            {"correct": False},
        )

    def test_dict_evaluator_unpacks_was_correct(self) -> None:
        # USPTO's evaluator returns {"was_correct": ..., "metrics": {...}}.
        self.assertEqual(
            score_correctness("USPTO", '{"final_answer":"A.B"}', "b.a"),
            {"correct": True},
        )


class OutputExtractionTests(unittest.TestCase):
    """A candidate harness's predict() returns (answer, metadata). Weave hands
    that whole output to scorers, so we need to pull the answer and the
    per-prediction context size back out (a list, since Weave may serialize the
    tuple)."""

    def test_prediction_from_tuple(self) -> None:
        self.assertEqual(prediction_from_output(("diabetes", {"a": 1})), "diabetes")

    def test_prediction_from_list(self) -> None:
        self.assertEqual(prediction_from_output(["diabetes", {}]), "diabetes")

    def test_prediction_from_plain_string(self) -> None:
        self.assertEqual(prediction_from_output("diabetes"), "diabetes")

    def test_context_chars_reads_metadata(self) -> None:
        self.assertEqual(
            context_chars_from_output(("ans", {"context_chars": 42})), 42
        )

    def test_context_chars_defaults_to_zero_when_absent(self) -> None:
        self.assertEqual(context_chars_from_output(("ans", {})), 0)
        self.assertEqual(context_chars_from_output("ans"), 0)


class BuildDatasetTests(unittest.TestCase):
    """The val/test examples become a weave.Dataset of exactly {input, target}
    rows so every candidate is scored against the same comparable dataset, and
    bulky fields (raw_question, context) stay out of the logged payload."""

    def test_rows_preserve_input_and_target(self) -> None:
        examples = [
            {"input": "patient has X", "target": "diabetes"},
            {"input": "patient has Y", "target": "flu"},
        ]
        ds = build_weave_dataset(examples, name="symptom2disease-val")
        rows = list(ds.rows)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["input"], "patient has X")
        self.assertEqual(rows[0]["target"], "diabetes")
        self.assertEqual(rows[1]["target"], "flu")

    def test_drops_bulky_extra_fields(self) -> None:
        examples = [
            {"input": "x", "target": "diabetes", "raw_question": "q", "context": "c"}
        ]
        ds = build_weave_dataset(examples, name="val")
        row = list(ds.rows)[0]
        self.assertNotIn("raw_question", row)
        self.assertNotIn("context", row)


class SummarizeResultTests(unittest.TestCase):
    """Extract {accuracy, correct, total, context_chars_mean} from the dict that
    weave.Evaluation.evaluate() actually returns (shape confirmed empirically)."""

    def test_extracts_from_weave_summary(self) -> None:
        summary = {
            "correctness": {"correct": {"true_count": 2, "true_fraction": 2 / 3}},
            "cost": {"context_chars": {"mean": 7.0}},
            "model_latency": {"mean": 0.001},
        }
        result = summarize_result(summary, total=3)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["total"], 3)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["context_chars_mean"], 7.0)


class _FakeMemory:
    """Stands in for a trained MemorySystem: maps inputs to answers and reports
    a fixed injected-context size."""

    def __init__(self, mapping: dict[str, str], context_chars: int) -> None:
        self._mapping = mapping
        self._context_chars = context_chars

    def predict(self, input: str):
        return self._mapping.get(input, "unknown"), {}

    def get_context_length(self) -> int:
        return self._context_chars

    def get_last_prompt_info(self) -> dict:
        return {"prompt_len": 0, "prompt_hash": None, "prompt_text": ""}


class EvaluateCandidateTests(unittest.TestCase):
    """End-to-end: a real weave.Evaluation over a fake trained memory returns the
    same {accuracy, correct, total} the harness's val.json carries. Runs offline
    (no WANDB_PROJECT -> no backend)."""

    def test_returns_accuracy_correct_total(self) -> None:
        examples = [
            {"input": "q1", "target": "diabetes"},
            {"input": "q2", "target": "flu"},
            {"input": "q3", "target": "asthma"},
        ]
        memory = _FakeMemory(
            {"q1": "diabetes", "q2": "flu", "q3": "wrong-answer"}, context_chars=11
        )
        result = evaluate_candidate(
            memory, examples, task="Symptom2Disease", name="test-candidate-val"
        )
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["total"], 3)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["context_chars_mean"], 11.0)


class PredictionLogPayloadTests(unittest.TestCase):
    """The in-loop logger maps an evaluate_memory prediction dict to an
    EvaluationLogger payload of inputs + scores only — never the full prompt
    text (the bulky payload that errors the Weave UI)."""

    def _pred(self) -> dict:
        return {
            "input": "patient has X",
            "prediction": "diabetes",
            "target": "diabetes",
            "was_correct": True,
            "prompt_len": 4096,
            "context_len": 1200,
            "prompt_text": "HUGE PROMPT ..." * 100,
        }

    def test_payload_carries_input_output_and_scores(self) -> None:
        payload = prediction_log_payload(self._pred())
        self.assertEqual(payload["inputs"], {"input": "patient has X"})
        self.assertEqual(payload["output"], "diabetes")
        self.assertEqual(payload["scores"]["correctness"], True)
        self.assertEqual(payload["scores"]["context_chars"], 1200)

    def test_payload_never_includes_prompt_text(self) -> None:
        payload = prediction_log_payload(self._pred())
        flat = repr(payload)
        self.assertNotIn("HUGE PROMPT", flat)
        self.assertNotIn("prompt_text", flat)
        self.assertNotIn("prompt_len", flat)


class LogWeaveEvaluationDisabledTests(unittest.TestCase):
    """When Weave is disabled (no WANDB_PROJECT) the in-loop logger is a no-op:
    returns None and never raises, so the harness path is unaffected."""

    def test_noop_when_disabled(self) -> None:
        preds = [
            {"input": "a", "prediction": "x", "was_correct": True, "context_len": 5}
        ]
        result = log_weave_evaluation(
            model_name="cand", dataset_name="Symptom2Disease-val", predictions=preds
        )
        self.assertIsNone(result)


class EvaluateMemoryInputTests(unittest.TestCase):
    """evaluate_memory must surface the example input in each prediction so the
    in-loop Evaluation logger can record it."""

    def test_predictions_include_input_text(self) -> None:
        from text_classification.inner_loop import evaluate_memory

        examples = [{"input": "symptoms A", "target": "diabetes"}]
        memory = _FakeMemory({"symptoms A": "diabetes"}, context_chars=5)

        def check(pred, target, **kw):
            return pred == target

        result = evaluate_memory(memory, examples, check, max_workers=1)
        self.assertEqual(result["predictions"][0]["input"], "symptoms A")


if __name__ == "__main__":
    unittest.main()
