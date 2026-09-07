import copy
import json
from pathlib import Path

import pytest

from support_troubleshooting_agent.evaluation.calibration import compare_review, prediction_fingerprint
from support_troubleshooting_agent.evaluation.entailment import THRESHOLD, grade_entailment
from support_troubleshooting_agent.evaluation.runner import attach_rubric


def criterion(*hypotheses):
    return {"description": "Required meaning", "hypotheses": list(hypotheses)}


def test_semantic_alternatives_are_or_and_required_criteria_are_and(monkeypatch):
    class Evaluator:
        def score(self, premise, hypothesis):
            return (0.95 if "supported" in hypothesis else 0.02), 8

    monkeypatch.setattr("support_troubleshooting_agent.evaluation.entailment.get_evaluator", lambda path: Evaluator())
    case = {"evaluation_criteria": {
        "diagnosis": [criterion("absent", "supported synonym")],
        "recommendation": [criterion("supported remedy"), criterion("missing recovery check")],
    }}
    result = grade_entailment(case, "Diagnosis", ["Remedy"])
    assert result["diagnosis_pass"]
    assert not result["recommendation_pass"]
    assert result["semantic_usage"] == {"comparisons": 4, "input_tokens": 32, "generated_tokens": 0,
                                        "tokenizer": "cross-encoder/nli-deberta-v3-small"}
    assert result["threshold"] == THRESHOLD


def test_rubric_rejects_stale_references_and_ignores_review_labels(tmp_path):
    case = {"id": "one", "ticket": "Example", "expected_diagnosis": "Cause", "expected_recommendation": "Repair"}
    entry = {**case, "diagnosis": [criterion("Cause")], "recommendation": [criterion("Repair")],
             "manual_passed": True, "predicted_diagnosis": "DO NOT SEND"}
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps({"version": "test", "cases": [entry]}))
    original = {"cases": [case]}
    result = attach_rubric(original, path)
    assert set(result["cases"][0]["evaluation_criteria"]) == {"diagnosis", "recommendation"}
    assert "manual_passed" not in result["cases"][0]
    assert "predicted_diagnosis" not in result["cases"][0]
    assert "evaluation_criteria" not in original["cases"][0]
    with pytest.raises(ValueError, match="stale"):
        attach_rubric({"cases": [{**case, "expected_diagnosis": "Different"}]}, path)


def test_review_detects_dimension_false_negative_even_when_case_matches():
    prediction = {"id": "one", "ticket": "T", "expected_diagnosis": "D", "expected_recommendation": "R",
                  "predicted_diagnosis": "D", "predicted_recommendation": ["Bad action"],
                  "passed": False, "grading": {"diagnosis_pass": False, "recommendation_pass": False}}
    report = {"results": [prediction]}
    review = {"reviewer": "test", "cases": [{"id": "one", "split": "calibration", "diagnosis_pass": True,
               "recommendation_pass": False, "reason": "Diagnosis equivalent; wrong action",
               "prediction_sha256": prediction_fingerprint(prediction)}]}
    audit = compare_review(report, review, "calibration")
    assert audit["matched_cases"] == 0
    assert audit["case_false_negatives"] == 0
    assert audit["dimension_errors"]["diagnosis"]["false_negatives"] == 1
    changed = copy.deepcopy(report)
    changed["results"][0]["predicted_diagnosis"] = "Changed"
    with pytest.raises(ValueError, match="changed"):
        compare_review(changed, review, "calibration")


def test_quotes_must_come_from_candidate_not_reference(monkeypatch):
    from support_troubleshooting_agent.evaluation.judge import Assessment, JudgeError, grade_case

    class HallucinatingJudge:
        def with_structured_output(self, schema, **kwargs):
            assert schema is Assessment
            return self

        def invoke(self, messages, config):
            return {"parsed": {"relationship": "satisfies", "reason": "Claims a fix",
                               "evidence_quotes": ["Renew and deploy the certificate"]}}

    monkeypatch.setenv("EVAL_JUDGE_BACKEND", "llm")
    monkeypatch.setattr("support_troubleshooting_agent.evaluation.judge.get_judge_model", lambda: HallucinatingJudge())
    with pytest.raises(JudgeError):
        grade_case({"expected_diagnosis": "Expired certificate", "expected_recommendation": "Renew and deploy the certificate"},
                   "Expired certificate", ["Monitor certificate dates"], None)


def test_pinned_local_model_distinguishes_monitoring_from_repair():
    from support_troubleshooting_agent.evaluation.entailment import MODEL_FILE, cache_path, get_evaluator

    if not (cache_path() / MODEL_FILE).exists():
        pytest.skip("Optional model cache is not installed; unit tests never download assets")
    evaluator = get_evaluator(str(cache_path().resolve()))
    match, _ = evaluator.score("The certificate has expired.", "The certificate is expired.")
    omission, _ = evaluator.score("The recommendation is to monitor disk usage.",
                                  "The recommendation is to test that application writes succeed.")
    assert match >= THRESHOLD
    assert omission < THRESHOLD
    with pytest.raises(ValueError, match="512"):
        evaluator.score("word " * 600, "A hypothesis")
