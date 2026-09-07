import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from support_troubleshooting_agent.evaluation.runner import TokenUsage, grade_case, load_dataset, run_evaluation
from support_troubleshooting_agent.graph import builder
from support_troubleshooting_agent.evaluation.judge import Assessment, JudgeError


@pytest.fixture(autouse=True)
def use_llm_backend_for_callback_tests(monkeypatch):
    monkeypatch.setenv("EVAL_JUDGE_BACKEND", "llm")


def case(identifier="case-1", ticket="Database timeout"):
    return {"id": identifier, "ticket": ticket, "expected_diagnosis": "Database latency",
            "expected_recommendation": "Restore connectivity and validate"}


def passing_grade(*args):
    return {"diagnosis_pass": True, "recommendation_pass": True, "reason": "Matches references"}


def test_real_graph_callbacks_and_review_boundary(monkeypatch, tmp_path):
    model = FakeMessagesListChatModel(responses=[AIMessage(
        content="diagnosis", usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})])

    def agent(module, function, state):
        assert "expected_diagnosis" not in state
        if function == "ticket_agent":
            return {"ticket_summary": "confirmed root cause"}
        if function == "diagnosis_agent":
            model.invoke("test nested callbacks")
            return {"root_cause": {"primary_cause": "Database latency"}}
        if function == "recommendation_agent":
            return {"recommendations": {"recommended_actions": ["Restore connectivity"]}}
        return {}

    monkeypatch.setattr(builder, "_invoke_agent", agent)
    output = tmp_path / "results.json"
    golden = Path(__file__).resolve().parents[2] / "data/evaluation/golden_dataset.json"
    report = run_evaluation(load_dataset(golden), output,
                            workflow=builder.build_workflow(), grader=passing_grade)
    assert report["summary"]["passed"] == 30
    assert report["results"][27]["ticket"] == ""
    assert json.loads(output.read_text()) == report
    for result in report["results"]:
        assert result["awaiting_human_review"]
        assert result["token_usage"]["total_tokens"] == 14
        assert result["token_usage"]["complete"]
        assert result["latency_seconds"] >= 0


def test_errors_continue_and_checkpoint_previous_result(tmp_path):
    output = tmp_path / "results.json"

    def invoke(state, config):
        if state["ticket"] == "crash":
            raise RuntimeError("broken workflow")
        assert len(json.loads(output.read_text())["results"]) >= 1
        return {"root_cause": {"primary_cause": "DB"}, "recommendations": ["Repair"]}

    calls = []

    def grade(*args):
        calls.append(args)
        if len(calls) == 1:
            raise ValueError("invalid judge JSON")
        return {"diagnosis_pass": True, "recommendation_pass": False, "reason": "Missing validation"}

    report = run_evaluation({"cases": [case("1", "crash"), case("2"), case("3")]}, output,
                            workflow=SimpleNamespace(invoke=invoke), grader=grade)
    assert [r["status"] for r in report["results"]] == ["error", "error", "fail"]
    assert report["complete"]
    assert report["summary"]["failed"] == 1
    assert report["summary"]["errors"] == 2
    assert report["summary"]["judge_errors"] == 1
    assert report["summary"]["workflow_errors"] == 1
    assert report["summary"]["grading_coverage"] == 1 / 3
    assert report["summary"]["pass_rate"] == 0
    assert report["results"][0]["passed"] is None
    assert report["results"][0]["token_usage"]["total_tokens"] is None


def test_usage_fallback_deduplication_and_partial_reporting():
    usage = TokenUsage()
    response = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]],
                         llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    run_id = uuid4()
    usage.on_llm_end(response, run_id=run_id)
    usage.on_llm_end(response, run_id=run_id)
    usage.on_llm_error(RuntimeError(), run_id=uuid4())
    assert usage.snapshot() == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10,
                                "model_calls": 2, "calls_with_usage": 1, "complete": False}


def test_dataset_validation(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"cases": [case(), case()]}))
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(path)
    path.write_text(json.dumps({"cases": [case(ticket="")]}))
    assert load_dataset(path)["cases"][0]["ticket"] == ""
    golden = Path(__file__).resolve().parents[2] / "data/evaluation/golden_dataset.json"
    assert len(load_dataset(golden)["cases"]) == 30


def test_missing_predictions_fail_without_judging(tmp_path):
    def unexpected_grade(*args):
        raise AssertionError("Should not grade missing output")

    report = run_evaluation({"cases": [case()]}, tmp_path / "results.json",
                            workflow=SimpleNamespace(invoke=lambda *args, **kwargs: {}),
                            grader=unexpected_grade)
    result = report["results"][0]
    assert result["status"] == "error"
    assert result["judge_latency_seconds"] == 0


def assessment(passes=True):
    return {"evidence_quotes": ["<auto>"] if passes else [], "relationship": "satisfies" if passes else "does_not_satisfy", "reason": "Covered by answer" if passes else "Essential action missing"}


class FakeJudge:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.messages = []

    def with_structured_output(self, schema, **kwargs):
        assert schema is Assessment
        assert kwargs == {"method": "json_schema", "include_raw": True}
        return self

    def invoke(self, messages, config):
        self.calls += 1
        self.messages.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        if response.get("evidence_quotes") == ["<auto>"]:
            response = {**response, "evidence_quotes": [next(json.loads(message.content)["answer_to_check"] for message in reversed(messages) if message.content.startswith("{") and "answer_to_check" in json.loads(message.content))]}
        return {"parsed": response, "parsing_error": None}


def test_judge_records_structured_verdict(monkeypatch):
    model = FakeJudge([assessment(), assessment()])
    monkeypatch.setattr("support_troubleshooting_agent.evaluation.judge.get_chat_model", lambda: model)
    result = grade_case(case(), "DB latency", {"validation_steps": ["verify orders"]}, TokenUsage())
    assert result["diagnosis_pass"] and result["recommendation_pass"]
    assert result["assessment"]["diagnosis"]["criteria"][0]["evidence_quotes"] == ["DB latency"]
    assert result["assessment"]["recommendation"]["criteria"][0]["quotes_verified"]
    assert result["attempts"] == [{"dimension": dimension, "criterion_index": 0, "attempt": 1, "status": "valid"} for dimension in ("diagnosis", "recommendation")]


def test_judge_retries_invalid_assessment_but_not_negative_verdict(monkeypatch):
    model = FakeJudge([{}, ValueError("invalid JSON"), assessment(), assessment(False)])
    monkeypatch.setattr("support_troubleshooting_agent.evaluation.judge.get_chat_model", lambda: model)
    result = grade_case(case(), "DB", ["monitor"], TokenUsage())
    assert model.calls == 4
    assert any("Database latency" in message.content and "answer_to_check" in message.content for message in model.messages[2])
    assert result["diagnosis_pass"] and not result["recommendation_pass"]
    assert [item["status"] for item in result["attempts"]] == ["error", "error", "valid", "valid"]
    model = FakeJudge([assessment(False), assessment()])
    monkeypatch.setattr("support_troubleshooting_agent.evaluation.judge.get_chat_model", lambda: model)
    assert not grade_case(case(), "Wrong", ["monitor"], TokenUsage())["diagnosis_pass"]
    assert model.calls == 2


def test_judge_exhaustion_is_error(monkeypatch):
    model = FakeJudge([{}, {}, {}])
    monkeypatch.setattr("support_troubleshooting_agent.evaluation.judge.get_chat_model", lambda: model)
    with pytest.raises(JudgeError) as caught:
        grade_case(case(), "DB", ["repair"], TokenUsage())
    assert len(caught.value.attempts) == 3


def test_empty_recommendation_is_deterministic_error(tmp_path):
    report = run_evaluation({"cases": [case()]}, tmp_path / "result.json",
        workflow=SimpleNamespace(invoke=lambda *args, **kwargs: {
            "root_cause": {"primary_cause": "DB"}, "recommendations": {"recommended_actions": [" "]}}),
        grader=lambda *args: pytest.fail("Empty recommendations must not reach the judge"))
    assert report["results"][0]["error_stage"] == "workflow"
    assert report["summary"]["pass_rate"] is None


def test_regrade_preserves_predictions_and_metrics_without_graph(monkeypatch, tmp_path):
    previous = {"provider": "OldProvider", "model": "OldModel", "results": [{
        **case(), "predicted_diagnosis": "DB latency", "predicted_recommendation": ["repair"],
        "status": "error", "error": "Judge error: JSONDecodeError", "latency_seconds": 12.3,
        "token_usage": {"total_tokens": 123}, "awaiting_human_review": True}]}
    monkeypatch.setattr("support_troubleshooting_agent.evaluation.runner.build_workflow",
                        lambda: pytest.fail("Regrade must not construct a graph"))
    result = run_evaluation({"cases": [case()]}, tmp_path / "result.json",
                            previous_report=previous, grader=passing_grade)
    assert result["mode"] == "regrade"
    assert result["provider"] == "OldProvider"
    assert result["results"][0]["latency_seconds"] == 12.3
    assert result["results"][0]["token_usage"] == {"total_tokens": 123}
    assert result["summary"]["passed"] == 1
    with pytest.raises(ValueError, match="inputs differ"):
        run_evaluation({"cases": [case(ticket="Changed")]}, tmp_path / "other.json", previous_report=previous)


def test_judge_override_does_not_mutate_agent_model(monkeypatch):
    from langchain_ollama import ChatOllama
    from support_troubleshooting_agent.evaluation.judge import get_judge_model, get_judge_configuration

    agent_model = ChatOllama(model="llama3.2")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "gemma3:4b")
    monkeypatch.setattr("support_troubleshooting_agent.evaluation.judge.get_chat_model", lambda: agent_model)
    monkeypatch.setattr("support_troubleshooting_agent.evaluation.judge.get_model_configuration",
                        lambda: ("Ollama", "llama3.2"))
    judge_model = get_judge_model()
    assert judge_model.model == "gemma3:4b"
    assert agent_model.model == "llama3.2"
    assert get_judge_configuration() == ("Ollama", "gemma3:4b")


@pytest.mark.parametrize("invalid", [
    {"relationship": True, "reason": "Equivalent"},
    {"relationship": "satisfies", "reason": " "},
    {"relationship": "satisfies", "reason": "Equivalent", "unexpected": True},
])
def test_judge_rejects_invalid_schema_values(monkeypatch, invalid):
    invalid = {**assessment(), **invalid}
    model = FakeJudge([invalid, invalid, invalid])
    monkeypatch.setattr("support_troubleshooting_agent.evaluation.judge.get_judge_model", lambda: model)
    with pytest.raises(JudgeError):
        grade_case(case(), "DB", ["repair"], TokenUsage())
