"""Run golden tickets through the real graph and grade their reference criteria."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from support_troubleshooting_agent.graph.builder import build_workflow
from support_troubleshooting_agent.models.llm_factory import get_model_configuration
from .judge import JUDGE_VERSION, get_judge_configuration, grade_case


class TokenUsage(BaseCallbackHandler):
    """Count reported chat tokens, without mistaking missing telemetry for zero."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._seen: set[Any] = set()
        self.calls = self.reported_calls = 0
        self.input_tokens = self.output_tokens = self.total_tokens = 0

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        with self._lock:
            if run_id in self._seen:
                return
            self._seen.add(run_id)
            self.calls += 1
            usage = None
            if response.generations and response.generations[0]:
                message = getattr(response.generations[0][0], "message", None)
                usage = getattr(message, "usage_metadata", None)
            if not usage:
                raw = (response.llm_output or {}).get("token_usage", {})
                if "prompt_tokens" in raw and "completion_tokens" in raw:
                    usage = {"input_tokens": raw["prompt_tokens"],
                             "output_tokens": raw["completion_tokens"],
                             "total_tokens": raw.get("total_tokens", raw["prompt_tokens"] + raw["completion_tokens"])}
            if usage:
                self.reported_calls += 1
                self.input_tokens += usage["input_tokens"]
                self.output_tokens += usage["output_tokens"]
                self.total_tokens += usage["total_tokens"]

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        with self._lock:
            if run_id not in self._seen:
                self._seen.add(run_id)
                self.calls += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens if self.reported_calls else None,
            "output_tokens": self.output_tokens if self.reported_calls else None,
            "total_tokens": self.total_tokens if self.reported_calls else None,
            "model_calls": self.calls,
            "calls_with_usage": self.reported_calls,
            "complete": self.calls > 0 and self.calls == self.reported_calls,
        }


def load_dataset(path: Path) -> dict[str, Any]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or not isinstance(dataset.get("cases"), list) or not dataset["cases"]:
        raise ValueError("Dataset must contain a nonempty cases array.")
    seen = set()
    for case in dataset["cases"]:
        for key in ("id", "ticket", "expected_diagnosis", "expected_recommendation"):
            if not isinstance(case, dict) or not isinstance(case.get(key), str):
                raise ValueError(f"Every case must have a string {key}.")
            if key != "ticket" and not case[key].strip():
                raise ValueError(f"Case {key} cannot be blank.")
        if case["id"] in seen:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        seen.add(case["id"])
    return dataset


def has_recommendation(value: Any) -> bool:
    """Reject empty structures without asking an LLM to evaluate their meaning."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(has_recommendation(item) for item in value.values())
    if isinstance(value, list):
        return any(has_recommendation(item) for item in value)
    return False


def attach_rubric(dataset: dict[str, Any], path: Path) -> dict[str, Any]:
    """Bind reviewed criteria to reference text, never to predictions or verdicts."""
    raw = path.read_bytes()
    rubric = json.loads(raw)
    entries = {item["id"]: item for item in rubric["cases"]}
    if len(entries) != len(rubric["cases"]):
        raise ValueError("Duplicate rubric case IDs.")
    cases = []
    for case in dataset["cases"]:
        entry = entries.get(case["id"], {})
        if any(entry.get(key) != case[key] for key in ("expected_diagnosis", "expected_recommendation")):
            raise ValueError(f"Missing or stale rubric for {case['id']}.")
        for dimension in ("diagnosis", "recommendation"):
            criteria = entry.get(dimension)
            if not isinstance(criteria, list) or not criteria or any(
                not isinstance(item, dict) or not isinstance(item.get("description"), str) or not item["description"].strip()
                or not isinstance(item.get("hypotheses"), list) or not item["hypotheses"]
                or any(not isinstance(hypothesis, str) or not hypothesis.strip() for hypothesis in item["hypotheses"])
                for item in criteria
            ):
                raise ValueError(f"Invalid {dimension} criteria for {case['id']}.")
        cases.append({**case, "evaluation_criteria": {key: entry[key] for key in ("diagnosis", "recommendation")}})
    return {**dataset, "cases": cases, "rubric_version": rubric["version"],
            "rubric_sha256": hashlib.sha256(raw).hexdigest()}


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(item["status"] == "pass" for item in results)
    failed = sum(item["status"] == "fail" for item in results)
    graded = passed + failed
    return {
        "total": len(results), "graded": graded, "passed": passed, "failed": failed,
        "errors": len(results) - graded,
        "judge_errors": sum(item.get("error_stage") == "judge" for item in results),
        "workflow_errors": sum(item.get("error_stage") == "workflow" for item in results),
        "pass_rate": passed / graded if graded else None,
        "grading_coverage": graded / len(results) if results else 0.0,
        "diagnosis_passed": sum(item.get("grading", {}).get("diagnosis_pass", False) for item in results),
        "recommendation_passed": sum(item.get("grading", {}).get("recommendation_pass", False) for item in results),
    }


def run_evaluation(dataset: dict[str, Any], output: Path, *, workflow: Any = None,
                   grader: Any = grade_case, previous_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate sequentially, checkpointing the JSON report after each case."""
    previous = {}
    if previous_report is not None:
        previous = {item["id"]: item for item in previous_report["results"]}
        if len(previous) != len(previous_report["results"]) or set(previous) != {case["id"] for case in dataset["cases"]}:
            raise ValueError("Regrade report must contain each dataset case exactly once.")
        for case in dataset["cases"]:
            if any(previous[case["id"]].get(key) != case[key] for key in
                   ("ticket", "expected_diagnosis", "expected_recommendation")):
                raise ValueError(f"Regrade inputs differ for {case['id']}.")
    else:
        workflow = workflow if workflow is not None else build_workflow()
    provider, model = get_model_configuration()
    judge_provider, judge_model = get_judge_configuration()
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {key: value for key, value in dataset.items() if key != "cases"},
        "provider": previous_report["provider"] if previous_report else provider,
        "model": previous_report["model"] if previous_report else model,
        "judge_provider": judge_provider, "judge_model": judge_model, "grader": JUDGE_VERSION,
        "judge_backend": os.getenv("EVAL_JUDGE_BACKEND", "entailment"),
        "mode": "regrade" if previous_report else "workflow_run",
        "source_started_at": previous_report.get("started_at") if previous_report else None,
        "planned_cases": len(dataset["cases"]), "results": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for case in dataset["cases"]:
        usage, judge_usage = TokenUsage(), TokenUsage()
        result = {**case, "predicted_diagnosis": "", "predicted_recommendation": {},
                  "passed": None, "status": "fail", "errors": [], "judge_latency_seconds": 0.0}
        state: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            # A fresh input per case; the normal review interrupt is the evaluation boundary.
            # Never pass references to the workflow or automatically approve an RCA.
            if previous_report is not None:
                old = previous[case["id"]]
                if old.get("error_stage") == "workflow" or (old.get("status") == "error" and not old.get("grading")
                        and not str(old.get("error", "")).startswith("Judge error:")):
                    raise RuntimeError(old.get("error", "Prior workflow failed."))
                state = {"root_cause": {"primary_cause": old["predicted_diagnosis"]},
                         "recommendations": old["predicted_recommendation"], "errors": old.get("errors", []),
                         "__interrupt__": old.get("awaiting_human_review", False)}
            else:
                state = workflow.invoke({"ticket": case["ticket"], "uploaded_files": [], "errors": []},
                                        config={"callbacks": [usage]})
            result["predicted_diagnosis"] = (state.get("root_cause") or {}).get("primary_cause", "")
            result["predicted_recommendation"] = state.get("recommendations") or {}
            result["errors"] = state.get("errors") or []
            result["awaiting_human_review"] = bool(state.get("__interrupt__"))
        except Exception as exc:
            result.update(status="error", error_stage="workflow", error=f"{type(exc).__name__}: {exc}")
        result["latency_seconds"] = time.perf_counter() - started
        result["token_usage"] = usage.snapshot()
        if previous_report is not None:
            result["latency_seconds"] = previous[case["id"]]["latency_seconds"]
            result["token_usage"] = previous[case["id"]]["token_usage"]
        result["deterministic_checks"] = {
            "diagnosis_present": isinstance(result["predicted_diagnosis"], str) and bool(result["predicted_diagnosis"].strip()),
            "recommendation_present": has_recommendation(result["predicted_recommendation"]),
            "workflow_completed": result.get("awaiting_human_review", False) or state.get("current_step") == "human_review_agent",
        }
        if result["status"] != "error":
            if not all(result["deterministic_checks"][key] for key in ("diagnosis_present", "recommendation_present")):
                result.update(status="error", error_stage="workflow", error="Workflow did not produce diagnosis and recommendations.")
            else:
                started = time.perf_counter()
                try:
                    grade = grader(case, result["predicted_diagnosis"], result["predicted_recommendation"], judge_usage)
                    result["grading"] = grade
                    result["passed"] = grade["diagnosis_pass"] and grade["recommendation_pass"]
                    result["status"] = "pass" if result["passed"] else "fail"
                except Exception as exc:
                    result.update(status="error", error_stage="judge", error=f"Judge error: {type(exc).__name__}: {exc}")
                    result["judge_attempts"] = getattr(exc, "attempts", [])
                result["judge_latency_seconds"] = time.perf_counter() - started
        result["judge_token_usage"] = judge_usage.snapshot()
        report["results"].append(result)
        results = report["results"]
        report["summary"] = summarize_results(results)
        report["complete"] = len(results) == report["planned_cases"]
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/golden_dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation/results.json"))
    parser.add_argument("--regrade", type=Path, help="Regrade saved predictions without rerunning agents.")
    parser.add_argument("--judge-model", help="Override the judge model only (same provider as the workflow).")
    parser.add_argument("--judge-backend", choices=("entailment", "llm"), default="entailment")
    parser.add_argument("--rubric", type=Path, default=Path("data/evaluation/judge_rubric.json"),
                        help="Reviewed semantic criteria bound to the dataset references.")
    args = parser.parse_args()
    if args.dataset.resolve() == args.output.resolve():
        parser.error("Output path must differ from the dataset path.")
    if args.regrade and args.regrade.resolve() == args.output.resolve():
        parser.error("Regrade source and output paths must differ.")
    from dotenv import load_dotenv
    load_dotenv()
    os.environ["EVAL_JUDGE_BACKEND"] = args.judge_backend
    if args.judge_model:
        os.environ["EVAL_JUDGE_MODEL"] = args.judge_model
    try:
        dataset = attach_rubric(load_dataset(args.dataset), args.rubric)
        previous_report = json.loads(args.regrade.read_text(encoding="utf-8")) if args.regrade else None
        report = run_evaluation(dataset, args.output, previous_report=previous_report)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["summary"], indent=2))
    print(f"Results: {args.output}")
    return 0 if report["summary"]["failed"] == 0 and report["summary"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
