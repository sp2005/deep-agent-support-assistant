"""Audit judge agreement against frozen, directly reviewed saved predictions.

The manual labels are compared only AFTER grading and are never sent to the judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .runner import attach_rubric, load_dataset, run_evaluation


def prediction_fingerprint(case: dict[str, Any]) -> str:
    keys = ("id", "ticket", "expected_diagnosis", "expected_recommendation",
            "predicted_diagnosis", "predicted_recommendation")
    return hashlib.sha256(json.dumps({key: case[key] for key in keys}, sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()


def compare_review(report: dict[str, Any], review: dict[str, Any], split: str) -> dict[str, Any]:
    labels = [item for item in review["cases"] if split == "all" or item["split"] == split]
    predictions = {item["id"]: item for item in report["results"]}
    if len(predictions) != len(report["results"]):
        raise ValueError("Duplicate prediction IDs in report.")
    comparisons = []
    for label in labels:
        item = predictions.get(label["id"])
        if item is None or prediction_fingerprint(item) != label["prediction_sha256"]:
            raise ValueError(f"Missing or changed prediction for reviewed case {label['id']}.")
        expected = [label["diagnosis_pass"], label["recommendation_pass"]]
        actual = [item.get("grading", {}).get(key) for key in ("diagnosis_pass", "recommendation_pass")]
        comparisons.append({"id": label["id"], "expected": expected, "actual": actual,
                            "matches": actual == expected, "manual_reason": label["reason"],
                            "case_false_positive": item.get("passed") is True and not all(expected),
                            "case_false_negative": item.get("passed") is False and all(expected)})
    matched = sum(item["matches"] for item in comparisons)
    dimension_errors = {}
    for i, dimension in enumerate(("diagnosis", "recommendation")):
        dimension_errors[dimension] = {
            "false_positives": sum(item["actual"][i] is True and not item["expected"][i] for item in comparisons),
            "false_negatives": sum(item["actual"][i] is False and item["expected"][i] for item in comparisons),
            "ungraded": sum(item["actual"][i] is None for item in comparisons),
        }
    return {"split": split, "reviewer": review["reviewer"], "matched_cases": matched,
            "total_cases": len(comparisons), "all_dimensions_agree": bool(comparisons) and matched == len(comparisons),
            "case_false_positives": sum(item["case_false_positive"] for item in comparisons),
            "case_false_negatives": sum(item["case_false_negative"] for item in comparisons),
            "dimension_errors": dimension_errors, "comparisons": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/golden_dataset.json"))
    parser.add_argument("--rubric", type=Path, default=Path("data/evaluation/judge_rubric.json"))
    parser.add_argument("--source", type=Path, default=Path("reports/evaluation/results.v2.json"))
    parser.add_argument("--review", type=Path, default=Path("reports/evaluation/manual_review.json"))
    parser.add_argument("--split", choices=("calibration", "validation", "all"), default="calibration")
    parser.add_argument("--compare-only", type=Path, help="Audit this existing report without model calls.")
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation/judge_calibration.json"))
    args = parser.parse_args()
    if args.output.resolve() in {path.resolve() for path in (args.source, args.review, args.dataset, args.rubric)} or (args.compare_only and args.output.resolve() == args.compare_only.resolve()):
        parser.error("Output must differ from input paths.")
    load_dotenv()
    review = json.loads(args.review.read_text())
    if len({item["id"] for item in review["cases"]}) != len(review["cases"]):
        parser.error("Duplicate review IDs.")
    if args.compare_only:
        report = json.loads(args.compare_only.read_text())
    else:
        source = json.loads(args.source.read_text())
        # Validate provenance before making model calls; the result is not supplied to the judge.
        compare_review(source, review, args.split)
        ids = {item["id"] for item in review["cases"] if args.split == "all" or item["split"] == args.split}
        if args.split == "calibration" and len(ids) < 10:
            parser.error("Calibration requires at least ten reviewed cases.")
        dataset = attach_rubric(load_dataset(args.dataset), args.rubric)
        dataset["cases"] = [case for case in dataset["cases"] if case["id"] in ids]
        previous = {**source, "results": [item for item in source["results"] if item["id"] in ids]}
        report = run_evaluation(dataset, args.output, previous_report=previous)
    report["manual_review_comparison"] = compare_review(report, review, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    audit = report["manual_review_comparison"]
    print(f"Agreement: {audit['matched_cases']}/{audit['total_cases']} cases (both dimensions). {args.output}")
    return 0 if audit["all_dimensions_agree"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
