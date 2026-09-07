"""Local semantic coverage checks using a pinned natural-language inference model.

Model: https://huggingface.co/cross-encoder/nli-deberta-v3-small
No prediction or ticket text is sent over the network. Downloads are explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_ID = "cross-encoder/nli-deberta-v3-small"
REVISION = "80dd65bdcda723aee9ed855dc1c3082d8b3e6842"
MODEL_FILE = "onnx/model_qint8_arm64.onnx"
MODEL_SHA256 = "a1da6815c8da542d16381c7d84892d409b4b6604d655bb3054ecb15dd8714e6e"
THRESHOLD = 0.8


def cache_path() -> Path:
    return Path(os.getenv("EVAL_NLI_CACHE", "data/evaluation/model_cache/nli-deberta-v3-small"))


class EntailmentEvaluator:
    def __init__(self, path: Path) -> None:
        model_path = path / MODEL_FILE
        if not model_path.exists():
            raise ValueError("Entailment model missing. Run python -m support_troubleshooting_agent.evaluation.entailment --download")
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError("Entailment model checksum does not match the pinned model.")
        config = json.loads((path / "config.json").read_text())
        self.entailment_index = int(config["label2id"]["entailment"])
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        # Fail visibly on long inputs rather than silently dropping a remedy or contradiction.
        self.tokenizer.no_truncation()
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.model = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])

    def score(self, premise: str, hypothesis: str) -> tuple[float, int]:
        encoded = self.tokenizer.encode(premise, hypothesis)
        if len(encoded.ids) > 512:
            raise ValueError("Entailment input exceeds 512 tokens; requires review or an explicitly reviewed chunking strategy.")
        inputs = {"input_ids": np.asarray([encoded.ids], dtype=np.int64),
                  "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64)}
        logits = self.model.run(None, inputs)[0][0]
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        return float(probabilities[self.entailment_index]), len(encoded.ids)


@lru_cache(maxsize=1)
def get_evaluator(path: str) -> EntailmentEvaluator:
    return EntailmentEvaluator(Path(path))


def flatten_answer(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(flatten_answer(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(flatten_answer(item) for item in value)
    return str(value)


def grade_entailment(case: dict[str, Any], diagnosis: str, recommendation: Any) -> dict[str, Any]:
    criteria = case.get("evaluation_criteria")
    if not criteria:
        raise ValueError("Entailment evaluation requires reviewed acceptance criteria.")
    evaluator = get_evaluator(str(cache_path().resolve()))
    assessments = {}
    comparisons = input_tokens = 0
    for dimension, answer in (("diagnosis", diagnosis), ("recommendation", flatten_answer(recommendation))):
        premise = ("The diagnosis is: " if dimension == "diagnosis" else "The recommendation is: ") + answer
        decisions = []
        for criterion in criteria[dimension]:
            scores = []
            for hypothesis in criterion["hypotheses"]:
                statement = hypothesis if dimension == "diagnosis" else "The recommendation is to " + hypothesis[0].lower() + hypothesis[1:]
                score, tokens = evaluator.score(premise, statement)
                comparisons += 1
                input_tokens += tokens
                scores.append({"hypothesis": hypothesis, "entailment_score": score})
            best = max(item["entailment_score"] for item in scores)
            decisions.append({"criterion": criterion["description"], "alternatives": scores,
                              "score": best, "satisfied": best >= THRESHOLD})
        assessments[dimension] = {"verdict": "pass" if all(item["satisfied"] for item in decisions) else "fail",
                                  "criteria": decisions}
    unmet = [f"{dimension}: {item['criterion']}" for dimension, assessment in assessments.items()
             for item in assessment["criteria"] if not item["satisfied"]]
    return {"diagnosis_pass": assessments["diagnosis"]["verdict"] == "pass",
            "recommendation_pass": assessments["recommendation"]["verdict"] == "pass",
            "reason": "Unsupported required meaning: " + "; ".join(unmet) if unmet else "All required meanings supported.",
            "assessment": assessments, "threshold": THRESHOLD, "model_revision": REVISION,
            "model_sha256": MODEL_SHA256,
            "semantic_usage": {"comparisons": comparisons, "input_tokens": input_tokens,
                               "generated_tokens": 0, "tokenizer": MODEL_ID}, "attempts": []}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", required=True)
    parser.parse_args()
    from huggingface_hub import hf_hub_download
    for filename in (MODEL_FILE, "tokenizer.json", "config.json"):
        hf_hub_download(MODEL_ID, filename, revision=REVISION, local_dir=str(cache_path()), token=False)
    get_evaluator(str(cache_path().resolve()))
    print(f"Verified pinned entailment model at {cache_path()}")


if __name__ == "__main__":
    main()
