Evaluation baseline recorded 2026-09-06T21:39:35.663552+00:00.

Only the evaluation framework changed. Workflow and judge both used Ollama `llama3.2`; the final judge is `semantic_reference_judge_v2`. Agent code, graph code, model factory, and the golden dataset were unchanged.

| Run | Cases | Passed | Semantic failures | Errors | Pass rate among graded cases |
|---|---:|---:|---:|---:|---:|
| Original v1 | 30 | 1 | 24 | 5 | 4.0% |
| Original predictions regraded with v2 | 30 | 15 | 15 | 0 | 50.0% |
| Fresh workflow run with v2 | 30 | 15 | 15 | 0 | 50.0% |

The original report's 3.3% rate was 1/30, including five judge errors in its denominator. Its pass rate among successfully graded cases was 1/25 = 4.0%. V2 reports grading coverage separately and excludes ungraded errors from semantic pass/fail. Fresh grading coverage is 100%; diagnosis passed 19/30 and recommendation passed 19/30. Both must pass for a case to pass.

All 30 fresh predictions exactly match the original predictions, and the regrade and fresh run both score 15/30. The score change reflects the evaluator, not improved agent outputs. All workflows reached their normal human-review boundary without approving the RCA.

Workflow latency averaged 5.039 seconds per case; judge latency averaged 1.593 seconds separately. Reported workflow chat tokens totaled 40,332; judge tokens totaled 47,074. All usage records are complete for callback-visible chat calls. Judge assessment requires two model calls per case, before any retries. Embedding usage and hidden provider-internal retries are not included.

The final rubric matched all 6 labeled calibration cases (12 dimension judgments), including paraphrases, extra harmless detail, missing remedies, unrelated/related wrong causes, and unsupported certainty. These are development calibration examples, not a held-out accuracy estimate. Unit validation: 18 tests passed.

Residual limitation: the local judge still makes mistakes outside calibration. For example, it passes demo11's recommendation even though it does not renew the expired certificate, and passes demo28's diagnosis despite confusing an absent ticket with absent logs. Treat the 50% result as a provisional LLM-judged baseline; manually review substantive claims and calibrate a stronger judge before using it as an agent-quality target. No scores were manually overridden.

Recorded failed cases: demo1-deployment-failure, demo3-redis-timeout, demo4-nginx-504, demo6-cache-corruption, demo7-kafka-lag, demo9-rate-limit, demo12-memory-pressure, demo17-conflicting-evidence, demo18-intermittent-timeout, demo19-missing-retrieval, demo20-multiple-causes, demo21-partial-outage, demo22-stale-metrics, demo23-duplicate-retry, demo24-known-openai-quota.

Reproduce the fresh run:

```bash
python -m support_troubleshooting_agent.evaluation.runner --judge-model llama3.2
```

Regrade the preserved original answers without calling agents:

```bash
python -m support_troubleshooting_agent.evaluation.runner --judge-model llama3.2 --regrade reports/evaluation/results.v1.json --output reports/evaluation/results.regraded.json
```

Artifacts: [fresh baseline](results.json), [original baseline](results.v1.json), [original answers regraded](results.regraded.json), [calibration](judge_calibration.json).
