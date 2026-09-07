The measured baseline improved from **2/30 to 5/30 (6.7% to 16.7%, +10 percentage points)** for both recommendation acceptance and overall pass rate. There are **25 failing cases and 0 evaluation errors**. Every workflow agent still uses **Llama 3.2**. No new model was installed or configured.

This is the final full-workflow run using the unchanged calibrated evaluator, not the earlier 15/30 provisional judge or an exploratory candidate. The original baseline is preserved in [recommendation_before.json](recommendation_before.json); final predictions are in [recommendation_after.json](recommendation_after.json). [Machine-readable comparison](recommendation_comparison.json) includes all case transitions and artifact fingerprints.

| Measure | Before | After |
|---|---:|---:|
| Recommendation acceptance | 2/30 (6.7%) | 5/30 (16.7%) |
| Overall pass rate | 2/30 (6.7%) | 5/30 (16.7%) |
| Diagnosis acceptance | 15/30 | 15/30 |
| Evaluation / judge / workflow errors | 0 / 0 / 0 | 0 / 0 / 0 |
| Mean workflow latency | 5.04 s | 4.22 s |
| Total workflow chat tokens | 40,332 | 49,269 |
| Workflow model calls | 120 | 122 |

Latency is an observed single-run comparison, not a controlled performance benchmark. Token totals include all workflow agents; the local NLI evaluator has separate classification usage.

Four cases changed from fail to pass in both dimensions:

- **demo2-mongodb-timeout:** confirmed connectivity repair plus order-creation recovery checks.
- **demo3-redis-timeout:** Redis performance/connectivity remediation plus session/logout recovery checks.
- **demo7-kafka-lag:** processing repair, capacity conditions, and checks for lag drainage without data loss.
- **demo10-dns-resolution:** confirmed resolver/network repair and worker recovery checks; direct review flags a possible gap in explicit notification-delivery verification.

**Scored regression: demo16-no-logs.** Its new response explicitly collects logs/performance data, timestamps, request IDs, affected endpoints and reproduction steps, and defers production changes. Direct review considers this appropriate; the regression appears to be a judge false negative. It remains counted as a failure in the reported baseline. **demo27-ollama-unavailable** remains passing.

The Recommendation Agent now selects from generic, conditional remediation patterns supported by incident evidence, resolves component names back to the selected source, and generates a focused recovery test. It separates immediate assessment/mitigation, corrective remediation, verification and prevention while retaining the existing four-field downstream contract. Every corrective step has a reason, paired verification and recovery guidance. Low confidence remains explicit. Generic KB prose alone cannot establish an incident failure mode; unsupported categories lead to evidence gathering. The patterns contain no evaluation case IDs, reference answers, or judge labels.

Operational actions and rollback guidance come from bounded patterns rather than unconstrained model prose. Validation rejects missing sources, invented numeric settings/targets, command or security-bypass instructions in generated tests, and common inverted recovery predicates. Invalid test generation retries once and then uses a positive pattern-based recovery check; provider failures retain conservative fallback behavior. These checks reduce unsupported guidance but are not a proof of semantic correctness.

Only the Recommendation Agent implementation changed among the pre-recorded source-file hashes. The evaluator, rubric, golden dataset, LangSmith integration, model factory and other agents are unchanged. All 30 predicted diagnosis strings are identical to the previous baseline. **51 tests pass**, including source grounding, unsupported diagnoses/runbooks, bounded retry/fallback, low confidence, numeric units, command rejection and failure-state inversion.

Remaining limits: **18/30 cases return information-gathering-only guidance**. The limited category set and conservative selection still defer some clear incidents, including certificate expiration. Component selection can confuse symptoms with components, and complex/multiple-cause incidents remain difficult. Some generated tests imply more end-to-end coverage than they explicitly demonstrate. The same 30 development cases informed iteration; this result does not establish unseen-case generalization or run-to-run stability.

A direct review of 11 representative final recommendations found four likely recommendation false negatives (demo5, demo8, demo16, demo28) and one possible over-credit (demo10). These notes do not change automated grades or calibrate the evaluator. The new rate is a **measured automated baseline**, not a claim that the judge is error-free on the new output format. [Detailed review and prediction fingerprints](recommendation_review.json).

| Case | Judge recommendation | Direct review | Reason |
|---|---|---|---|
| demo2-mongodb-timeout | pass | pass | Conditional connectivity repair is paired with successful order creation and disappearance of database-selection timeouts. |
| demo3-redis-timeout | pass | pass | Restores Redis connectivity/performance with checks for reduced latency and no intermittent logouts; no invented capacity setting. |
| demo4-nginx-504 | fail | fail | The inverted 504 recovery predicate is rejected and replaced by a positive test. The plan still targets Nginx broadly instead of tracing the slow checkout upstream. |
| demo5-authentication | fail | pass | Updates trusted certificate configuration after confirming mismatch, reloads authentication, and verifies login/token issuance. The automated rejection appears to be a false negative. |
| demo7-kafka-lag | pass | pass | Repairs blocked processing or scales only with capacity evidence; preserves offsets and explicitly verifies lag drains without loss or duplicate processing. |
| demo8-disk-full | fail | pass | Frees disk space safely or expands after capacity checks, preserves business data, and explicitly verifies application/logging writes. The automated rejection appears to be a false negative. |
| demo10-dns-resolution | pass | needs_review | Corrective DNS/network guidance is useful, but the test explicitly checks hostname resolution and retries rather than delivery of a notification. The automated pass may over-credit end-to-end coverage. |
| demo11-tls-expiry | fail | fail | The selector chooses evidence gathering despite a supported certificate-expiration category; no renewal/deployment is returned. |
| demo16-no-logs | fail | pass | Collects logs/performance data, timestamps, failing request IDs, affected endpoints and reproduction steps, while deferring production changes. This is appropriate evidence gathering; the scored regression appears to be a false negative. |
| demo27-ollama-unavailable | pass | pass | Conditionally starts the configured service, checks readiness/dependencies and retries the model request. |
| demo28-missing-ticket | fail | pass | Explicitly requests the original support ticket and affected operation. Recommendation rejection appears to be a false negative; the unchanged diagnosis still fails. |

Reproduce with the existing Llama 3.2 configuration:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m support_troubleshooting_agent.evaluation.runner --output reports/evaluation/recommendation_after.json
```

The evaluation exits 1 because 25 cases fail; this is not a runtime exception.
