# SeeWeeS Evaluation Metrics

This document explains the evaluation metrics used to score the SeeWeeS multi-agent dispatch system.

The goal of these evals is to answer one practical question:

> Did the agent produce a dispatch plan that is operationally safe, grounded in the data, and useful for decision-makers?

The current eval framework uses deterministic metrics first. These are rule-based checks that do not require an LLM judge.

## Overall Scoring

Each metric returns:

- **Pass/fail**: whether the output satisfies the requirement.
- **Score**: a value from `0.0` to `1.0`.
- **Weight**: how important the metric is in the final score.

The final deterministic score is a weighted average of all metric scores.

The current pass threshold is:

```text
70%
```

Some metrics are treated as critical, especially:

- SLA compliance
- Cold-chain coverage
- Weather faithfulness

If a critical metric fails, the overall result can fail even when the weighted score is high.

## Metric Summary

| Metric | Weight | What It Checks |
|---|---:|---|
| SLA compliance | 2.0 | Deliveries stay within promised time limits. |
| Cold-chain coverage | 2.0 | Temperature-sensitive medicine has enough refrigerated capacity. |
| Buffer policy | 1.5 | Weather risk maps to the correct travel-time buffer. |
| Resource utilization | 1.5 | Trucks and drivers are used safely and efficiently. |
| Cost guardrails | 1.0 | Allocation cost stays under the configured ceiling. |
| Fairness metrics | 1.0 | Capacity is not unfairly concentrated in one corridor. |
| Weather consistency | 1.0 | Weather risk scores match the raw weather values. |
| Plan completeness | 1.5 | The dispatch plan covers all required operational topics. |
| Weather faithfulness | 1.5 | The plan does not invent weather conditions. |
| Report structure | 1.0 | The executive report includes required sections and tables. |
| Validation loop | 1.0 | Validation issues are acknowledged or resolved. |

## 1. SLA Compliance

SLA compliance means the dispatch plan keeps deliveries within the promised delivery-time limits.

For this project, the playbook defines two SLA tiers:

| SLA Tier | Meaning | Max Time In Transit |
|---|---|---:|
| Tier 1 | Life-critical medicine | 6 hours |
| Tier 2 | Standard specialty medicine | 12 hours |

The eval checks whether the route's weather-adjusted travel time stays within the allowed limit.

Example:

```text
Base travel time: 4.6 hours
Weather risk buffer: +10%
Adjusted travel time: 5.06 hours
Tier 1 limit: 6 hours
Result: SLA compliant
```

If the adjusted travel time is over the limit, the plan must clearly flag the risk or breach and propose mitigation, such as:

- early dispatch
- alternate routing
- escalation to operations leadership
- reallocating trucks or drivers

## 2. Cold-Chain Coverage

Cold-chain coverage checks whether temperature-sensitive medicines have enough refrigerated truck capacity.

Cold-chain items include products such as insulin, biologics, Remdesivir, and strict cold-chain clinical-trial drugs.

The eval checks:

```text
temperature-controlled truck capacity >= cold-chain demand after buffer
```

A passing plan must provide 100% cold-chain coverage.

Failing this metric means some temperature-sensitive units may not be safely deliverable.

## 3. Buffer Policy

Buffer policy checks whether the agent applies the correct travel-time buffer for the weather risk score.

The playbook defines:

| Weather Risk Score | Required Travel Buffer |
|---:|---:|
| 0 | 0% |
| 1 | 10% |
| 2 | 25% |
| 3 | 40% plus escalation |

Example:

```text
Risk score: 2
Expected buffer: 25%
Plan buffer: 25%
Result: pass
```

If a plan uses a 10% buffer for a risk score of 2, this metric fails.

## 4. Resource Utilization

Resource utilization checks whether trucks and drivers are assigned responsibly.

It looks for two things:

- Utilization should generally be in a healthy range.
- Driver count should be enough for the number of trucks assigned.

The evaluator treats very high utilization as risky because it leaves no room for delays, packing inefficiency, route changes, or emergency adjustments.

The current utilization guide is:

| Utilization | Interpretation |
|---:|---|
| 75-90% | Ideal |
| 90-95% | Slightly high but acceptable |
| More than 95% | Critical overload |
| Less than 75% | Under-utilized but usually acceptable |

## 5. Cost Guardrails

Cost guardrails check whether the estimated daily cost per corridor stays within the configured ceiling.

The current ceiling is:

```text
$3,000 per corridor per day
```

This metric is useful for detecting plans that solve an operational problem by throwing too much capacity at it.

## 6. Fairness Metrics

Fairness metrics check whether one corridor is starved while another corridor receives most of the resources.

The evaluator currently uses:

- max-min fairness ratio
- NSW score

The main threshold is:

```text
max-min fairness ratio >= 0.70
```

This does not mean every corridor must receive the same resources. Tier 1 and high-risk corridors can still receive priority. The metric only checks that lower-priority corridors are not ignored without justification.

## 7. Weather Consistency

Weather consistency checks whether the weather risk score matches the actual weather values.

The playbook thresholds are:

| Weather Condition | Threshold |
|---|---:|
| Heavy rain | precipitation >= 15mm/day |
| High wind | wind gusts >= 45 km/h |
| Freezing | minimum temperature <= 0C |

Example:

```text
Precipitation: 16mm
Wind gusts: 30 km/h
Minimum temperature: 5C
Expected flag: heavy rain
Expected risk score: 1
```

The metric fails if the plan reports a risk score that does not match the triggered weather flags.

## 8. Plan Completeness

Plan completeness checks whether the dispatch plan includes the operational details a planner or executive would expect.

It checks for:

- all corridors are mentioned
- weather or risk is discussed
- buffer or travel-time adjustment is included
- trucks, drivers, or resource allocation is included
- contingency or escalation triggers are included
- KPI, SLA, cost, utilization, or compliance impact is included

This metric does not judge style. It checks whether the plan contains the necessary operational content.

## 9. Weather Faithfulness

Weather faithfulness checks that the agent does not invent weather conditions.

For example, if the weather data only includes precipitation, wind, and temperature, the plan should not suddenly mention:

- snowfall
- snow accumulation
- fog
- visibility
- ice storm

This metric is important because hallucinated weather can lead to unnecessary or incorrect dispatch decisions.

## 10. Report Structure

Report structure checks whether the final HTML report includes the required executive-report elements.

It checks for:

- weather risk summary
- applied travel buffers
- valid and excluded shipment counts
- SLA risk flags
- HTML table structure
- corridor-level data

The report should be easy for leadership to scan and act on.

## 11. Validation Loop

Validation loop checks whether the system handles validation issues coherently.

If the validator finds no issues, this metric passes.

If the validator finds issues, the final plan should either:

- resolve them, or
- clearly acknowledge them and explain the remaining risk.

Example:

```text
Validation issue: C2_NJ_PHL requires escalation.
Expected plan behavior: explicitly mention C2_NJ_PHL escalation and mitigation.
```

## Project-Specific Metrics

The embedded framework also includes project-specific validators for this implementation.

These are defined in `evals/validators_mia.py`.

| Metric | What It Checks |
|---|---|
| Penalty model | Total penalty and feasibility flags are mathematically consistent. |
| Shortfall detection | Truck or driver shortages are flagged and addressed. |
| Escalation check | Risk score 3 corridors require escalation and must be mentioned in the plan. |
| Cold-chain over-assignment | Temp-controlled trucks assigned across corridors do not exceed availability. |
| Retry coherence | If the planner retried after validation, the final plan addresses the violations. |
| Day feasibility | Remaining resource pools are never negative. |

## Current Verified Results

For the embedded `happy_path` scenario:

| Aggregate | Result |
|---|---:|
| Deterministic score | 100% |
| Overall score | 100% |
| Pass threshold | 70% |
| Status | PASS |

For local project tests:

```text
6 passed
```

## How To Run

Run the local deterministic project tests:

```bash
pytest
```

Run the known-good embedded scenario:

```bash
python -m evals --scenario happy_path
```

Run the actual dispatch pipeline and score the generated output:

```bash
RUN_EVALS=true python src/main.py
```

The actual pipeline eval writes a JSON report to:

```text
eval_results/
```
