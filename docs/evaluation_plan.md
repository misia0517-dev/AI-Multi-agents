# SeeWeeS Evaluation Plan

This project has two different kinds of behavior, so the evaluation suite should stay split:

1. Deterministic operations logic that can be checked exactly.
2. LLM-generated planning and executive reporting that should be checked with rubrics and scenario judges.

## Current Local Evals

The local pytest evals in `tests/test_evaluations.py` cover the stable, API-free layer:

| Eval | Purpose | Pass criteria |
|---|---|---|
| CSV reconciliation and 48h KPIs | Verifies item-master mapping, DQ exclusions, planning-window counts, corridor/day demand, and truck requirements. | Matches known counts from the enhancement dataset and validates alias/legacy mappings. |
| Weather risk thresholds | Verifies heavy rain, wind, freezing, day-risk aggregation, buffer mapping, and escalation behavior using mocked forecasts. | Risk flags and buffer match the playbook rules. |
| Resource allocation under scarcity | Verifies the allocator prioritizes Tier 1 corridor demand before Tier 2 when capacity is constrained. | Tier 1 has zero impact, Tier 2 absorbs expected penalty. |
| Planner validation loop | Verifies the graph detects an escalated corridor missing from the dispatch plan and routes back to planner. | One validation violation and retry route to planner. |

Run:

```bash
pytest tests/test_evaluations.py
```

## Embedded Eval Harness

The fuller eval framework now lives in `evals/`. It includes:

| File | Role |
|---|---|
| `evals/runner.py` | CLI runner and weighted score aggregation. |
| `evals/fixtures.py` | Built-in golden scenarios for pass/fail behavior. |
| `evals/validators.py` | Agent-agnostic deterministic validators. |
| `evals/adapter_mia.py` | Adapter for this project's LangGraph final state. |
| `evals/validators_mia.py` | Project-specific validators for penalty scoring, shortfalls, escalation, retries, and day-level feasibility. |
| `evals/llm_judges.py` | Optional LLM-as-judge rubrics. |
| `evals/compare.py` | Comparison utility for saved eval JSON files. |

Run one passing built-in fixture:

```bash
python -m evals --scenario happy_path
```

Run the full fixture suite, including intentionally failing scenarios that verify the validators catch bad outputs:

```bash
python -m evals
```

Run embedded evals after the actual LangGraph pipeline completes:

```bash
RUN_EVALS=true python src/main.py
```

When `RUN_EVALS=true`, the app saves a scored report to `eval_results/eval_<timestamp>.json`.

## Recommended LLM Evals

Add these after deterministic evals are stable and model/API access is configured:

| Eval | Input fixture | Rubric |
|---|---|---|
| Planner completeness | Fixed business context, ops insights, weather risk, and allocation JSON. | Mentions each corridor and Day0/Day1, explicitly prioritizes Tier 1, includes resource rationale, handles shortfalls, and defines monitoring triggers. |
| Planner correction | Same scenario plus synthetic validation violations. | Addresses every violation by name and does not introduce new impossible allocations. |
| Report executive quality | Fixed final state with dispatch plan and KPI JSON. | Contains required sections, KPI cards, corridor risk table, resource table, DQ summary, owner-tagged actions, and no markdown fences. |
| Grounding | Fixed final state plus generated plan/report. | Every numeric claim about units, penalties, buffers, and resources must match the provided state. |

For LLM evals, prefer a JSON fixture per scenario under `evals/fixtures/` and a judge prompt that returns structured pass/fail fields. Keep exact numeric checks in Python wherever possible, and reserve model judging for clarity, completeness, prioritization, and executive usefulness.

## Suggested Scenario Fixtures

1. **Nominal 48h plan**: current enhancement dataset with enough resources and low weather risk.
2. **Cold-chain constrained**: only one temperature-controlled truck available per day.
3. **Escalated Tier 2 weather**: C2 has risk score 3 and must be explicitly addressed.
4. **Tier 1 shortfall**: insufficient drivers for C1, requiring explicit defer/reassign mitigation.
5. **DQ-heavy feed**: missing IDs, legacy IDs, aliases, and duplicates to stress reconciliation/reporting.
