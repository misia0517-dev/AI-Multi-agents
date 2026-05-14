"""
LLM-as-Judge Evaluators
========================
Qualitative evaluation using any LLM as a rubric-based judge.
Fully provider-agnostic — accepts a simple callable interface.

The judge_fn signature:
    judge_fn(system_prompt: str, user_prompt: str) -> str

This can wrap OpenAI, Anthropic, Google, or any other provider.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .validators import EvalResult
from .schemas import AgentPipelineOutput


# Type alias for the judge function
JudgeFn = Callable[[str, str], str]


# ---------------------------------------------------------------------------
# Rubric definitions
# ---------------------------------------------------------------------------

PLAN_COHERENCE_RUBRIC = """
You are evaluating a dispatch plan for a specialty medicine logistics operation.

Score the plan on COHERENCE (1-5 scale):

5 = Excellent: Plan is internally consistent, all recommendations follow logically
    from the data (weather, resources, demand). No contradictions.
4 = Good: Minor inconsistencies but overall sound logic.
3 = Adequate: Some logical gaps or unsupported recommendations.
2 = Poor: Significant contradictions or recommendations that conflict with the data.
1 = Failing: Plan is incoherent or contradicts its own inputs.

EVALUATION CRITERIA:
- Do weather buffer recommendations match the stated risk scores?
- Do resource assignments align with stated demand volumes?
- Are contingency plans consistent with identified risks?
- Do KPI estimates match the allocation data?

Return ONLY a JSON object:
{"score": <1-5>, "reasoning": "<brief explanation>"}
"""

ACTIONABILITY_RUBRIC = """
You are evaluating a dispatch plan for executive decision-making.

Score the plan on ACTIONABILITY (1-5 scale):

5 = Excellent: Every recommendation is specific, time-bound, and assignable.
    A dispatcher could execute this plan immediately.
4 = Good: Most recommendations are actionable with minor clarification needed.
3 = Adequate: Mix of actionable and vague recommendations.
2 = Poor: Mostly vague guidance with few concrete actions.
1 = Failing: No actionable recommendations.

EVALUATION CRITERIA:
- Are truck/driver assignments specific (numbers, corridors, dates)?
- Are time windows specified for dispatches?
- Are escalation triggers clearly defined?
- Could a non-technical ops manager act on this plan?

Return ONLY a JSON object:
{"score": <1-5>, "reasoning": "<brief explanation>"}
"""

COMPLETENESS_RUBRIC = """
You are evaluating a dispatch plan for a multi-corridor medicine logistics operation.

Score the plan on COMPLETENESS (1-5 scale):

5 = Excellent: Covers ALL corridors, ALL resource types, weather impacts,
    SLA risks, cost implications, contingencies, and monitoring actions.
4 = Good: Covers most required areas with 1-2 minor gaps.
3 = Adequate: Covers core areas but missing significant sections.
2 = Poor: Major sections missing (e.g., no contingency plan, no cost analysis).
1 = Failing: Covers only 1-2 aspects.

REQUIRED SECTIONS (per the SeeWeeS Playbook):
1. Per-corridor dispatch schedule
2. Weather risk assessment with buffers
3. Resource allocation rationale
4. Cold-chain compliance verification
5. SLA risk flags
6. Contingency triggers
7. Cost estimates
8. Monitoring recommendations

Return ONLY a JSON object:
{"score": <1-5>, "reasoning": "<brief explanation>", "missing_sections": [<list>]}
"""

REPORT_QUALITY_RUBRIC = """
You are evaluating an HTML dispatch report intended for C-suite executives.

Score the report on EXECUTIVE READINESS (1-5 scale):

5 = Excellent: Concise, structured, decision-oriented. Top risks highlighted,
    concrete actions proposed, "why" explained. A non-technical executive can
    act on it immediately.
4 = Good: Well-structured with minor improvements needed for clarity.
3 = Adequate: Contains the information but not optimized for executive consumption.
2 = Poor: Data dump without clear takeaways or recommendations.
1 = Failing: Unstructured, confusing, or missing critical information.

EVALUATION CRITERIA:
- Does it lead with the most important decision/risk?
- Are tables/visuals used effectively?
- Is jargon minimized?
- Are recommendations clearly separated from data?
- Is the report skimmable (headings, bullets, highlights)?

Return ONLY a JSON object:
{"score": <1-5>, "reasoning": "<brief explanation>"}
"""


# ---------------------------------------------------------------------------
# Judge runner
# ---------------------------------------------------------------------------

def _parse_judge_response(raw: str) -> Dict[str, Any]:
    """Extract JSON from judge response, tolerant of markdown wrapping."""
    # Try direct JSON parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... }
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"score": 0, "reasoning": f"Failed to parse judge response: {raw[:200]}"}


def run_judge(
    judge_fn: JudgeFn,
    rubric: str,
    content_to_evaluate: str,
    context: str = "",
) -> Dict[str, Any]:
    """
    Run a single LLM judge evaluation.

    Args:
        judge_fn: Callable(system_prompt, user_prompt) -> str
        rubric: The scoring rubric (system prompt)
        content_to_evaluate: The agent output to score
        context: Additional context (inputs the agent received)

    Returns:
        Dict with score (1-5), reasoning, and any extra fields
    """
    user_prompt = f"""
CONTEXT (inputs the agent received):
{context[:3000]}

CONTENT TO EVALUATE:
{content_to_evaluate[:5000]}

Score this content according to the rubric.
"""
    raw_response = judge_fn(rubric, user_prompt)
    return _parse_judge_response(raw_response)


# ---------------------------------------------------------------------------
# Eval wrappers that return EvalResult
# ---------------------------------------------------------------------------

def judge_plan_coherence(
    output: AgentPipelineOutput, judge_fn: JudgeFn
) -> EvalResult:
    """LLM judge: Is the dispatch plan internally coherent?"""
    context = (
        f"Weather risks: {json.dumps([vars(w) for w in output.corridor_weather_risks], default=str)[:1500]}\n"
        f"Resource allocation: {json.dumps(vars(output.resource_allocation) if output.resource_allocation else {}, default=str)[:1500]}"
    )
    result = run_judge(judge_fn, PLAN_COHERENCE_RUBRIC, output.dispatch_plan, context)
    score_raw = result.get("score", 0)
    score_norm = score_raw / 5.0  # normalize to 0-1
    return EvalResult(
        name="plan_coherence_judge",
        passed=score_raw >= 3,
        score=score_norm,
        weight=1.5,
        details=result.get("reasoning", "No reasoning provided"),
    )


def judge_plan_actionability(
    output: AgentPipelineOutput, judge_fn: JudgeFn
) -> EvalResult:
    """LLM judge: Is the plan actionable for dispatchers?"""
    result = run_judge(judge_fn, ACTIONABILITY_RUBRIC, output.dispatch_plan)
    score_raw = result.get("score", 0)
    return EvalResult(
        name="plan_actionability_judge",
        passed=score_raw >= 3,
        score=score_raw / 5.0,
        weight=1.0,
        details=result.get("reasoning", ""),
    )


def judge_plan_completeness(
    output: AgentPipelineOutput, judge_fn: JudgeFn
) -> EvalResult:
    """LLM judge: Does the plan cover all required sections?"""
    context = f"Corridors: {output.corridors}"
    result = run_judge(judge_fn, COMPLETENESS_RUBRIC, output.dispatch_plan, context)
    score_raw = result.get("score", 0)
    missing = result.get("missing_sections", [])
    return EvalResult(
        name="plan_completeness_judge",
        passed=score_raw >= 3,
        score=score_raw / 5.0,
        weight=1.0,
        details=f"{result.get('reasoning', '')} Missing: {missing}" if missing else result.get("reasoning", ""),
    )


def judge_report_quality(
    output: AgentPipelineOutput, judge_fn: JudgeFn
) -> EvalResult:
    """LLM judge: Is the report executive-ready?"""
    result = run_judge(judge_fn, REPORT_QUALITY_RUBRIC, output.report_html)
    score_raw = result.get("score", 0)
    return EvalResult(
        name="report_quality_judge",
        passed=score_raw >= 3,
        score=score_raw / 5.0,
        weight=1.0,
        details=result.get("reasoning", ""),
    )


# ---------------------------------------------------------------------------
# Judge Registry
# ---------------------------------------------------------------------------

ALL_JUDGES = [
    judge_plan_coherence,
    judge_plan_actionability,
    judge_plan_completeness,
    judge_report_quality,
]


def run_all_judges(
    output: AgentPipelineOutput, judge_fn: JudgeFn
) -> List[EvalResult]:
    """Run all LLM-as-judge evaluators. Returns results list."""
    results = []
    for judge in ALL_JUDGES:
        try:
            results.append(judge(output, judge_fn))
        except Exception as e:
            results.append(EvalResult(
                name=judge.__name__,
                passed=False, score=0.0, weight=1.0,
                details=f"Judge error: {str(e)}",
            ))
    return results


# ---------------------------------------------------------------------------
# Provider adapter helpers (convenience wrappers)
# ---------------------------------------------------------------------------

def make_openai_judge(model: str = "gpt-4o-mini", temperature: float = 0.0) -> JudgeFn:
    """Create a judge function using OpenAI's API."""
    def judge(system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""
    return judge


def make_anthropic_judge(model: str = "claude-sonnet-4-20250514", temperature: float = 0.0) -> JudgeFn:
    """Create a judge function using Anthropic's API."""
    def judge(system_prompt: str, user_prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text
    return judge


def make_mock_judge() -> JudgeFn:
    """Create a mock judge for testing (no API calls). Returns score=3 always."""
    def judge(system_prompt: str, user_prompt: str) -> str:
        return json.dumps({
            "score": 3,
            "reasoning": "Mock judge — replace with real LLM for production evals",
        })
    return judge
