from __future__ import annotations
from typing import Dict, Any, List
from langchain_anthropic import ChatAnthropic
from prompts import (
    PDF_CONTEXT_PROMPT,
    OPS_ANALYSIS_PROMPT,
    PLANNER_PROMPT,
    PLANNER_RETRY_PROMPT,
    REPORT_PROMPT,
)

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.2,
    max_tokens=4096,
)

# Report agent needs more tokens — full HTML doc with tables, CSS, and multiple sections
llm_report = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.2,
    max_tokens=8192,
)


def run_context_agent(snippets: str) -> str:
    return llm.invoke(PDF_CONTEXT_PROMPT.format_messages(snippets=snippets)).content


def run_ops_agent(summary: Dict[str, Any], kpis: Dict[str, Any], anomalies_md: str) -> str:
    return llm.invoke(OPS_ANALYSIS_PROMPT.format_messages(
        summary=summary, kpis=kpis, anomalies_md=anomalies_md
    )).content


def run_planner_agent(
    business_context: str,
    ops_insights: str,
    weather_risk: Dict[str, Any],
    resource_allocation: Dict[str, Any] | None = None,
    human_review_summary: str = "",
) -> str:
    return llm.invoke(PLANNER_PROMPT.format_messages(
        business_context=business_context,
        ops_insights=ops_insights,
        weather_risk=weather_risk,
        resource_allocation=resource_allocation or {},
        human_review_summary=human_review_summary,
    )).content


def run_planner_retry_agent(
    business_context: str,
    ops_insights: str,
    weather_risk: Dict[str, Any],
    resource_allocation: Dict[str, Any] | None = None,
    violations: List[str] | None = None,
    human_review_summary: str = "",
) -> str:
    """
    Called when node_validate returns failures.
    Passes the specific constraint violations so the planner can address them.
    """
    violation_text = "\n".join(f"• {v}" for v in (violations or []))
    return llm.invoke(PLANNER_RETRY_PROMPT.format_messages(
        business_context=business_context,
        ops_insights=ops_insights,
        weather_risk=weather_risk,
        resource_allocation=resource_allocation or {},
        violations=violation_text,
        human_review_summary=human_review_summary,
    )).content


def run_report_agent(
    business_context: str,
    kpis: Dict[str, Any],
    anomaly_highlights: str,
    weather_risk: Dict[str, Any],
    dispatch_plan: str,
    resource_allocation: Dict[str, Any] | None = None,
    human_review_summary: str = "",
) -> str:
    return llm_report.invoke(REPORT_PROMPT.format_messages(
        business_context=business_context,
        kpis=kpis,
        anomaly_highlights=anomaly_highlights,
        weather_risk=weather_risk,
        dispatch_plan=dispatch_plan,
        resource_allocation=resource_allocation or {},
        human_review_summary=human_review_summary,
    )).content
