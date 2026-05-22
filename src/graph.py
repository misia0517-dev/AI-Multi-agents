from __future__ import annotations

import operator
import os
import re
from typing import Annotated, Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents import (
    run_context_agent,
    run_ops_agent,
    run_planner_agent,
    run_planner_retry_agent,
    run_report_agent,
)
from tools.csv_tools import (
    analyze_csv,
    analyze_csv_with_overrides,
    detect_planning_special_case_items,
)
from tools.email_tools import send_email_smtp
from tools.human_review_tools import (
    collect_human_review_decision,
    detect_human_review_triggers,
    summarize_human_review,
)
from tools.pdf_tools import PdfRag
from tools.policy_tools import load_dispatch_policy
from tools.resource_tools import allocate_resources, load_resource_availability
from tools.weather_tools import get_single_corridor_weather_risk

load_dotenv()

MAX_PLANNER_RETRIES = 2


class AppState(TypedDict, total=False):
    pdf_path: str
    csv_path: str
    resource_path: str
    corridor_id: str

    business_context: str

    csv_summary: Dict[str, Any]
    csv_kpis: Dict[str, Any]
    corridor_day_summary: Dict[str, Any]
    anomalies_md: str
    ops_insights: str

    corridor_weather_results: Annotated[List[Dict[str, Any]], operator.add]
    corridor_weather_risk: Dict[str, Any]

    human_review: Dict[str, Any]
    human_review_summary: str
    human_review_approvals: List[str]
    retry_policy: Dict[str, Any]
    retry_count: int

    resource_allocation: Dict[str, Any]
    special_case_items: Dict[str, Any]
    resource_reserve: Dict[str, Dict[str, int]]

    validation_violations: List[str]
    planner_retry_count: int

    dispatch_plan: str
    report_html: str


def _clean_html(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:html)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def node_pdf_context(state: AppState) -> AppState:
    print("[pdf_context] Building RAG index from Dispatch Playbook...")
    rag = PdfRag(persist_dir="chroma_db")
    vectordb = rag.build(state["pdf_path"])
    retriever = rag.retriever(vectordb, k=6)

    query = "Extract KPI definitions, thresholds, SLAs, constraints, dispatch rules, exceptions."
    docs = retriever.invoke(query)
    snippets = "\n\n---\n\n".join(d.page_content for d in docs)

    business_context = run_context_agent(snippets)
    return {"business_context": business_context}


def node_csv_analysis(state: AppState) -> AppState:
    print("[csv_analysis] Analysing multi-corridor shipment CSV...")
    res = analyze_csv(
        state["csv_path"],
        policy_path=state.get("pdf_path"),
    )
    special_case_items = detect_planning_special_case_items(
        state["csv_path"],
        policy_path=state.get("pdf_path"),
    )

    anomalies_md = "(none detected or insufficient numeric data)"
    if not res.anomalies.empty:
        anomalies_md = res.anomalies.head(12).to_markdown(index=False)

    ops_insights = run_ops_agent(
        summary=res.summary,
        kpis=res.kpis,
        anomalies_md=anomalies_md,
    )

    return {
        "csv_summary": res.summary,
        "csv_kpis": res.kpis,
        "corridor_day_summary": res.corridor_day_summary,
        "anomalies_md": anomalies_md,
        "ops_insights": ops_insights,
        "special_case_items": special_case_items,
    }


def _active_corridor_ids(state: AppState) -> List[str]:
    policy = load_dispatch_policy(state.get("pdf_path"))
    return list(policy.corridor_waypoints.keys())


def node_router(state: AppState) -> AppState:
    corridors = _active_corridor_ids(state)
    print(
        f"[router] pdf_context + csv_analysis complete. "
        f"Fanning out weather fetch to {len(corridors)} corridors: {corridors}"
    )
    return {}


def route_to_weather_corridors(state: AppState) -> List[Send]:
    return [
        Send("node_weather_corridor", {**state, "corridor_id": corridor_id})
        for corridor_id in _active_corridor_ids(state)
    ]


def node_weather_corridor(state: AppState) -> AppState:
    corridor_id = state["corridor_id"]
    print(f"  [weather_corridor] {corridor_id}: fetching waypoint forecasts...")
    risk = get_single_corridor_weather_risk(
        corridor_id,
        policy_path=state.get("pdf_path"),
    )
    print(
        f"  [weather_corridor] {corridor_id}: risk={risk.get('risk_score_48h', '?')} "
        f"| buffer={risk.get('travel_buffer_pct', '?')}% "
        f"| escalation={risk.get('escalation_required', '?')}"
    )
    return {"corridor_weather_results": [{"corridor_id": corridor_id, "risk": risk}]}


def node_collect_weather(state: AppState) -> AppState:
    results = state.get("corridor_weather_results", [])
    corridor_weather_risk: Dict[str, Any] = {}
    for entry in results:
        corridor_weather_risk[entry["corridor_id"]] = entry["risk"]

    print(f"[collect_weather] Aggregated weather risk for {len(corridor_weather_risk)} corridors.")
    return {"corridor_weather_risk": corridor_weather_risk}


def node_resource_allocator(state: AppState) -> AppState:
    resource_path = state.get(
        "resource_path",
        "data-for-enhancement/Resource_availability_48h.csv",
    )
    print("[resource_allocator] Loading resource availability...")
    availability = load_resource_availability(resource_path)
    availability = _apply_resource_reserve(availability, state.get("resource_reserve", {}))

    print("[resource_allocator] Running greedy allocation with penalty model...")
    allocation = allocate_resources(
        corridor_day_summary=state.get("corridor_day_summary", {}),
        availability=availability,
        corridor_weather_risk=state.get("corridor_weather_risk", {}),
        policy_path=state.get("pdf_path"),
    )

    summary = allocation.get("summary_48h", {})
    print(
        f"  Total penalty score: {summary.get('total_penalty_score', '?')} | "
        f"Feasible: {summary.get('allocation_feasible', '?')}"
    )
    return {"resource_allocation": allocation}


def node_human_review(state: AppState) -> AppState:
    triggers = detect_human_review_triggers(
        special_case_items=state.get("special_case_items", {}),
        resource_allocation=state.get("resource_allocation", {}),
        resource_reserve=state.get("resource_reserve", {}),
    )

    review = collect_human_review_decision(
        triggers=triggers,
        retry_count=int(state.get("retry_count", 0)),
        prior_approvals=state.get("human_review_approvals", []),
    )
    human_review_approvals = state.get("human_review_approvals", []) + review.get("approvals", [])
    human_review_summary = summarize_human_review(
        review=review,
        previous_summary=state.get("human_review_summary", ""),
    )

    return {
        "human_review": review,
        "human_review_summary": human_review_summary,
        "human_review_approvals": human_review_approvals,
        "retry_policy": review.get("retry_policy", {}),
    }


def node_apply_retry_policy(state: AppState) -> AppState:
    policy = state.get("retry_policy", {})
    hold_item_ids = [int(x) for x in policy.get("hold_item_ids", [])]
    hold_corridors = [str(x) for x in policy.get("hold_corridors", [])]
    resource_reserve = policy.get("resource_reserve", {})

    res = analyze_csv_with_overrides(
        state["csv_path"],
        policy_path=state.get("pdf_path"),
        hold_item_ids=hold_item_ids,
        hold_corridors=hold_corridors,
    )

    anomalies_md = "(none detected or insufficient numeric data)"
    if not res.anomalies.empty:
        anomalies_md = res.anomalies.head(12).to_markdown(index=False)

    updated_special_case_items = detect_planning_special_case_items(
        state["csv_path"],
        policy_path=state.get("pdf_path"),
        hold_item_ids=hold_item_ids,
        hold_corridors=hold_corridors,
    )

    if hold_item_ids or hold_corridors:
        updated_special_case_items = dict(updated_special_case_items)
        updated_special_case_items["held_by_human_review"] = hold_item_ids
        updated_special_case_items["held_corridors_by_human_review"] = hold_corridors

    return {
        "csv_summary": res.summary,
        "csv_kpis": res.kpis,
        "corridor_day_summary": res.corridor_day_summary,
        "anomalies_md": anomalies_md,
        "special_case_items": updated_special_case_items,
        "resource_reserve": resource_reserve,
        "retry_count": int(state.get("retry_count", 0)) + 1,
        "validation_violations": [],
        "planner_retry_count": 0,
    }


def node_planner(state: AppState) -> AppState:
    retry_count = state.get("planner_retry_count", 0)
    violations = state.get("validation_violations", [])

    if retry_count > 0 and violations:
        print(
            f"[planner] Retry #{retry_count} - revising plan to address "
            f"{len(violations)} constraint violation(s)..."
        )
        plan = run_planner_retry_agent(
            business_context=state.get("business_context", ""),
            ops_insights=state.get("ops_insights", ""),
            weather_risk=state.get("corridor_weather_risk", {}),
            resource_allocation=state.get("resource_allocation", {}),
            human_review_summary=state.get("human_review_summary", ""),
            violations=violations,
        )
    else:
        print("[planner] Generating initial multi-corridor dispatch plan...")
        plan = run_planner_agent(
            business_context=state.get("business_context", ""),
            ops_insights=state.get("ops_insights", ""),
            weather_risk=state.get("corridor_weather_risk", {}),
            human_review_summary=state.get("human_review_summary", ""),
            resource_allocation=state.get("resource_allocation", {}),
        )

    return {"dispatch_plan": plan}


def node_validate(state: AppState) -> AppState:
    violations: List[str] = []
    allocation = state.get("resource_allocation", {})
    plan_text = state.get("dispatch_plan", "").lower()

    for day in ["Day0", "Day1"]:
        day_data = allocation.get(day, {})
        corridors_data = day_data.get("corridors", {})
        for corridor_id, stats in corridors_data.items():
            if not isinstance(stats, dict):
                continue

            sf_temp = stats.get("shortfall_temp_trucks", 0)
            sf_std = stats.get("shortfall_std_trucks", 0)
            sf_drv = stats.get("shortfall_drivers", 0)

            if sf_temp > 0:
                violations.append(
                    f"[{day}] {corridor_id}: cold-chain shortfall - "
                    f"{sf_temp} temp-controlled truck(s) unavailable. "
                    f"{stats.get('undelivered_units', '?')} units at risk."
                )
            if sf_std > 0:
                violations.append(
                    f"[{day}] {corridor_id}: standard truck shortfall - "
                    f"{sf_std} truck(s) unavailable."
                )
            if sf_drv > 0:
                violations.append(
                    f"[{day}] {corridor_id}: driver shortfall - "
                    f"{sf_drv} driver(s) unavailable; plan must defer or reassign."
                )

    for day in ["Day0", "Day1"]:
        corridors_data = allocation.get(day, {}).get("corridors", {})
        available_temp = allocation.get(day, {}).get("available", {}).get(
            "truck_temp_controlled",
            999,
        )
        total_assigned_temp = sum(
            c.get("allocated_temp_trucks", 0)
            for c in corridors_data.values()
            if isinstance(c, dict)
        )
        if total_assigned_temp > available_temp:
            violations.append(
                f"[{day}] Temp-controlled trucks over-assigned: "
                f"{total_assigned_temp} assigned but only {available_temp} available."
            )

    corridor_weather_risk = state.get("corridor_weather_risk", {})
    for corridor_id, risk in corridor_weather_risk.items():
        if risk.get("escalation_required"):
            corridor_token = corridor_id.lower().replace("_", "")
            plan_token = plan_text.replace("_", "").replace(" ", "")
            if corridor_token not in plan_token:
                violations.append(
                    f"Corridor {corridor_id} requires escalation "
                    f"(risk_score={risk.get('risk_score_48h')}) "
                    f"but is not explicitly addressed in the dispatch plan."
                )

    if violations:
        print(f"[validate] FAIL - {len(violations)} constraint violation(s):")
        for v in violations:
            print(f"  - {v}")
    else:
        print("[validate] PASS - all constraints satisfied.")

    current_retries = state.get("planner_retry_count", 0)
    return {
        "validation_violations": violations,
        "planner_retry_count": current_retries + (1 if violations else 0),
    }


def route_after_validate(state: AppState) -> str:
    violations = state.get("validation_violations", [])
    retry_count = state.get("planner_retry_count", 0)

    if violations and retry_count <= MAX_PLANNER_RETRIES:
        print(
            f"[route] Violations found - looping back to planner "
            f"(retry {retry_count}/{MAX_PLANNER_RETRIES})"
        )
        return "planner"

    if violations:
        print(
            f"[route] Max retries ({MAX_PLANNER_RETRIES}) reached - "
            f"proceeding to report with {len(violations)} unresolved violation(s)."
        )
    else:
        print("[route] Plan validated - proceeding to report.")
    return "report"


def _build_kpi_banner_html(state: AppState) -> str:
    allocation = state.get("resource_allocation", {})
    summary = allocation.get("summary_48h", {})
    weather_risk = state.get("corridor_weather_risk", {})

    penalty = int(summary.get("total_penalty_score", 0))
    tier1 = int(summary.get("tier1_units_impacted", 0))
    feasible = summary.get("allocation_feasible", True)

    escalation_count = sum(
        1 for v in weather_risk.values() if v.get("escalation_required")
    )

    sla_risk_count = 0
    for day in ["Day0", "Day1"]:
        corridors_data = allocation.get(day, {}).get("corridors", {})
        for _, stats in corridors_data.items():
            if isinstance(stats, dict) and stats.get("corridor_penalty", 0) > 0:
                sla_risk_count += 1

    def _card_color(val, thresholds):
        if val <= thresholds[0]:
            return ("#e8f5e9", "#2e7d32", "#a5d6a7")
        if val <= thresholds[1]:
            return ("#fff8e1", "#f57f17", "#ffe082")
        return ("#fce4ec", "#c62828", "#ef9a9a")

    penalty_colors = _card_color(penalty, (0, 200))
    tier1_colors = _card_color(tier1, (0, 5))
    feasible_colors = (
        ("#e8f5e9", "#2e7d32", "#a5d6a7")
        if feasible
        else ("#fce4ec", "#c62828", "#ef9a9a")
    )
    escalation_colors = _card_color(escalation_count, (0, 0))
    sla_colors = _card_color(sla_risk_count, (0, 2))

    cards = [
        ("TOTAL PENALTY SCORE (48H)", f"{penalty}", "pts" + (" - infeasible" if not feasible else ""), penalty_colors),
        ("TIER 1 UNITS IMPACTED", f"{tier1}", "Boston fully served" if tier1 == 0 else f"{tier1} unit(s) at risk", tier1_colors),
        ("ALLOCATION FEASIBILITY", "YES" if feasible else "NO", "all demand met" if feasible else "3PRC required", feasible_colors),
        ("CORRIDORS W/ ESCALATION", f"{escalation_count}", "weather threshold not met" if escalation_count == 0 else f"{escalation_count} corridor(s) flagged", escalation_colors),
        ("SLA RISK ITEMS", f"{sla_risk_count}", f"{sla_risk_count} corridor-day(s) with penalty" if sla_risk_count > 0 else "no SLA breaches", sla_colors),
    ]

    card_html_parts = []
    for label, value, subtitle, (bg, txt, border) in cards:
        card_html_parts.append(
            f'<div style="flex:1;min-width:140px;background:{bg};border-left:3px solid {border};'
            f'border-radius:4px;padding:10px 12px;text-align:center;">'
            f'<p style="margin:0 0 2px;font-size:10px;font-weight:700;letter-spacing:.04em;'
            f'text-transform:uppercase;color:#444;">{label}</p>'
            f'<p style="margin:0;font-size:24px;font-weight:700;color:{txt};line-height:1.2;">{value}</p>'
            f'<p style="margin:2px 0 0;font-size:10px;color:#555;">{subtitle}</p>'
            f'</div>'
        )

    return (
        '<div style="margin:0.5rem 0 1rem;">'
        '<h2 style="font-size:15px;font-weight:600;margin:0 0 8px;color:#1a1a2e;">'
        'KPI Banner</h2>'
        '<div style="display:flex;flex-wrap:wrap;gap:8px;">'
        + "".join(card_html_parts)
        + '</div></div>'
    )


def _build_sla_violations_html(state: AppState) -> str:
    allocation = state.get("resource_allocation", {})

    rows = []
    for day in ["Day0", "Day1"]:
        corridors_data = allocation.get(day, {}).get("corridors", {})
        for corridor_id, stats in corridors_data.items():
            if not isinstance(stats, dict):
                continue

            penalty = stats.get("corridor_penalty", 0)
            undelivered = stats.get("undelivered_units", 0)
            sf_temp = stats.get("shortfall_temp_trucks", 0)
            sf_std = stats.get("shortfall_std_trucks", 0)
            sf_drv = stats.get("shortfall_drivers", 0)
            sla_tier = stats.get("sla_tier", "-")
            wx_score = stats.get("weather_risk_score", 0)
            escalation = stats.get("escalation_required", False)

            if penalty > 0 or sf_temp > 0 or sf_std > 0 or sf_drv > 0 or escalation:
                reason_parts = []
                if sf_temp > 0:
                    reason_parts.append(f"temp-truck shortfall ({sf_temp})")
                if sf_std > 0:
                    reason_parts.append(f"std-truck shortfall ({sf_std})")
                if sf_drv > 0:
                    reason_parts.append(f"driver shortfall ({sf_drv})")
                if escalation:
                    reason_parts.append(f"weather escalation (score {wx_score})")

                rows.append({
                    "day": day,
                    "corridor": corridor_id,
                    "sla_tier": sla_tier,
                    "penalty": penalty,
                    "undelivered": undelivered,
                    "reason": "; ".join(reason_parts) if reason_parts else "penalty accrued",
                })

    if not rows:
        return (
            '<div style="margin:0.5rem 0 1rem;">'
            '<h2 style="font-size:15px;font-weight:600;margin:0 0 8px;color:#1a1a2e;">'
            'SLA Violation Corridors</h2>'
            '<p style="color:#2e7d32;font-weight:600;margin:0;font-size:13px;">'
            'No SLA violations detected across any corridor for the 48-hour window.</p>'
            '</div>'
        )

    header = (
        '<tr>'
        '<th style="padding:10px 14px;text-align:left;background:#b71c1c;color:#ffffff;font-weight:700;border:1px solid #999;">Day</th>'
        '<th style="padding:10px 14px;text-align:left;background:#b71c1c;color:#ffffff;font-weight:700;border:1px solid #999;">Corridor</th>'
        '<th style="padding:10px 14px;text-align:left;background:#b71c1c;color:#ffffff;font-weight:700;border:1px solid #999;">SLA Tier</th>'
        '<th style="padding:10px 14px;text-align:right;background:#b71c1c;color:#ffffff;font-weight:700;border:1px solid #999;">Penalty Pts</th>'
        '<th style="padding:10px 14px;text-align:right;background:#b71c1c;color:#ffffff;font-weight:700;border:1px solid #999;">Undelivered Units</th>'
        '<th style="padding:10px 14px;text-align:left;background:#b71c1c;color:#ffffff;font-weight:700;border:1px solid #999;">Root Cause</th>'
        '</tr>'
    )

    row_html = ""
    for r in rows:
        bg = "#fff0f0" if r["penalty"] > 0 else "#fffde7"
        row_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:10px 14px;color:#222;border:1px solid #ccc;">{r["day"]}</td>'
            f'<td style="padding:10px 14px;color:#222;font-weight:600;border:1px solid #ccc;">{r["corridor"]}</td>'
            f'<td style="padding:10px 14px;color:#222;border:1px solid #ccc;">{r["sla_tier"]}</td>'
            f'<td style="padding:10px 14px;text-align:right;font-weight:700;color:#b71c1c;border:1px solid #ccc;">{r["penalty"]}</td>'
            f'<td style="padding:10px 14px;text-align:right;color:#222;font-weight:600;border:1px solid #ccc;">{r["undelivered"]}</td>'
            f'<td style="padding:10px 14px;color:#333;font-size:13px;border:1px solid #ccc;">{r["reason"]}</td>'
            f'</tr>'
        )

    return (
        '<div style="margin:0.5rem 0 1rem;">'
        '<h2 style="font-size:15px;font-weight:600;margin:0 0 8px;color:#1a1a2e;">'
        'SLA Violation Corridors</h2>'
        '<table style="border-collapse:collapse;width:100%;background:#ffffff;">'
        + header + row_html
        + '</table></div>'
    )


def node_report(state: AppState) -> AppState:
    llm_html = run_report_agent(
        business_context=state.get("business_context", ""),
        kpis=state.get("csv_kpis", {}),
        anomaly_highlights=state.get("anomalies_md", "(none)"),
        weather_risk=state.get("corridor_weather_risk", {}),
        dispatch_plan=state.get("dispatch_plan", ""),
        resource_allocation=state.get("resource_allocation", {}),
        human_review_summary=state.get("human_review_summary", ""),
    )

    kpi_banner_html = _build_kpi_banner_html(state)
    sla_violations_html = _build_sla_violations_html(state)

    cleaned_llm = _clean_html(llm_html)
    cleaned_llm = cleaned_llm.replace("<!-- KPI_BANNER_PLACEHOLDER -->", "")
    cleaned_llm = cleaned_llm.replace("<!-- SLA_VIOLATIONS_PLACEHOLDER -->", "")

    h2_pattern = re.compile(r"(<h2[\s>])", re.IGNORECASE)
    h2_positions = [m.start() for m in h2_pattern.finditer(cleaned_llm)]

    if len(h2_positions) >= 2:
        insert_kpi_pos = h2_positions[1]
        cleaned_llm = (
            cleaned_llm[:insert_kpi_pos]
            + kpi_banner_html + "\n\n"
            + cleaned_llm[insert_kpi_pos:]
        )
        h2_positions = [m.start() for m in h2_pattern.finditer(cleaned_llm)]

    if len(h2_positions) >= 4:
        insert_sla_pos = h2_positions[3]
        cleaned_llm = (
            cleaned_llm[:insert_sla_pos]
            + sla_violations_html + "\n\n"
            + cleaned_llm[insert_sla_pos:]
        )
    else:
        cleaned_llm = cleaned_llm + "\n" + sla_violations_html

    if kpi_banner_html not in cleaned_llm:
        cleaned_llm = kpi_banner_html + "\n" + cleaned_llm

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SeeWeeS Dispatch Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 0 auto; padding: 2rem;
         color: #111 !important; background: #ffffff !important; }}
  div, section, header, article {{ background: transparent !important; color: #111 !important; }}
  h1 {{ font-size: 22px; font-weight: 700; color: #1a1a2e !important;
       margin: 0 0 4px; padding: 0 !important; background: transparent !important; }}
  h2 {{ font-size: 16px; font-weight: 600; color: #1a1a2e !important;
       margin: 1.2rem 0 0.5rem; background: transparent !important; }}
  h3 {{ font-weight: 600; color: #1a1a2e !important; background: transparent !important; }}
  p, li, span, strong, em, b {{ color: #222 !important; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0;
          background: #ffffff !important; }}
  th {{ background: #1a1a2e !important; color: #ffffff !important; font-weight: 700;
       font-size: 13px; padding: 10px 14px; text-align: left; border: 1px solid #2d2d4a;
       white-space: nowrap; }}
  td {{ padding: 10px 14px; text-align: left; border: 1px solid #dde;
       color: #222 !important; font-size: 13px; background: #fafbfc !important;
       white-space: nowrap; }}
  tr:nth-child(even) td {{ background: #f0f2f5 !important; }}
</style>
</head>
<body>
{cleaned_llm}
</body>
</html>"""

    return {"report_html": full_html}


def node_email(state: AppState) -> AppState:
    to_email = os.getenv("REPORT_EMAIL_TO", "").strip()
    if not to_email:
        print("REPORT_EMAIL_TO not set - skipping email send.")
        return {}

    subject = "MSBA Ops Multi-Agent Dispatch Report - Multi-Corridor"
    send_email_smtp(subject=subject, html_body=state["report_html"], to_email=to_email)
    return {}


def route_after_human_review(state: AppState) -> str:
    review = state.get("human_review", {})
    return "retry" if review.get("decision") == "retry" else "approved"


def _apply_resource_reserve(
    availability: Dict[str, Dict[str, int]],
    reserve: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    adjusted = {day: dict(resources) for day, resources in availability.items()}
    for day, resources in reserve.items():
        adjusted.setdefault(day, {})
        for resource_type, count in resources.items():
            current = int(adjusted[day].get(resource_type, 0))
            adjusted[day][resource_type] = max(0, current - int(count))
    return adjusted


def build_graph():
    g = StateGraph(AppState)

    g.add_node("pdf_context", node_pdf_context)
    g.add_node("csv_analysis", node_csv_analysis)
    g.add_node("router", node_router)
    g.add_node("node_weather_corridor", node_weather_corridor)
    g.add_node("collect_weather", node_collect_weather)
    g.add_node("resource_allocator", node_resource_allocator)
    g.add_node("human_review_node", node_human_review)
    g.add_node("apply_retry_policy", node_apply_retry_policy)
    g.add_node("planner", node_planner)
    g.add_node("validate", node_validate)
    g.add_node("report", node_report)
    g.add_node("email", node_email)

    g.add_edge(START, "pdf_context")
    g.add_edge(START, "csv_analysis")

    g.add_edge("pdf_context", "router")
    g.add_edge("csv_analysis", "router")

    g.add_conditional_edges("router", route_to_weather_corridors, ["node_weather_corridor"])
    g.add_edge("node_weather_corridor", "collect_weather")

    g.add_edge("collect_weather", "resource_allocator")
    g.add_edge("resource_allocator", "human_review_node")
    g.add_conditional_edges(
        "human_review_node",
        route_after_human_review,
        {
            "approved": "planner",
            "retry": "apply_retry_policy",
        },
    )

    g.add_edge("apply_retry_policy", "resource_allocator")
    g.add_edge("planner", "validate")

    g.add_conditional_edges(
        "validate",
        route_after_validate,
        {"planner": "planner", "report": "report"},
    )

    g.add_edge("report", "email")
    g.add_edge("email", END)

    return g.compile()
