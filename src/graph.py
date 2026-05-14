from __future__ import annotations
import os
import re
import operator
from typing import TypedDict, Dict, Any, List, Annotated

from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from dotenv import load_dotenv

from tools.pdf_tools import PdfRag
### Harish_added
from tools.csv_tools import analyze_csv, analyze_csv_with_overrides, detect_planning_special_case_items
from tools.weather_tools import get_single_corridor_weather_risk, CORRIDOR_WAYPOINTS
from tools.resource_tools import load_resource_availability, allocate_resources
# Harish_added
from tools.human_review_tools import (
    collect_human_review_decision,
    detect_human_review_triggers,
    summarize_human_review,
)
from tools.email_tools import send_email_smtp
from agents import (
    run_context_agent,
    run_ops_agent,
    run_planner_agent,
    run_planner_retry_agent,
    run_report_agent,
)

load_dotenv()

MAX_PLANNER_RETRIES = 2


# ---------------------------------------------------------------------------
# Shared application state
# NOTE: corridor_weather_results uses operator.add as a reducer so that
#       results from parallel weather nodes are appended, not overwritten.
# ---------------------------------------------------------------------------
class AppState(TypedDict, total=False):
    # Inputs
    pdf_path:              str
    csv_path:              str
    resource_path:         str
    corridor_id:           str   # injected per-corridor by the Send fan-out

    # PDF context node
    business_context:      str

    # CSV analysis node
    csv_summary:           Dict[str, Any]
    csv_kpis:              Dict[str, Any]
    corridor_day_summary:  Dict[str, Any]
    anomalies_md:          str
    ops_insights:          str

    # Parallel weather accumulator — reducer appends lists across concurrent nodes
    corridor_weather_results: Annotated[List[Dict[str, Any]], operator.add]

    # Aggregated corridor weather (built by collect_weather after fan-in)
    corridor_weather_risk: Dict[str, Any]
    
    ## Harish_added
    human_review: Dict[str, Any]
    human_review_summary: str
    human_review_approvals: List[str]

    # Harish_added
    retry_policy: Dict[str, Any]
    retry_count: int

    # Resource allocator node
    resource_allocation:   Dict[str, Any]
    # Harish_added
    special_case_items: Dict[str, Any]
    resource_reserve: Dict[str, Dict[str, int]]

    # Validation loop
    validation_violations: List[str]
    planner_retry_count:   int

    # Planner + report nodes
    dispatch_plan:         str
    report_html:           str


# ---------------------------------------------------------------------------
# Helper: strip markdown code fences that LLMs sometimes wrap HTML in
# ---------------------------------------------------------------------------
def _clean_html(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:html)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ===========================================================================
# CHANGE 1: PARALLEL FAN-OUT AT TOP
# pdf_context and csv_analysis both connect FROM START — they run concurrently.
# LangGraph waits for both to finish before the router node executes.
# ===========================================================================

# ---------------------------------------------------------------------------
# Node 1a: PDF context  (runs in parallel with csv_analysis)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Node 1b: CSV analysis  (runs in parallel with pdf_context)
# ---------------------------------------------------------------------------
def node_csv_analysis(state: AppState) -> AppState:
    print("[csv_analysis] Analysing multi-corridor shipment CSV...")
    res = analyze_csv(state["csv_path"])
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
        "csv_summary":          res.summary,
        "csv_kpis":             res.kpis,
        "corridor_day_summary": res.corridor_day_summary,
        "anomalies_md":         anomalies_md,
        "ops_insights":         ops_insights,
        "special_case_items": special_case_items,
    }


# ===========================================================================
# CHANGE 2: MULTI-REGION WEATHER FAN-OUT VIA Send API
# The router node is a pass-through; the conditional edge function
# route_to_weather_corridors returns one Send per corridor, causing
# LangGraph to dispatch node_weather_corridor in parallel for each.
# Results accumulate in corridor_weather_results via the operator.add reducer,
# then collect_weather aggregates them into corridor_weather_risk.
# ===========================================================================

# ---------------------------------------------------------------------------
# Node 2: Router  (fan-in point for pdf + csv; triggers weather fan-out)
# ---------------------------------------------------------------------------
def node_router(state: AppState) -> AppState:
    corridors = list(CORRIDOR_WAYPOINTS.keys())
    print(f"[router] pdf_context + csv_analysis complete. "
          f"Fanning out weather fetch to {len(corridors)} corridors: {corridors}")
    return {}   # state already complete; fan-out handled by conditional edge below


# Conditional edge function — returns List[Send] to trigger parallel execution
def route_to_weather_corridors(state: AppState) -> List[Send]:
    return [
        Send("node_weather_corridor", {**state, "corridor_id": corridor_id})
        for corridor_id in CORRIDOR_WAYPOINTS
    ]


# ---------------------------------------------------------------------------
# Node 3: Per-corridor weather  (one instance per corridor, run in parallel)
# ---------------------------------------------------------------------------
def node_weather_corridor(state: AppState) -> AppState:
    corridor_id = state["corridor_id"]
    print(f"  [weather_corridor] {corridor_id}: fetching waypoint forecasts...")
    risk = get_single_corridor_weather_risk(corridor_id)
    print(f"  [weather_corridor] {corridor_id}: risk={risk.get('risk_score_48h', '?')} "
          f"| buffer={risk.get('travel_buffer_pct', '?')}% "
          f"| escalation={risk.get('escalation_required', '?')}")
    # Appended to list by operator.add reducer — safe for parallel execution
    return {"corridor_weather_results": [{"corridor_id": corridor_id, "risk": risk}]}


# ---------------------------------------------------------------------------
# Node 4: Collect weather  (fan-in — runs after ALL weather_corridor nodes done)
# ---------------------------------------------------------------------------
def node_collect_weather(state: AppState) -> AppState:
    results = state.get("corridor_weather_results", [])
    corridor_weather_risk: Dict[str, Any] = {}
    for entry in results:
        corridor_weather_risk[entry["corridor_id"]] = entry["risk"]
    print(f"[collect_weather] Aggregated weather risk for {len(corridor_weather_risk)} corridors.")
    return {"corridor_weather_risk": corridor_weather_risk}


# ---------------------------------------------------------------------------
# Node 5: Resource allocator
# ---------------------------------------------------------------------------
def node_resource_allocator(state: AppState) -> AppState:
    resource_path = state.get(
        "resource_path",
        "data-for-enhancement/Resource_availability_48h.csv",
    )
    print("[resource_allocator] Loading resource availability...")
    availability = load_resource_availability(resource_path)
    availability = _apply_resource_reserve(availability, state.get("resource_reserve", {}))

    print("[resource_allocator] Running greedy allocation with penalty model...")
    ## Harish_added
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

## Harish_added
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
    human_review_approvals = (
        state.get("human_review_approvals", [])
        + review.get("approvals", [])
    )
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

## Harish_added
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
        ##  Harish_added(GPT suggested to clean up validation memory)
        "validation_violations": [],
        "planner_retry_count": 0,

    }

# ---------------------------------------------------------------------------
# Node 6: Planner  (also used on retry after validation failure)
# ---------------------------------------------------------------------------
def node_planner(state: AppState) -> AppState:
    retry_count = state.get("planner_retry_count", 0)
    violations  = state.get("validation_violations", [])

    if retry_count > 0 and violations:
        print(f"[planner] Retry #{retry_count} — revising plan to address "
              f"{len(violations)} constraint violation(s)...")
        plan = run_planner_retry_agent(
            business_context=state.get("business_context", ""),
            ops_insights=state.get("ops_insights", ""),
            weather_risk=state.get("corridor_weather_risk", {}),
            resource_allocation=state.get("resource_allocation", {}),
            ##Harish_added
            human_review_summary=state.get("human_review_summary", ""),
            violations=violations,
        )
    else:
        print("[planner] Generating initial multi-corridor dispatch plan...")
        plan = run_planner_agent(
            business_context=state.get("business_context", ""),
            ops_insights=state.get("ops_insights", ""),
            weather_risk=state.get("corridor_weather_risk", {}),
            ##Harish_added
            human_review_summary=state.get("human_review_summary", ""),
            resource_allocation=state.get("resource_allocation", {}),
        )

    return {"dispatch_plan": plan}


# ===========================================================================
# CHANGE 3: VALIDATION LOOP WITH CONDITIONAL EDGE
# node_validate checks three constraint classes. If any fail and we haven't
# hit MAX_PLANNER_RETRIES, route_after_validate sends the plan back to
# node_planner with the violation list. Otherwise proceed to report.
# ===========================================================================

# ---------------------------------------------------------------------------
# Node 7: Validate  (constraint checker)
# ---------------------------------------------------------------------------
def node_validate(state: AppState) -> AppState:
    violations: List[str] = []
    allocation = state.get("resource_allocation", {})
    plan_text  = state.get("dispatch_plan", "").lower()

    # --- Check 1: resource shortfalls (trucks / drivers) ---
    for day in ["Day0", "Day1"]:
        day_data = allocation.get(day, {})
        corridors_data = day_data.get("corridors", {})
        for corridor_id, stats in corridors_data.items():
            if not isinstance(stats, dict):
                continue

            sf_temp = stats.get("shortfall_temp_trucks", 0)
            sf_std  = stats.get("shortfall_std_trucks", 0)
            sf_drv  = stats.get("shortfall_drivers", 0)

            if sf_temp > 0:
                violations.append(
                    f"[{day}] {corridor_id}: cold-chain shortfall — "
                    f"{sf_temp} temp-controlled truck(s) unavailable. "
                    f"{stats.get('undelivered_units', '?')} units at risk."
                )
            if sf_std > 0:
                violations.append(
                    f"[{day}] {corridor_id}: standard truck shortfall — "
                    f"{sf_std} truck(s) unavailable."
                )
            if sf_drv > 0:
                violations.append(
                    f"[{day}] {corridor_id}: driver shortfall — "
                    f"{sf_drv} driver(s) unavailable; plan must defer or reassign."
                )

    # --- Check 2: cold-chain over-assignment across corridors on the same day ---
    for day in ["Day0", "Day1"]:
        corridors_data = allocation.get(day, {}).get("corridors", {})
        available_temp = allocation.get(day, {}).get("available", {}).get(
            "truck_temp_controlled", 999
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

    # --- Check 3: high-risk corridors must be explicitly addressed in the plan ---
    corridor_weather_risk = state.get("corridor_weather_risk", {})
    for corridor_id, risk in corridor_weather_risk.items():
        if risk.get("escalation_required"):
            # Plan text must mention the corridor name (flexible match)
            corridor_token = corridor_id.lower().replace("_", "")
            plan_token     = plan_text.replace("_", "").replace(" ", "")
            if corridor_token not in plan_token:
                violations.append(
                    f"Corridor {corridor_id} requires escalation "
                    f"(risk_score={risk.get('risk_score_48h')}) "
                    f"but is not explicitly addressed in the dispatch plan."
                )

    # Log results
    if violations:
        print(f"[validate] FAIL — {len(violations)} constraint violation(s):")
        for v in violations:
            print(f"  • {v}")
    else:
        print("[validate] PASS — all constraints satisfied.")

    # Increment retry counter only when violations exist
    current_retries = state.get("planner_retry_count", 0)
    return {
        "validation_violations": violations,
        "planner_retry_count":   current_retries + (1 if violations else 0),
    }


# Conditional edge function — decides where to go after validate
def route_after_validate(state: AppState) -> str:
    violations  = state.get("validation_violations", [])
    retry_count = state.get("planner_retry_count", 0)

    if violations and retry_count <= MAX_PLANNER_RETRIES:
        print(f"[route] Violations found — looping back to planner "
              f"(retry {retry_count}/{MAX_PLANNER_RETRIES})")
        return "planner"

    if violations:
        print(f"[route] Max retries ({MAX_PLANNER_RETRIES}) reached — "
              f"proceeding to report with {len(violations)} unresolved violation(s).")
    else:
        print("[route] Plan validated — proceeding to report.")
    return "report"


# ---------------------------------------------------------------------------
# Deterministic HTML builders — no LLM, no hallucinated markup
# ---------------------------------------------------------------------------

def _build_kpi_banner_html(state: AppState) -> str:
    """Build the KPI banner row from live state data. Returns clean HTML."""
    allocation = state.get("resource_allocation", {})
    summary = allocation.get("summary_48h", {})
    weather_risk = state.get("corridor_weather_risk", {})
    kpis = state.get("csv_kpis", {})
    violations = state.get("validation_violations", [])

    penalty = int(summary.get("total_penalty_score", 0))
    tier1 = int(summary.get("tier1_units_impacted", 0))
    feasible = summary.get("allocation_feasible", True)

    # Count corridors with escalation
    escalation_count = sum(
        1 for v in weather_risk.values() if v.get("escalation_required")
    )

    # Count SLA risk items: corridors×days where penalty > 0
    sla_risk_count = 0
    for day in ["Day0", "Day1"]:
        corridors_data = allocation.get(day, {}).get("corridors", {})
        for cid, stats in corridors_data.items():
            if isinstance(stats, dict) and stats.get("corridor_penalty", 0) > 0:
                sla_risk_count += 1

    # Colour helpers
    def _card_color(val, thresholds):
        """Return (bg, text, border) based on value vs thresholds."""
        if val <= thresholds[0]:
            return ("#e8f5e9", "#2e7d32", "#a5d6a7")   # green
        elif val <= thresholds[1]:
            return ("#fff8e1", "#f57f17", "#ffe082")    # amber
        return ("#fce4ec", "#c62828", "#ef9a9a")        # red

    penalty_colors = _card_color(penalty, (0, 200))
    tier1_colors = _card_color(tier1, (0, 5))
    feasible_colors = ("#e8f5e9", "#2e7d32", "#a5d6a7") if feasible else ("#fce4ec", "#c62828", "#ef9a9a")
    escalation_colors = _card_color(escalation_count, (0, 0))
    sla_colors = _card_color(sla_risk_count, (0, 2))

    cards = [
        ("TOTAL PENALTY SCORE (48H)", f"{penalty}", "pts" + (" — infeasible" if not feasible else ""), penalty_colors),
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
    """Build the SLA Violation Corridors section from allocation data."""
    allocation = state.get("resource_allocation", {})
    weather_risk = state.get("corridor_weather_risk", {})

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
            sla_tier = stats.get("sla_tier", "—")
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


# ---------------------------------------------------------------------------
# Node 8: Report
# ---------------------------------------------------------------------------
def node_report(state: AppState) -> AppState:
    llm_html = run_report_agent(
        business_context=state.get("business_context", ""),
        kpis=state.get("csv_kpis", {}),
        anomaly_highlights=state.get("anomalies_md", "(none)"),
        weather_risk=state.get("corridor_weather_risk", {}),
        dispatch_plan=state.get("dispatch_plan", ""),
        resource_allocation=state.get("resource_allocation", {}),
        ## Harish_added
        human_review_summary=state.get("human_review_summary", ""),
    )

    # Build deterministic sections (no LLM hallucination)
    kpi_banner_html = _build_kpi_banner_html(state)
    sla_violations_html = _build_sla_violations_html(state)

    cleaned_llm = _clean_html(llm_html)

    # Strip any placeholders the LLM may have emitted (they could land
    # inside table cells or other bad locations — don't rely on them).
    cleaned_llm = cleaned_llm.replace("<!-- KPI_BANNER_PLACEHOLDER -->", "")
    cleaned_llm = cleaned_llm.replace("<!-- SLA_VIOLATIONS_PLACEHOLDER -->", "")

    # Inject KPI banner after the first </table> or first </div> that closes
    # the executive summary, and SLA violations after the corridor risk dashboard.
    # Strategy: find the section headings in the LLM HTML and inject after them.
    # We search for common heading patterns that mark section boundaries.

    # Split LLM HTML into sections at <h2 boundaries for reliable injection
    import re as _re
    h2_pattern = _re.compile(r'(<h2[\s>])', _re.IGNORECASE)
    h2_positions = [m.start() for m in h2_pattern.finditer(cleaned_llm)]

    if len(h2_positions) >= 2:
        # Insert KPI banner before the 2nd <h2> (after Executive Summary)
        insert_kpi_pos = h2_positions[1]
        cleaned_llm = (
            cleaned_llm[:insert_kpi_pos]
            + kpi_banner_html + "\n\n"
            + cleaned_llm[insert_kpi_pos:]
        )
        # Re-find positions after insertion
        h2_positions = [m.start() for m in h2_pattern.finditer(cleaned_llm)]

    if len(h2_positions) >= 4:
        # Insert SLA violations before the 4th <h2> (after Corridor Risk Dashboard)
        # Sections after KPI injection: 1=Exec, 2=KPI(injected h2), 3=Corridor Risk, 4=Resource Alloc
        # We want SLA violations between Corridor Risk Dashboard and Resource Allocation
        insert_sla_pos = h2_positions[3]
        cleaned_llm = (
            cleaned_llm[:insert_sla_pos]
            + sla_violations_html + "\n\n"
            + cleaned_llm[insert_sla_pos:]
        )
    else:
        # Fallback: append at end
        cleaned_llm = cleaned_llm + "\n" + sla_violations_html

    # If KPI banner wasn't inserted (< 2 h2 tags), prepend it
    if kpi_banner_html not in cleaned_llm:
        cleaned_llm = kpi_banner_html + "\n" + cleaned_llm

    # Wrap in a clean HTML document
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SeeWeeS Dispatch Report</title>
<style>
  /* Force light background everywhere — override any LLM-generated dark banners */
  body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 0 auto; padding: 2rem;
         color: #111 !important; background: #ffffff !important; }}
  div, section, header, article {{ background: transparent !important; color: #111 !important; }}
  h1 {{ font-size: 22px; font-weight: 700; color: #1a1a2e !important;
       margin: 0 0 4px; padding: 0 !important; background: transparent !important; }}
  h2 {{ font-size: 16px; font-weight: 600; color: #1a1a2e !important;
       margin: 1.2rem 0 0.5rem; background: transparent !important; }}
  h3 {{ font-weight: 600; color: #1a1a2e !important; background: transparent !important; }}
  p, li, span, strong, em, b {{ color: #222 !important; }}

  /* Tables — dark header, light rows */
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


# ---------------------------------------------------------------------------
# Node 9: Email
# ---------------------------------------------------------------------------
def node_email(state: AppState) -> AppState:
    to_email = os.getenv("REPORT_EMAIL_TO", "").strip()
    if not to_email:
        print("REPORT_EMAIL_TO not set — skipping email send.")
        return {}

    subject = "MSBA Ops Multi-Agent Dispatch Report — Multi-Corridor"
    send_email_smtp(subject=subject, html_body=state["report_html"], to_email=to_email)
    return {}

## Harish_added
def route_after_human_review(state: AppState) -> str:
    review = state.get("human_review", {})
    return "retry" if review.get("decision") == "retry" else "approved"

## Harish_added
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

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(AppState)

    # Register all nodes
    g.add_node("pdf_context",           node_pdf_context)
    g.add_node("csv_analysis",          node_csv_analysis)
    g.add_node("router",                node_router)
    g.add_node("node_weather_corridor", node_weather_corridor)
    g.add_node("collect_weather",       node_collect_weather)
    g.add_node("resource_allocator",    node_resource_allocator)
    g.add_node("planner",               node_planner)
    ##Harish_added
    g.add_node("human_review", node_human_review)
    g.add_node("apply_retry_policy", node_apply_retry_policy)
    g.add_node("validate",              node_validate)
    g.add_node("report",                node_report)
    g.add_node("email",                 node_email)

    # ── CHANGE 1: Parallel fan-out from START ────────────────────────────────
    g.add_edge(START, "pdf_context")    # \  run concurrently;
    g.add_edge(START, "csv_analysis")  # /  LangGraph waits for both

    # Both converge at router (executed only after both predecessors finish)
    g.add_edge("pdf_context",  "router")
    g.add_edge("csv_analysis", "router")

    # ── CHANGE 2: Multi-region weather fan-out via Send API ──────────────────
    # route_to_weather_corridors returns List[Send], one per corridor.
    # LangGraph dispatches node_weather_corridor in parallel for each Send.
    g.add_conditional_edges("router", route_to_weather_corridors, ["node_weather_corridor"])

    # All parallel weather_corridor instances converge at collect_weather
    g.add_edge("node_weather_corridor", "collect_weather")

    # Sequential path: collect → allocate → plan → validate
    g.add_edge("collect_weather",    "resource_allocator")
    ## Harish_added
    g.add_edge("resource_allocator", "human_review")
    g.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "approved": "planner",
            "retry": "apply_retry_policy",
        },
    )
    g.add_edge("apply_retry_policy", "resource_allocator")
    g.add_edge("planner",            "validate")

    # ── CHANGE 3: Validation loop (conditional edge) ─────────────────────────
    # route_after_validate returns "planner" (retry) or "report" (pass)
    g.add_conditional_edges(
        "validate",
        route_after_validate,
        {"planner": "planner", "report": "report"},
    )

    g.add_edge("report", "email")
    g.add_edge("email",  END)

    return g.compile()
