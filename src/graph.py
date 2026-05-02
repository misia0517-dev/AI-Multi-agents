from __future__ import annotations
import os
import re
from typing import TypedDict, Dict, Any

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from tools.pdf_tools import PdfRag
from tools.csv_tools import analyze_csv
from tools.weather_tools import get_all_corridors_weather_risk
from tools.resource_tools import load_resource_availability, allocate_resources
from tools.email_tools import send_email_smtp
from agents import run_context_agent, run_ops_agent, run_planner_agent, run_report_agent

load_dotenv()


# ---------------------------------------------------------------------------
# Shared application state — passed between all nodes
# ---------------------------------------------------------------------------
class AppState(TypedDict, total=False):
    # Inputs
    pdf_path:              str
    csv_path:              str
    resource_path:         str

    # PDF context node
    business_context:      str

    # CSV analysis node
    csv_summary:           Dict[str, Any]
    csv_kpis:              Dict[str, Any]
    corridor_day_summary:  Dict[str, Any]
    anomalies_md:          str
    ops_insights:          str

    # Weather node (multi-corridor)
    corridor_weather_risk: Dict[str, Any]

    # Resource allocator node
    resource_allocation:   Dict[str, Any]

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


# ---------------------------------------------------------------------------
# Node 1: PDF context
# ---------------------------------------------------------------------------
def node_pdf_context(state: AppState) -> AppState:
    rag = PdfRag(persist_dir="chroma_db")
    vectordb = rag.build(state["pdf_path"])
    retriever = rag.retriever(vectordb, k=6)

    query = "Extract KPI definitions, thresholds, SLAs, constraints, dispatch rules, exceptions."
    docs = retriever.invoke(query)
    snippets = "\n\n---\n\n".join(d.page_content for d in docs)

    business_context = run_context_agent(snippets)
    return {"business_context": business_context}


# ---------------------------------------------------------------------------
# Node 2: CSV analysis
# ---------------------------------------------------------------------------
def node_csv_analysis(state: AppState) -> AppState:
    res = analyze_csv(state["csv_path"])

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
    }


# ---------------------------------------------------------------------------
# Node 3: Weather — fetches per corridor (multi-waypoint)
# ---------------------------------------------------------------------------
def node_weather(state: AppState) -> AppState:
    print("[weather] Fetching weather for all corridor waypoints...")
    corridor_weather_risk = get_all_corridors_weather_risk()

    for corridor_id, risk in corridor_weather_risk.items():
        print(
            f"  {corridor_id}: 48h risk={risk['risk_score_48h']} "
            f"| buffer={risk['travel_buffer_pct']}% "
            f"| escalation={risk['escalation_required']}"
        )

    return {"corridor_weather_risk": corridor_weather_risk}


# ---------------------------------------------------------------------------
# Node 4: Resource Allocator (Option 5)
# ---------------------------------------------------------------------------
def node_resource_allocator(state: AppState) -> AppState:
    resource_path = state.get(
        "resource_path",
        "data-for-enhancement/Resource_availability_48h.csv",
    )

    print("[resource_allocator] Loading resource availability...")
    availability = load_resource_availability(resource_path)

    print("[resource_allocator] Running allocation with penalty model...")
    allocation = allocate_resources(
        corridor_day_summary=state.get("corridor_day_summary", {}),
        availability=availability,
        corridor_weather_risk=state.get("corridor_weather_risk", {}),
    )

    summary = allocation.get("summary_48h", {})
    print(
        f"  Total penalty score: {summary.get('total_penalty_score', '?')} | "
        f"Feasible: {summary.get('allocation_feasible', '?')}"
    )

    return {"resource_allocation": allocation}


# ---------------------------------------------------------------------------
# Node 5: Planner
# ---------------------------------------------------------------------------
def node_planner(state: AppState) -> AppState:
    plan = run_planner_agent(
        business_context=state.get("business_context", ""),
        ops_insights=state.get("ops_insights", ""),
        weather_risk=state.get("corridor_weather_risk", {}),
        resource_allocation=state.get("resource_allocation", {}),
    )
    return {"dispatch_plan": plan}


# ---------------------------------------------------------------------------
# Node 6: Report
# ---------------------------------------------------------------------------
def node_report(state: AppState) -> AppState:
    html = run_report_agent(
        business_context=state.get("business_context", ""),
        kpis=state.get("csv_kpis", {}),
        anomaly_highlights=state.get("anomalies_md", "(none)"),
        weather_risk=state.get("corridor_weather_risk", {}),
        dispatch_plan=state.get("dispatch_plan", ""),
        resource_allocation=state.get("resource_allocation", {}),
    )
    return {"report_html": _clean_html(html)}


# ---------------------------------------------------------------------------
# Node 7: Email
# ---------------------------------------------------------------------------
def node_email(state: AppState) -> AppState:
    to_email = os.getenv("REPORT_EMAIL_TO", "").strip()
    if not to_email:
        print("REPORT_EMAIL_TO not set -> skipping email send.")
        return {}

    subject = "MSBA Ops Multi-Agent Dispatch Report — Multi-Corridor"
    send_email_smtp(subject=subject, html_body=state["report_html"], to_email=to_email)
    return {}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(AppState)

    g.add_node("pdf_context",        node_pdf_context)
    g.add_node("csv_analysis",       node_csv_analysis)
    g.add_node("weather",            node_weather)
    g.add_node("resource_allocator", node_resource_allocator)
    g.add_node("planner",            node_planner)
    g.add_node("report",             node_report)
    g.add_node("email",              node_email)

    g.set_entry_point("pdf_context")
    g.add_edge("pdf_context",        "csv_analysis")
    g.add_edge("csv_analysis",       "weather")
    g.add_edge("weather",            "resource_allocator")
    g.add_edge("resource_allocator", "planner")
    g.add_edge("planner",            "report")
    g.add_edge("report",             "email")
    g.add_edge("email",              END)

    return g.compile()
