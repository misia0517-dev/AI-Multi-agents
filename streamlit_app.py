from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

_MISSING_ANTHROPIC_API_KEY = "missing-key-for-streamlit-startup"
if not os.getenv("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = _MISSING_ANTHROPIC_API_KEY

from tracing import init_langsmith_tracing

init_langsmith_tracing()

from graph import (  # noqa: E402
    MAX_PLANNER_RETRIES,
    _active_corridor_ids,
    node_apply_retry_policy,
    node_collect_weather,
    node_csv_analysis,
    node_email,
    node_pdf_context,
    node_planner,
    node_report,
    node_resource_allocator,
    node_validate,
    node_weather_corridor,
    route_after_validate,
)
from tools.human_review_tools import (  # noqa: E402
    MAX_RETRY_COUNT,
    detect_human_review_triggers,
    summarize_human_review,
)


DEFAULT_PDF = "data-for-enhancement/playbook_variant_02_weather_escalation_score2.md"
DEFAULT_CSV = "data-for-enhancement/Incoming_shipments_14d_multi_corridor.csv"
DEFAULT_RESOURCES = "data-for-enhancement/Resource_availability_48h.csv"


st.set_page_config(
    page_title="SeeWeeS Agentic Dispatch",
    page_icon="",
    layout="wide",
)


st.markdown(
    """
    <style>
      .main .block-container { padding-top: 1.25rem; max-width: 1280px; }
      .stage-card {
        border: 1px solid #d8dee9;
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
        min-height: 88px;
      }
      .metric-card {
        border-left: 4px solid #2f80ed;
        background: #f8fafc;
        border-radius: 8px;
        padding: 12px 14px;
      }
      .risk-card {
        border: 1px solid #f1c27d;
        border-left: 4px solid #f2994a;
        background: #fffaf3;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 10px 0;
      }
      .ok-card {
        border: 1px solid #b7dfc2;
        border-left: 4px solid #219653;
        background: #f4fbf6;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 10px 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _new_state(pdf_path: str, csv_path: str, resource_path: str) -> Dict[str, Any]:
    return {
        "pdf_path": pdf_path,
        "csv_path": csv_path,
        "resource_path": resource_path,
        "human_review_approvals": [],
        "human_review_summary": "",
        "retry_count": 0,
        "planner_retry_count": 0,
        "validation_violations": [],
        "resource_reserve": {},
    }


def _run_upstream(state: Dict[str, Any]) -> Dict[str, Any]:
    state.update(node_pdf_context(state))
    state.update(node_csv_analysis(state))

    weather_results: List[Dict[str, Any]] = []
    for corridor_id in _active_corridor_ids(state):
        result = node_weather_corridor({**state, "corridor_id": corridor_id})
        weather_results.extend(result.get("corridor_weather_results", []))

    state["corridor_weather_results"] = weather_results
    state.update(node_collect_weather(state))
    state.update(node_resource_allocator(state))
    return state


def _current_triggers(state: Dict[str, Any]) -> Dict[str, Any]:
    return detect_human_review_triggers(
        special_case_items=state.get("special_case_items", {}),
        resource_allocation=state.get("resource_allocation", {}),
        resource_reserve=state.get("resource_reserve", {}),
    )


def _review_not_required(state: Dict[str, Any], triggers: Dict[str, Any]) -> Dict[str, Any]:
    review = {
        "decision": "approved",
        "reason": "No new human-review business triggers detected.",
        "retry_policy": {},
        "triggers": triggers,
        "approvals": [],
    }
    state["human_review"] = review
    state["human_review_summary"] = summarize_human_review(
        review=review,
        previous_summary=state.get("human_review_summary", ""),
    )
    return state


def _apply_review(state: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
    state["human_review"] = review
    state["human_review_approvals"] = (
        state.get("human_review_approvals", []) + review.get("approvals", [])
    )
    state["human_review_summary"] = summarize_human_review(
        review=review,
        previous_summary=state.get("human_review_summary", ""),
    )
    state["retry_policy"] = review.get("retry_policy", {})

    if review.get("decision") == "retry":
        state.update(node_apply_retry_policy(state))
        state.update(node_resource_allocator(state))

    return state


def _build_review_from_form(triggers: Dict[str, Any], retry_count: int) -> Dict[str, Any]:
    retry_policy: Dict[str, Any] = {}
    approvals: List[str] = []

    special = triggers.get("special_case_items")
    if special:
        special_choice = st.session_state.get("review_special_case")
        if special_choice == "Hold/quarantine and retry":
            retry_policy["hold_item_ids"] = special.get("items", [])
        else:
            approvals.append("special_case_items_approved")

    hold_corridors = set()
    for idx, item in enumerate(triggers.get("weather_escalation_corridors", [])):
        key = f"review_weather_{idx}"
        choice = st.session_state.get(key)
        if choice == "Hold corridor and retry":
            hold_corridors.add(item["corridor_id"])
        else:
            approvals.append(
                f"weather_escalation_approved:{item['day']}:{item['corridor_id']}"
            )

    if hold_corridors:
        retry_policy["hold_corridors"] = sorted(hold_corridors)

    reserve_by_day: Dict[str, Dict[str, int]] = {}
    for idx, item in enumerate(triggers.get("zero_temp_buffer_days", [])):
        key = f"review_temp_{idx}"
        day = str(item["day"])
        choice = st.session_state.get(key)
        if choice == "Reserve 1 temp truck and retry":
            reserve_by_day.setdefault(day, {})["truck_temp_controlled"] = 1
        else:
            approvals.append(f"zero_temp_buffer_approved:{day}")

    if reserve_by_day:
        retry_policy["resource_reserve"] = reserve_by_day

    if retry_policy and retry_count >= MAX_RETRY_COUNT:
        return {
            "decision": "approved",
            "reason": "Retry limit reached; proceeding after review.",
            "retry_policy": {},
            "triggers": triggers,
            "approvals": approvals,
        }

    return {
        "decision": "retry" if retry_policy else "approved",
        "reason": "Human selected retry controls." if retry_policy else "Human approved triggered risks.",
        "retry_policy": retry_policy,
        "triggers": triggers,
        "approvals": approvals,
    }


def _run_planner_to_report(state: Dict[str, Any], send_email: bool) -> Dict[str, Any]:
    while True:
        state.update(node_planner(state))
        state.update(node_validate(state))

        route = route_after_validate(state)
        if route != "planner":
            break

        if state.get("planner_retry_count", 0) > MAX_PLANNER_RETRIES:
            break

    state.update(node_report(state))

    if send_email:
        state.update(node_email(state))

    return state


def _render_stage_status(state: Dict[str, Any]) -> None:
    stages = [
        ("Context", "business_context" in state),
        ("CSV", "csv_kpis" in state),
        ("Weather", "corridor_weather_risk" in state),
        ("Resources", "resource_allocation" in state),
        ("Review", "human_review" in state),
        ("Planner", "dispatch_plan" in state),
        ("Validate", "validation_violations" in state and "dispatch_plan" in state),
        ("Report", "report_html" in state),
    ]

    cols = st.columns(len(stages))
    for col, (label, done) in zip(cols, stages):
        with col:
            status = "Complete" if done else "Waiting"
            st.markdown(
                f"""
                <div class="stage-card">
                  <div style="font-size:12px;color:#667085;text-transform:uppercase;">{label}</div>
                  <div style="font-size:18px;font-weight:700;margin-top:6px;">{status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_review(triggers: Dict[str, Any]) -> bool:
    st.subheader("Human Review Checkpoint")
    st.caption("Only detected risk controls appear here.")

    with st.form("human_review_form"):
        special = triggers.get("special_case_items")
        if special:
            st.markdown('<div class="risk-card">', unsafe_allow_html=True)
            st.write("**Special clinical-trial item detected**")
            st.write(f"Item IDs: `{special.get('items', [])}`")
            st.radio(
                "Decision",
                ["Approve dispatch", "Hold/quarantine and retry"],
                key="review_special_case",
                horizontal=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        for idx, item in enumerate(triggers.get("weather_escalation_corridors", [])):
            st.markdown('<div class="risk-card">', unsafe_allow_html=True)
            st.write("**Severe weather escalation detected**")
            st.write(
                f"{item['day']} / `{item['corridor_id']}` / "
                f"risk score `{item['weather_risk_score']}`"
            )
            st.radio(
                "Decision",
                ["Approve with escalation", "Hold corridor and retry"],
                key=f"review_weather_{idx}",
                horizontal=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        for idx, item in enumerate(triggers.get("zero_temp_buffer_days", [])):
            st.markdown('<div class="risk-card">', unsafe_allow_html=True)
            st.write("**Zero spare temp-controlled truck buffer**")
            st.write(
                f"{item['day']} has `{item['allocated_temp_trucks']}` "
                "temp-controlled truck(s) allocated and no spare buffer."
            )
            st.radio(
                "Decision",
                ["Approve zero buffer", "Reserve 1 temp truck and retry"],
                key=f"review_temp_{idx}",
                horizontal=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        return st.form_submit_button("Submit Review Decision", type="primary")


def _render_kpis(state: Dict[str, Any]) -> None:
    allocation = state.get("resource_allocation", {})
    summary = allocation.get("summary_48h", {})
    weather = state.get("corridor_weather_risk", {})
    violations = state.get("validation_violations", [])

    cols = st.columns(4)
    cols[0].metric("Penalty Score", summary.get("total_penalty_score", "N/A"))
    cols[1].metric(
        "Allocation Feasible",
        "Yes" if summary.get("allocation_feasible") else "No",
    )
    cols[2].metric(
        "Escalation Corridors",
        sum(1 for risk in weather.values() if risk.get("escalation_required")),
    )
    cols[3].metric("Validation Issues", len(violations))


def main() -> None:
    st.title("SeeWeeS Agentic Dispatch Control Room")
    st.caption("Streamlit interface for the existing multi-agent dispatch pipeline.")

    with st.sidebar:
        st.header("Inputs")
        pdf_path = st.text_input("Playbook / policy file", DEFAULT_PDF)
        csv_path = st.text_input("Shipment CSV", DEFAULT_CSV)
        resource_path = st.text_input("Resource availability CSV", DEFAULT_RESOURCES)
        send_email = st.checkbox("Send email after report", value=False)

        st.divider()
        run_clicked = st.button("Run Dispatch Pipeline", type="primary", use_container_width=True)
        reset_clicked = st.button("Reset Session", use_container_width=True)

    if reset_clicked:
        st.session_state.clear()
        st.rerun()

    if "pipeline_state" not in st.session_state:
        st.session_state.pipeline_state = {}
    if "awaiting_review" not in st.session_state:
        st.session_state.awaiting_review = False

    state: Dict[str, Any] = st.session_state.pipeline_state
    _render_stage_status(state)

    if run_clicked:
        if os.getenv("ANTHROPIC_API_KEY") == _MISSING_ANTHROPIC_API_KEY:
            st.error("ANTHROPIC_API_KEY is not set. Add it to `.env` before running the LLM stages.")
            st.stop()

        state = _new_state(pdf_path, csv_path, resource_path)

        with st.spinner("Running context, CSV, weather, and resource allocation stages..."):
            try:
                state = _run_upstream(state)
            except Exception as exc:
                st.exception(exc)
                st.stop()

        triggers = _current_triggers(state)
        state["pending_human_review_triggers"] = triggers
        st.session_state.pipeline_state = state

        if triggers.get("review_required"):
            st.session_state.awaiting_review = True
            st.rerun()

        with st.spinner("No human review needed. Continuing to planner and report..."):
            state = _review_not_required(state, triggers)
            state = _run_planner_to_report(state, send_email=send_email)
            st.session_state.pipeline_state = state
            st.session_state.awaiting_review = False
            st.rerun()

    state = st.session_state.pipeline_state

    if st.session_state.awaiting_review and state:
        triggers = state.get("pending_human_review_triggers", {})

        if _render_review(triggers):
            review = _build_review_from_form(
                triggers=triggers,
                retry_count=int(state.get("retry_count", 0)),
            )

            with st.spinner("Applying review decision and continuing pipeline..."):
                try:
                    state = _apply_review(state, review)
                    next_triggers = _current_triggers(state)
                    state["pending_human_review_triggers"] = next_triggers

                    if next_triggers.get("review_required") and review.get("decision") == "retry":
                        st.session_state.pipeline_state = state
                        st.session_state.awaiting_review = True
                        st.rerun()

                    state = _run_planner_to_report(state, send_email=send_email)
                except Exception as exc:
                    st.exception(exc)
                    st.stop()

            st.session_state.pipeline_state = state
            st.session_state.awaiting_review = False
            st.rerun()

    if state:
        st.divider()
        _render_kpis(state)

        if state.get("human_review_summary"):
            with st.expander("Human Review Summary", expanded=True):
                st.text(state["human_review_summary"])

        if state.get("validation_violations"):
            st.warning("Validation completed with unresolved issues.")
            st.write(state["validation_violations"])
        elif state.get("dispatch_plan"):
            st.success("Validation passed or all retry checks completed.")

        if state.get("dispatch_plan"):
            with st.expander("Dispatch Plan", expanded=False):
                st.markdown(state["dispatch_plan"])

        if state.get("report_html"):
            st.subheader("Executive HTML Report")
            components.html(state["report_html"], height=900, scrolling=True)
            st.download_button(
                "Download HTML Report",
                data=state["report_html"],
                file_name="seewees_dispatch_report.html",
                mime="text/html",
            )
    else:
        st.info("Choose the default files or update the paths, then run the dispatch pipeline.")


if __name__ == "__main__":
    main()
