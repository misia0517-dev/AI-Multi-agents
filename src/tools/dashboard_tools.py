from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(value: float, lo: float, hi: float) -> float:
    """Map value onto 0-100% within [lo, hi], clamped."""
    if hi == lo:
        return 0.0
    return max(0.0, min(100.0, (value - lo) / (hi - lo) * 100.0))


def _kpi_card(
    title: str,
    zones: List[Tuple[str, int]],      # [(hex_color, flex_weight), ...]
    scale_labels: List[str],
    badges: List[Tuple[str, str]],     # [(css_class, text), ...]
    marker_pct: Optional[float] = None,
    actual_label: str = "",
) -> str:
    """Render one dark KPI card with a gauge bar."""

    track_parts = "".join(
        f'<div style="flex:{w};background:{c};height:100%;"></div>'
        for c, w in zones
    )

    marker_html = ""
    if marker_pct is not None:
        marker_html = (
            f'<div style="position:absolute;top:-1px;bottom:-1px;width:3px;'
            f'background:#fff;border-radius:2px;'
            f'left:{marker_pct:.1f}%;transform:translateX(-50%);'
            f'box-shadow:0 0 3px rgba(0,0,0,.6);"></div>'
        )

    scale_html = "".join(f"<span>{l}</span>" for l in scale_labels)

    badge_parts = "".join(
        f'<span style="font-size:10px;padding:3px 8px;border-radius:4px;'
        f'font-weight:500;{_badge_style(cls)}">{txt}</span>'
        for cls, txt in badges
    )

    value_tag = ""
    if actual_label:
        value_tag = (
            f' <span style="color:#aaa;font-size:11px;font-weight:400;">'
            f"— {actual_label}</span>"
        )

    return f"""
<div style="background:#1e1e1e;border-radius:10px;padding:14px 16px 12px;color:#fff;">
  <p style="font-size:13px;font-weight:500;color:#e8e8e8;margin:0 0 12px;">
    {title}{value_tag}
  </p>
  <div style="position:relative;height:10px;border-radius:5px;overflow:hidden;
              display:flex;margin-bottom:7px;">
    {track_parts}
    {marker_html}
  </div>
  <div style="display:flex;justify-content:space-between;font-size:10px;
              color:#888;margin-bottom:8px;">
    {scale_html}
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:5px;">
    {badge_parts}
  </div>
</div>"""


def _badge_style(cls: str) -> str:
    styles = {
        "ideal":    "background:#1e3a28;color:#52be80;",
        "watch":    "background:#3d2b0a;color:#f0a500;",
        "critical": "background:#3b0f0f;color:#e74c3c;",
    }
    return styles.get(cls, "background:#333;color:#ccc;")


def _section(label: str, cards_html: str) -> str:
    return f"""
<div style="margin-bottom:1.5rem;">
  <p style="font-size:12px;color:#999;letter-spacing:.06em;
            text-transform:uppercase;margin:0 0 10px;">{label}</p>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));
              gap:10px;">
    {cards_html}
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_kpi_dashboard_html(state: Dict[str, Any]) -> str:
    """
    Build the KPI gauge dashboard HTML from live state data.
    Returns a self-contained HTML <div> block ready to prepend to the report.
    """
    allocation   = state.get("resource_allocation", {})
    summary_48h  = allocation.get("summary_48h", {})
    weather_risk = state.get("corridor_weather_risk", {})
    kpis         = state.get("csv_kpis", {})

    # ── Extract real values ────────────────────────────────────────────────
    penalty          = float(summary_48h.get("total_penalty_score", 0))
    tier1_impacted   = float(summary_48h.get("tier1_units_impacted", 0))
    feasible         = summary_48h.get("allocation_feasible", True)
    dq_excl_rate     = float(kpis.get("exclusion_rate_pct", 0))
    total_plan_units = int(kpis.get("total_planning_units", 0))
    total_excl       = int(kpis.get("total_excluded_dq01", 0))
    total_flagged    = int(kpis.get("total_flagged_dq02", 0) or 0) + \
                       int(kpis.get("total_flagged_dq04", 0) or 0)

    # Weather: max risk score across corridors (0-3 scale)
    max_weather_score = max(
        (v.get("risk_score_48h", 0) for v in weather_risk.values()),
        default=0,
    )
    any_escalation = any(v.get("escalation_required", False) for v in weather_risk.values())

    # Driver/truck utilisation (Day0, across all corridors)
    day0 = allocation.get("Day0", {})
    avail_day0 = day0.get("available", {})
    used_drivers = sum(
        c.get("allocated_drivers", 0)
        for c in day0.get("corridors", {}).values()
        if isinstance(c, dict)
    )
    total_drivers = avail_day0.get("driver", 1)
    driver_util_pct = _pct(used_drivers, 0, max(total_drivers, 1)) if total_drivers else 0

    used_temp = sum(
        c.get("allocated_temp_trucks", 0)
        for c in day0.get("corridors", {}).values()
        if isinstance(c, dict)
    )
    total_temp = avail_day0.get("truck_temp_controlled", 1)
    temp_util_pct = _pct(used_temp, 0, max(total_temp, 1)) if total_temp else 0

    # ── Section 1: Penalty & Feasibility ──────────────────────────────────
    s1 = (
        _kpi_card(
            "Total penalty score",
            [("#1e8449", 20), ("#d68910", 30), ("#c0392b", 50)],
            ["0", "200", "500", "1 000+"],
            [("ideal", "Ideal: 0 pts"), ("watch", "Watch: 200–500"), ("critical", "Critical: &gt; 500")],
            marker_pct=_pct(penalty, 0, 1000),
            actual_label=f"{int(penalty)} pts",
        )
        + _kpi_card(
            "Tier 1 units at risk",
            [("#1e8449", 5), ("#d68910", 10), ("#c0392b", 15)],
            ["0", "5", "15", "30+"],
            [("ideal", "Ideal: 0"), ("watch", "Watch: 1–10"), ("critical", "Critical: &gt; 10")],
            marker_pct=_pct(tier1_impacted, 0, 30),
            actual_label=f"{int(tier1_impacted)} units",
        )
        + _kpi_card(
            "Allocation feasibility",
            [("#1e8449", 1), ("#c0392b", 1)],
            ["Infeasible", "", "Feasible"],
            [
                ("ideal" if feasible else "critical",
                 "CONFIRMED" if feasible else "SHORTFALL DETECTED"),
            ],
            marker_pct=100.0 if feasible else 0.0,
            actual_label="confirmed" if feasible else "shortfall",
        )
        + _kpi_card(
            "DQ exclusion rate",
            [("#1e8449", 2), ("#d68910", 3), ("#c0392b", 5)],
            ["0%", "2%", "5%", "10%+"],
            [("ideal", "Ideal: &lt; 2%"), ("watch", "Watch: 2–5%"), ("critical", "Critical: &gt; 5%")],
            marker_pct=_pct(dq_excl_rate, 0, 10),
            actual_label=f"{dq_excl_rate:.1f}% ({total_excl} excl. / {total_plan_units + total_excl} rows)",
        )
    )

    # ── Section 2: Resource utilisation ───────────────────────────────────
    s2 = (
        _kpi_card(
            "Driver utilisation (Day 0)",
            [("#c0392b", 12), ("#d68910", 10), ("#1e8449", 4), ("#d68910", 14)],
            ["0%", "60%", "90%", "100%"],
            [("ideal", "Ideal: 70–90%"), ("watch", "Watch: &lt; 60% or &gt; 90%"), ("critical", "Critical: &gt; 100%")],
            marker_pct=driver_util_pct,
            actual_label=f"{used_drivers}/{total_drivers} ({driver_util_pct:.0f}%)",
        )
        + _kpi_card(
            "Temp-controlled truck utilisation (Day 0)",
            [("#c0392b", 12), ("#d68910", 10), ("#1e8449", 4), ("#d68910", 14)],
            ["0%", "60%", "90%", "100%"],
            [("ideal", "Ideal: 70–90%"), ("watch", "Watch: &lt; 60% or &gt; 90%"), ("critical", "Critical: &gt; 100%")],
            marker_pct=temp_util_pct,
            actual_label=f"{used_temp}/{total_temp} ({temp_util_pct:.0f}%)",
        )
    )

    # Add per-corridor shortfall cards
    for corridor_id, cdata in day0.get("corridors", {}).items():
        if not isinstance(cdata, dict):
            continue
        sf_temp = cdata.get("shortfall_temp_trucks", 0)
        sf_std  = cdata.get("shortfall_std_trucks", 0)
        sf_drv  = cdata.get("shortfall_drivers", 0)
        total_sf = sf_temp + sf_std + sf_drv
        label_short = corridor_id.replace("C1_I95_NJ_BOS", "C1 NJ→BOS").replace("C2_NJ_PHL", "C2 NJ→PHL")
        s2 += _kpi_card(
            f"Shortfall — {label_short}",
            [("#1e8449", 1), ("#d68910", 2), ("#c0392b", 7)],
            ["0", "1", "3", "10+"],
            [("ideal", "Ideal: 0"), ("watch", "Watch: 1–2"), ("critical", "Critical: &gt; 3")],
            marker_pct=_pct(total_sf, 0, 10),
            actual_label=f"temp={sf_temp} std={sf_std} drv={sf_drv}",
        )

    # ── Section 3: Weather ─────────────────────────────────────────────────
    s3 = _kpi_card(
        "Max corridor weather risk score",
        [("#1e8449", 1), ("#d68910", 1), ("#c0392b", 1)],
        ["0", "1", "2", "3"],
        [("ideal", "Ideal: 0"), ("watch", "Watch: 1–2"), ("critical", "Critical: 3")],
        marker_pct=_pct(max_weather_score, 0, 3),
        actual_label=f"{max_weather_score}/3{' — escalation required' if any_escalation else ''}",
    )
    for corridor_id, risk in weather_risk.items():
        score  = risk.get("risk_score_48h", 0)
        buf    = risk.get("travel_buffer_pct", 0)
        flags  = risk.get("risk_flags_48h", {})
        active = [k.replace("_risk", "").replace("_", " ") for k, v in flags.items() if v]
        label_short = corridor_id.replace("C1_I95_NJ_BOS", "C1 NJ→BOS").replace("C2_NJ_PHL", "C2 NJ→PHL")
        s3 += _kpi_card(
            f"Weather risk — {label_short}",
            [("#1e8449", 1), ("#d68910", 1), ("#c0392b", 1)],
            ["0", "1", "2", "3"],
            [("ideal", "Ideal: 0"), ("watch", "Watch: 1–2"), ("critical", "Critical: 3")],
            marker_pct=_pct(score, 0, 3),
            actual_label=f"score={score}, buffer +{buf}%{', ' + '/'.join(active) if active else ''}",
        )

    # ── Section 4: Data quality ────────────────────────────────────────────
    s4 = (
        _kpi_card(
            "DQ flags (DQ-02 + DQ-04)",
            [("#1e8449", 5), ("#d68910", 5), ("#c0392b", 10)],
            ["0", "5", "10", "20+"],
            [("ideal", "Ideal: 0"), ("watch", "Watch: 1–10"), ("critical", "Critical: &gt; 10")],
            marker_pct=_pct(total_flagged, 0, 20),
            actual_label=f"{total_flagged} flagged rows",
        )
    )

    # ── Assemble ───────────────────────────────────────────────────────────
    html = f"""
<div style="font-family:sans-serif;padding:0 0 1.5rem;">
  <h2 style="font-size:18px;font-weight:600;color:#111;margin:0 0 6px;">
    SeeWeeS KPI Dashboard
  </h2>
  <p style="font-size:12px;color:#666;margin:0 0 20px;">
    White marker = current run value &nbsp;|&nbsp;
    <span style="color:#52be80;">&#9632;</span> Ideal &nbsp;
    <span style="color:#f0a500;">&#9632;</span> Watch &nbsp;
    <span style="color:#e74c3c;">&#9632;</span> Critical
  </p>
  {_section("Penalty &amp; Feasibility", s1)}
  {_section("Resource Utilisation — Day 0", s2)}
  {_section("Weather Risk", s3)}
  {_section("Data Quality", s4)}
</div>
<hr style="border:none;border-top:1px solid #e0e0e0;margin:0 0 2rem;">
"""
    return html
