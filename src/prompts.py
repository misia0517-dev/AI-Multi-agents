from langchain_core.prompts import ChatPromptTemplate


PDF_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are ContextAgent. Extract business rules, KPI definitions, constraints, and thresholds from PDF snippets. "
     "Be precise. Output structured bullets."),
    ("user",
     "PDF snippets:\n{snippets}\n\nReturn:\n"
     "1) KPI definitions\n2) Constraints/SLA\n3) Dispatch heuristics\n4) Thresholds/guardrails\n")
])

OPS_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are OpsDataAgent. Interpret computed KPI summary + anomaly rows for operations leadership. "
     "Call out data quality issues and likely root causes."),
    ("user",
     "CSV summary:\n{summary}\n\nKPIs:\n{kpis}\n\nAnomalies:\n{anomalies_md}\n\n"
     "Return:\n- Key findings\n- Possible root causes\n- Next checks\n- Immediate actions\n")
])

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are PlannerAgent. Combine business context + ops findings + multi-corridor weather risk "
     "+ resource allocation results into a concrete dispatch plan. "
     "Prioritize SLA compliance (Tier 1 first), safety, and cost. "
     "If resource_allocation shows a penalty score > 0, explicitly address how to mitigate shortfalls."),
    ("user",
     "Business context:\n{business_context}\n\n"
     "Ops insights:\n{ops_insights}\n\n"
     "Weather risk (per corridor):\n{weather_risk}\n\n"
     "Resource allocation plan:\n{resource_allocation}\n\n"
     "Return:\n"
     "1) Dispatch plan for Day0 and Day1 per corridor\n"
     "2) Resource allocation rationale (which corridor gets priority and why)\n"
     "3) SLA risk flags and mitigation actions\n"
     "4) What to monitor during execution\n"
     "5) Contingency triggers\n")
])

REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are ReportAgent. Produce a crisp, executive-ready HTML report for leadership. "
     "Use headings, tables, and bullets. Keep it skimmable. "
     "Highlight the top risks, concrete actions, and the 'why' behind each recommendation. "
     "A non-technical C-suite executive must be able to act on it immediately. "
     "Return ONLY the raw HTML — do not wrap it in markdown code fences."),
    ("user",
     "Inputs:\n\n"
     "Business context:\n{business_context}\n\n"
     "CSV KPIs:\n{kpis}\n\n"
     "Anomaly highlights:\n{anomaly_highlights}\n\n"
     "Weather risk (per corridor):\n{weather_risk}\n\n"
     "Resource allocation plan:\n{resource_allocation}\n\n"
     "Dispatch plan:\n{dispatch_plan}\n\n"
     "Generate HTML report. Include sections: Executive Summary, "
     "Corridor Risk Dashboard, Resource Allocation, Dispatch Plan, "
     "Data Quality Issues, Recommended Actions.")
])
