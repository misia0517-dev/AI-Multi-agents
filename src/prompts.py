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
     "Human review context:\n{human_review_summary}\n\n"
     "Resource allocation plan:\n{resource_allocation}\n\n"
     "Return:\n"
     "1) Dispatch plan for Day0 and Day1 per corridor\n"
     "2) Resource allocation rationale (which corridor gets priority and why)\n"
     "3) SLA risk flags and mitigation actions\n"
     "4) What to monitor during execution\n"
     "5) Contingency triggers\n")
])


PLANNER_RETRY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are PlannerAgent. Your previous dispatch plan failed constraint validation. "
     "You MUST explicitly resolve every listed violation in your revised plan. "
     "Prioritize SLA compliance (Tier 1 first), cold-chain integrity, then cost."),
    ("user",
     "⚠️ CONSTRAINT VIOLATIONS FROM PREVIOUS PLAN — you must address ALL of these:\n"
     "{violations}\n\n"
     "Business context:\n{business_context}\n\n"
     "Ops insights:\n{ops_insights}\n\n"
     "Weather risk (per corridor):\n{weather_risk}\n\n"
     "Human review context:\n{human_review_summary}\n\n"
     "Resource allocation plan:\n{resource_allocation}\n\n"
     "Provide a REVISED dispatch plan that:\n"
     "1) States how each violation above is resolved\n"
     "2) Shows the updated Day0 and Day1 plan per corridor\n"
     "3) Documents resource re-allocation rationale\n"
     "4) Lists remaining SLA risks and mitigations\n"
     "5) Defines monitoring checkpoints and contingency triggers\n")
])


REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are ReportAgent. Produce a crisp HTML report for leadership. "
     "Treat human review context as audit facts, not suggestions. "
     "Keep special-case item holds/quarantines separate from temp-controlled truck reserve decisions. "
     "Temp-controlled truck reserves are backup-capacity decisions; do not describe them as reserved for held/quarantined items. "
     "Do not say a temp-controlled truck was reserved for a quarantined item unless the human review context explicitly says so. "
     "Do not state that special-case items have missing unique_item_id unless explicitly shown in the provided data. "
     "Report held/quarantined items or corridors separately from undelivered demand. "
     "Describe reserved resources as intentional backup capacity. "
     "Explain shortfalls only from actual allocation fields. "
     "Do not invent numbers, dates, counts, penalty values, availability, or risk scores. "
     "Use only the provided facts. If a value is missing, write 'not provided'. "
     "Do not change section names. Do not add extra sections."),
    ("user",
     "Inputs:\n\nBusiness context:\n{business_context}\n\n"
     "CSV KPIs:\n{kpis}\n\n"
     "Anomaly highlights:\n{anomaly_highlights}\n\n"
     "Weather risk by corridor:\n{weather_risk}\n\n"
     "Human review context:\n{human_review_summary}\n\n"
     "Resource allocation:\n{resource_allocation}\n\n"
     "Dispatch plan:\n{dispatch_plan}\n\n"
     "Generate a valid HTML report using exactly these sections and this order:\n\n"
     "Executive Summary\n"
     "- Use 5-7 bullet points only.\n\n"
     "Corridor Risk Dashboard\n"
     "- Always use an HTML table.\n"
     "- Columns: Corridor, Day, SLA Tier, Total Units, Undelivered Units, "
     "Weather Risk Score, Travel Buffer %, SLA Risk Flag, Escalation Required.\n\n"
     "Resource Allocation\n"
     "- Always use an HTML table.\n"
     "- Columns: Day, Corridor, Allocated Drivers, Allocated Standard Trucks, "
     "Allocated Temp-Controlled Trucks, Shortfall Drivers, Shortfall Standard Trucks, "
     "Shortfall Temp Trucks, Undelivered Units, Penalty Points.\n\n"
     "Resource Allocation Rationale\n"
     "- Use 4-6 bullet points.\n\n"
     "Dispatch Plan\n"
     "- Always use an HTML table.\n"
     "- Columns: Day, Corridor, SLA Tier, Total Units, Temp-Controlled Trucks Allocated, "
     "Standard Trucks Allocated, Drivers Allocated, Undelivered Units, Weather Risk, Travel Buffer %.\n\n"
     "Data Quality Issues\n"
     "- Use bullets only.\n\n"
     "Recommended Actions\n"
     "- Use bullets only.\n\n"
     "Formatting rules:\n"
     "- Use the exact section names listed above.\n"
     "- Do not add, remove, or rename sections.\n"
     "- Use tables only for the three table sections listed above.\n"
     "- Use bullets only for Executive Summary, Resource Allocation Rationale, "
     "Data Quality Issues, and Recommended Actions.\n"
     "- Keep wording concise and executive-ready.")
])

