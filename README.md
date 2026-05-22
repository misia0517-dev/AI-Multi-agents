# SeeWeeS Ops — Multi-Agent Dispatch System

A multi-agent AI pipeline built with LangGraph and Claude/Anthropic for multi-corridor dispatch planning and operations analysis. The system ingests shipment data, fetches live weather, allocates resources, generates a validated dispatch plan, and emails an executive HTML report — all automatically.

---

## Requirements

- Python 3.10+
- pip
- A Gmail account (or other SMTP provider) for email delivery
- An Anthropic API key
- A [LangSmith API key](https://smith.langchain.com) *(optional — for tracing)*

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd AI-Multi-agents-main
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> On first run, HuggingFace will download the `all-MiniLM-L6-v2` embedding model (~90 MB). This is a one-time download.

### 3. Configure your environment

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` and set the following values:

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional — LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=your_project_name
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# Optional — Weather location (defaults to Newark, NJ)
WEATHER_LAT=40.7282
WEATHER_LON=-74.0776
WEATHER_TZ=America/New_York

# Optional — Email delivery
REPORT_EMAIL_TO=your@email.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@email.com
SMTP_PASSWORD=your_app_password
```

> **Gmail users:** Use an [App Password](https://support.google.com/accounts/answer/185833), not your regular Gmail password.

---

## Running the Agent

```bash
python src/main.py
```

The pipeline will run end-to-end and print progress as it goes. When complete, an HTML report is printed to the console and emailed to `REPORT_EMAIL_TO` (if configured).

## Running the Streamlit App

The Streamlit app is an optional executive-facing interface over the same dispatch logic. It keeps the CLI flow intact while replacing terminal human-review prompts with browser controls.

```bash
streamlit run streamlit_app.py
```

The app preloads the default playbook, shipment CSV, and resource availability file. It shows only the human-review risks that are actually triggered:

- special clinical-trial item review
- severe weather escalation review
- zero spare temp-controlled truck buffer review

If no review trigger is detected, the app continues directly to planning, validation, and the final executive HTML report.

---

## What the Pipeline Does

The system runs through these steps automatically:

```
[START] ──┬── pdf_context ──┐
           └── csv_analysis ─┴── router ──► weather (parallel, per corridor)
                                                └── collect_weather
                                                        └── resource_allocator
                                                                └── human_review
                                                                        └── planner ◄─┐
                                                                                └── validate
                                                                                    ├── (retry if violations) ──┘
                                                                                    └── report ──► email ──► [END]
```

1. **PDF context** — indexes the Dispatch Playbook via RAG to extract SLAs, KPIs, and dispatch rules
2. **CSV analysis** — analyzes 14 days of multi-corridor shipment data, runs data quality checks, detects anomalies
3. **Weather (parallel)** — fetches live forecasts for all 9 corridor waypoints concurrently and scores risk per corridor
4. **Resource allocator** — greedily assigns drivers, standard trucks, and temp-controlled trucks across corridors using a penalty model
5. **Human review** — flags special cases and resource shortfalls for human approval before planning
6. **Planner** — generates a multi-corridor dispatch plan via Claude using all upstream context
7. **Validate** — checks the plan against resource constraints and weather escalation rules; loops back to the planner if violations are found (up to 2 retries)
8. **Report** — builds a deterministic executive HTML report with a live KPI banner and SLA violation table
9. **Email** — sends the final report via SMTP

---

## Project Structure

```
.
├── src/
│   ├── main.py                   # Entry point
│   ├── graph.py                  # LangGraph workflow and all nodes
│   ├── agents.py                 # Claude-powered agents (context, ops, planner, report)
│   ├── prompts.py                # Prompt templates
│   ├── tracing.py                # LangSmith integration
│   └── tools/
│       ├── pdf_tools.py          # RAG pipeline for Playbook ingestion
│       ├── csv_tools.py          # Multi-corridor CSV analysis and DQ rules
│       ├── weather_tools.py      # Multi-waypoint weather risk scoring
│       ├── resource_tools.py     # Resource allocation and penalty model
│       ├── human_review_tools.py # Human-in-the-loop review and retry logic
│       └── email_tools.py        # SMTP email delivery
├── data/
│   └── SeeWeeS Specialty distribution.pdf
├── data-for-enhancement/
│   ├── SeeWeeS Specialty Dispatch Playbook.md
│   ├── Incoming_shipments_14d_multi_corridor.csv
│   └── Resource_availability_48h.csv
├── evals/                        # Evaluation harness
├── .env.example
├── requirements.txt
└── README.md
```

---

## Data Quality Rules

The CSV analysis node enforces four rules before any planning occurs:

| Rule  | Description                          | Action                 |
|-------|--------------------------------------|------------------------|
| DQ-01 | Missing `unique_item_id`             | Excluded from dispatch |
| DQ-02 | `item_id` not in Item Master         | Flagged                |
| DQ-03 | `item_name` mismatch vs. Item Master | Flagged                |
| DQ-04 | Duplicate `unique_item_id`           | Flagged                |

---

## Resource Penalty Model

The resource allocator minimizes total penalty score. Tie-breaks favor fewer Tier 1 units impacted.

| Violation                     | Penalty per Unit |
|-------------------------------|-----------------|
| Tier 1 SLA violation          | 100 pts         |
| Tier 2 SLA violation          | 40 pts          |
| Cold-chain violation (add-on) | +80 pts         |
| Non-SLA delay                 | 10 pts          |

---

## Running Evaluations

Run deterministic unit tests:

```bash
pytest
```

Run the built-in eval harness against a single scenario:

```bash
python -m evals --scenario happy_path
```

Run all eval fixtures (includes intentionally failing scenarios):

```bash
python -m evals
```

Run the full pipeline and score the final state:

```bash
RUN_EVALS=true python src/main.py
```

Eval reports are saved to `eval_results/`.

---

## Troubleshooting

**Dependency conflicts on install**
Make sure you're using Python 3.10+ and that `pip` is up to date (`pip install --upgrade pip`). The `requirements.txt` uses `>=` version ranges to let pip resolve compatible versions automatically.

**`chroma_db/` errors or stale vector index**
Delete the `chroma_db/` directory between runs to force a fresh index:

```bash
rm -rf chroma_db/
```

**Email not sending**
Double-check `SMTP_PASSWORD` — Gmail requires an App Password, not your account password. Confirm `REPORT_EMAIL_TO` is set in `.env`.

**LangSmith tracing not appearing**
Set `LANGCHAIN_TRACING_V2=true` and verify `LANGCHAIN_API_KEY` is valid. Tracing is optional and the pipeline runs fine without it.
