# SeeWeeS Ops Multi-Agent Dispatch System
**UCLA MSBA AI Agents Project Challenge 2026**

A multi-agent AI system built with LangGraph and Claude (Anthropic) for dispatch planning and operations analysis across multiple delivery corridors for SeeWeeS Specialty Logistics.

This project implements **Option 5 (Multi-Region & Multi-Day Resource Planning)** on top of the original starter repo.

---

## What's New (vs. Starter Repo)

### Option 5 — Multi-Region & Multi-Day Resource Planning
- **Multi-corridor weather risk**: fetches weather at all 9 waypoints (C1: Newark→Boston, C2: Newark→Philadelphia) and computes per-corridor risk scores
- **Item Master reconciliation**: resolves legacy IDs, name aliases, and typos using Appendix A from the Playbook
- **Resource allocator node**: greedy allocation of drivers, standard trucks, and temp-controlled trucks across corridors and days using a penalty model (Tier 1 SLA = 100pts, cold-chain = +80pts)
- **14-day multi-corridor dataset**: uses `Incoming_shipments_14d_multi_corridor.csv` and `Resource_availability_48h.csv`

### Technical Changes
- Switched LLM from OpenAI GPT-4.1-mini → **Claude Sonnet (claude-sonnet-4-6)**
- Switched embeddings from OpenAI → **HuggingFace all-MiniLM-L6-v2** (free, runs locally)
- Added `resource_allocator` node between `weather` and `planner` in the LangGraph flow

---

## Project Structure

```
.
├── src/
│   ├── main.py           # Entry point
│   ├── graph.py          # LangGraph workflow
│   ├── agents.py         # Claude-powered agents
│   ├── prompts.py        # Prompt templates
│   ├── tracing.py        # LangSmith integration
│   └── tools/
│       ├── pdf_tools.py      # RAG pipeline (Playbook ingestion)
│       ├── csv_tools.py      # Multi-corridor CSV analysis + DQ rules
│       ├── weather_tools.py  # Multi-waypoint weather risk
│       ├── resource_tools.py # Resource allocation + penalty model
│       └── email_tools.py    # SMTP email delivery
├── data/
│   └── SeeWeeS Specialty distribution.pdf
├── data-for-enhancement/
│   ├── SeeWeeS Specialty Dispatch Playbook.md
│   ├── Incoming_shipments_14d_multi_corridor.csv
│   └── Resource_availability_48h.csv
├── .env.example
└── requirements.txt
```

---

## Setup

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd MSBA_AI_Agents_Demo-data-enhancement-seewees
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
```
Edit `.env` and fill in:
- `ANTHROPIC_API_KEY` — get from [console.anthropic.com](https://console.anthropic.com/settings/keys)
- `REPORT_EMAIL_TO`, `SMTP_*` — optional, for email delivery

### 4. Run
```bash
python src/main.py
```

The pipeline will:
1. Load business rules from the Dispatch Playbook via RAG
2. Analyze 14-day multi-corridor shipment data with DQ checks
3. Fetch weather for all corridor waypoints (9 total)
4. Allocate resources using the penalty model
5. Generate a dispatch plan via Claude
6. Produce and email an executive HTML report

> **Note:** On first run, HuggingFace will download the `all-MiniLM-L6-v2` embedding model (~90MB). Delete `chroma_db/` between runs if you want a fresh vector index.

---

## LangGraph Flow

```
pdf_context → csv_analysis → weather → resource_allocator → planner → report → email
```

## Evaluations

Run the local deterministic checks:

```bash
pytest
```

Run the embedded eval harness against one passing built-in scenario:

```bash
python -m evals --scenario happy_path
```

`python -m evals` runs all built-in fixtures, including intentionally failing scenarios used to verify the validators catch bad outputs.

Run the actual dispatch pipeline and score the completed LangGraph state:

```bash
RUN_EVALS=true python src/main.py
```

Eval reports are written to `eval_results/`.

---

## Data Quality Rules

| Rule | Description | Action |
|------|-------------|--------|
| DQ-01 | Missing `unique_item_id` | Excluded from dispatch |
| DQ-02 | `item_id` not in Item Master | Flagged |
| DQ-03 | `item_name` mismatch | Flagged |
| DQ-04 | Duplicate `unique_item_id` | Flagged |

---

## Resource Penalty Model

| Violation | Penalty per Unit |
|-----------|-----------------|
| Tier 1 SLA violation | 100 pts |
| Tier 2 SLA violation | 40 pts |
| Cold-chain violation (add-on) | +80 pts |
| Non-SLA delay | 10 pts |

Allocation objective: minimize total penalty score; tie-break by fewer Tier 1 units impacted.
