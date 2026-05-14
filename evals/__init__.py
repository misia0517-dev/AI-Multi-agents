"""
SeeWeeS Agent-Agnostic Evaluation Framework
=============================================
Evaluates end-to-end output quality of any dispatch planning agent
regardless of the underlying LLM provider (GPT-4, Claude, Gemini, etc.).

Evaluation dimensions:
  1. Deterministic validators  — constraint / math checks
  2. LLM-as-judge evaluators   — qualitative rubric scoring
  3. Composite scoring          — weighted aggregation → pass/fail
"""
