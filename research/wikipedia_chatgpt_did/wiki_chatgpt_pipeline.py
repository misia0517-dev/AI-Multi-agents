"""Starter pipeline for reproducing the paper's article classification workflow.

This script is intentionally modular. It can collect Wikipedia data, generate
GPT-style article text, compute embedding similarity, and produce a classified
CSV that is ready for difference-in-differences analysis.

Required environment variable for GPT steps:
    OPENAI_API_KEY

Example:
    python research/wikipedia_chatgpt_did/wiki_chatgpt_pipeline.py \
        --start-month 2021-12 \
        --end-month 2023-11 \
        --top-month-start 2020-01 \
        --output research/wikipedia_chatgpt_did/articles_classified.csv \
        --panel-output research/wikipedia_chatgpt_did/did_panel.csv
"""

from __future__ import annotations

import argparse
import calendar
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests


WIKIMEDIA_HEADERS = {
    "User-Agent": "Wikipedia-ChatGPT-DiD-research/0.1 (local academic replication)"
}

ARTICLE_PROMPT = """You are an assistant whose task is to write an encyclopedic article for
a given topic chosen by the user, similar to those found on Wikipedia.
Generate an encyclopedic article in English with title "{title}"."""


@dataclass(frozen=True)
class Month:
    year: int
    month: int

    @property
    def key(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def month_range(start: str, end: str) -> list[Month]:
    start_dt = datetime.strptime(start, "%Y-%m")
    end_dt = datetime.strptime(end, "%Y-%m")
    months: list[Month] = []
    year, month = start_dt.year, start_dt.month
    while (year, month) <= (end_dt.year, end_dt.month):
        months.append(Month(year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def wikipedia_top_pages(month: Month, limit: int = 1000) -> pd.DataFrame:
    last_day = calendar.monthrange(month.year, month.month)[1]
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
        f"en.wikipedia/all-access/{month.year}/{month.month:02d}/{last_day:02d}"
    )
    response = requests.get(url, headers=WIKIMEDIA_HEADERS, timeout=30)
    response.raise_for_status()
    rows = response.json()["items"][0]["articles"][:limit]
    df = pd.DataFrame(rows)
    df["top_month"] = month.key
    return df.rename(columns={"article": "page_title", "views": "top_month_views"})


def clean_titles(titles: Iterable[str]) -> list[str]:
    cleaned = []
    blocked_prefixes = (
        "Special:",
        "Wikipedia:",
        "File:",
        "Template:",
        "Category:",
        "Help:",
        "Portal:",
        "Talk:",
    )
    blocked_titles = {"Main_Page", "-", "404.php"}
    for title in titles:
        if title in blocked_titles:
            continue
        if any(title.startswith(prefix) for prefix in blocked_prefixes):
            continue
        cleaned.append(title.replace("_", " "))
    return sorted(set(cleaned))


def fetch_page_extract(title: str) -> dict:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info|revisions",
        "explaintext": 1,
        "exsectionformat": "plain",
        "titles": title,
        "rvprop": "timestamp",
        "rvdir": "newer",
        "rvlimit": 1,
        "inprop": "url",
        "redirects": 1,
    }
    response = requests.get(url, params=params, headers=WIKIMEDIA_HEADERS, timeout=30)
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        return {}
    revisions = page.get("revisions") or []
    created_at = revisions[0]["timestamp"] if revisions else None
    text = page.get("extract", "") or ""
    return {
        "page_title": page.get("title", title),
        "created_at": created_at,
        "article_length": page.get("length", len(text)),
        "wikipedia_text": text,
        "fullurl": page.get("fullurl"),
    }


def fetch_monthly_views(title: str, start: str, end: str) -> pd.DataFrame:
    start_ts = start.replace("-", "") + "0100"
    end_year, end_month = map(int, end.split("-"))
    end_day = calendar.monthrange(end_year, end_month)[1]
    end_ts = f"{end_year:04d}{end_month:02d}{end_day:02d}00"
    encoded_title = quote(title.replace(" ", "_"), safe="")
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/user/{encoded_title}/monthly/{start_ts}/{end_ts}"
    )
    response = requests.get(url, headers=WIKIMEDIA_HEADERS, timeout=30)
    if response.status_code == 404:
        return pd.DataFrame(columns=["page_title", "month", "views"])
    response.raise_for_status()
    rows = response.json().get("items", [])
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["page_title", "month", "views"])
    out["page_title"] = title
    out["month"] = out["timestamp"].str.slice(0, 4) + "-" + out["timestamp"].str.slice(4, 6)
    return out[["page_title", "month", "views"]]


def iso_month_bounds(month: Month) -> tuple[str, str]:
    start = datetime(month.year, month.month, 1, tzinfo=timezone.utc)
    last_day = calendar.monthrange(month.year, month.month)[1]
    end = datetime(month.year, month.month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_monthly_edits(title: str, start: str, end: str, sleep_seconds: float = 0.1) -> pd.DataFrame:
    rows = []
    for month in month_range(start, end):
        start_iso, end_iso = iso_month_bounds(month)
        count = 0
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "titles": title,
            "rvprop": "timestamp",
            "rvlimit": "max",
            "rvdir": "newer",
            "rvstart": start_iso,
            "rvend": end_iso,
        }
        while True:
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                headers=WIKIMEDIA_HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            pages = payload.get("query", {}).get("pages", {})
            for page in pages.values():
                count += len(page.get("revisions", []))
            if "continue" not in payload:
                break
            params.update(payload["continue"])
            time.sleep(sleep_seconds)
        rows.append({"page_title": title, "month": month.key, "edits": count})
        time.sleep(sleep_seconds)
    return pd.DataFrame(rows)


def month_age(created_at: str, month_key: str) -> int | None:
    if not created_at:
        return None
    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    activity = datetime.strptime(month_key, "%Y-%m").replace(tzinfo=timezone.utc)
    return (activity.year - created.year) * 12 + (activity.month - created.month)


def build_did_panel(
    articles: pd.DataFrame,
    start: str,
    end: str,
    sleep_seconds: float,
) -> pd.DataFrame:
    panels = []
    for _, row in articles.iterrows():
        title = row["page_title"]
        print(f"Collecting monthly activity: {title}")
        views = fetch_monthly_views(title, start, end)
        edits = fetch_monthly_edits(title, start, end, sleep_seconds=sleep_seconds)
        panel = pd.merge(views, edits, on=["page_title", "month"], how="outer").fillna(
            {"views": 0, "edits": 0}
        )
        panel["created_at"] = row.get("created_at")
        panel["article_length"] = row.get("article_length")
        panel["similarity_score"] = row.get("similarity_score")
        panel["similar_label"] = row.get("similar_label")
        panel["post_chatgpt"] = (panel["month"] >= "2022-12").astype(int)
        panel["article_age_months"] = panel["month"].apply(
            lambda month_key: month_age(row.get("created_at"), month_key)
        )
        panels.append(panel)
    if not panels:
        return pd.DataFrame()
    return pd.concat(panels, ignore_index=True)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    a = np.array(left, dtype=float)
    b = np.array(right, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return math.nan
    return float(np.dot(a, b) / denom)


def openai_client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def generate_gpt_article(client, title: str, model: str = "gpt-3.5-turbo") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": ARTICLE_PROMPT.format(title=title)}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def embed_texts(client, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def classify_by_median(df: pd.DataFrame, score_col: str = "similarity_score") -> pd.DataFrame:
    result = df.copy()
    median_score = result[score_col].median()
    result["similar_label"] = (result[score_col] > median_score).astype(int)
    result["similarity_median"] = median_score
    return result


def collect_article_pool(top_month_start: str, end_month: str, sleep_seconds: float) -> list[str]:
    frames = []
    for month in month_range(top_month_start, end_month):
        print(f"Collecting top pages for {month.key}")
        frames.append(wikipedia_top_pages(month))
        time.sleep(sleep_seconds)
    top_pages = pd.concat(frames, ignore_index=True)
    return clean_titles(top_pages["page_title"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-month", default="2021-12")
    parser.add_argument("--end-month", default="2023-11")
    parser.add_argument("--top-month-start", default="2020-01")
    parser.add_argument("--max-articles", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--output", default="research/wikipedia_chatgpt_did/articles_classified.csv")
    parser.add_argument("--panel-output", default="")
    parser.add_argument("--skip-gpt", action="store_true")
    args = parser.parse_args()

    titles = collect_article_pool(args.top_month_start, args.end_month, args.sleep_seconds)
    if args.max_articles:
        titles = titles[: args.max_articles]

    metadata_rows = []
    for title in titles:
        print(f"Fetching article text: {title}")
        row = fetch_page_extract(title)
        if row and row.get("wikipedia_text"):
            metadata_rows.append(row)
        time.sleep(args.sleep_seconds)

    articles = pd.DataFrame(metadata_rows)
    if args.skip_gpt:
        articles.to_csv(args.output, index=False)
        print(f"Wrote metadata-only file to {args.output}")
        return

    client = openai_client()
    gpt_texts = []
    scores = []
    for _, row in articles.iterrows():
        title = row["page_title"]
        print(f"Generating and embedding: {title}")
        generated = generate_gpt_article(client, title)
        wiki_embedding, gpt_embedding = embed_texts(client, [row["wikipedia_text"], generated])
        gpt_texts.append(generated)
        scores.append(cosine_similarity(wiki_embedding, gpt_embedding))
        time.sleep(args.sleep_seconds)

    articles["gpt_text"] = gpt_texts
    articles["similarity_score"] = scores
    articles = classify_by_median(articles)
    articles.to_csv(args.output, index=False)
    print(f"Wrote classified articles to {args.output}")

    if args.panel_output:
        panel = build_did_panel(articles, args.start_month, args.end_month, args.sleep_seconds)
        panel.to_csv(args.panel_output, index=False)
        print(f"Wrote DiD panel to {args.panel_output}")


if __name__ == "__main__":
    main()
