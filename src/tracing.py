from __future__ import annotations
import os

def init_langsmith_tracing() -> None:
    """
    Enables LangSmith tracing via env vars.
    Kept as a small explicit init so it's obvious in workshops.

    Required .env keys to activate:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<your LangSmith key>
        LANGCHAIN_PROJECT=MSBA_AI_Agents_Demo          (optional)
        LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  (optional)
    """
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    os.environ.setdefault("LANGCHAIN_PROJECT", "MSBA_AI_Agents_Demo")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        api_key = os.getenv("LANGCHAIN_API_KEY", "")
        if not api_key:
            print("[tracing] WARNING: LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is not set. Tracing disabled.")
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
        else:
            print(f"[tracing] LangSmith tracing enabled — project: {os.getenv('LANGCHAIN_PROJECT')}")
    else:
        print("[tracing] LangSmith tracing disabled.")
