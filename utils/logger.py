"""
Structured logging + plan trace formatting.
  - Rich colored console output during development
  - JSON-structured logs for production (Render)
  - Plan trace formatter for the UI
  - Request/response logging middleware helper
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Any

#detect environment
IS_PRODUCTION=os.getenv("RENDER","") == "true" or os.getenv("ENV","")=="production"

def setup_logging(level:str="INFO") -> None:
    """
    Configure logging for the whole app.
    - Development : Rich colored output
    - Production  : Plain JSON lines (easier to read in Render logs)
    """
    log_level=getattr(logging,level.upper(),logging.INFO)

    if IS_PRODUCTION:
        #simple format for Render log viewer
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        logger = logging.getLogger("omni-agent-ai")
        logger.info("Logging initialised (production mode)")
    else:
        #rich colored output
        try:
            from rich.logging import RichHandler
            logging.basicConfig(
                level=log_level,
                format="%(message)s",
                datefmt="[%H:%M:%S]",
                handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
            )
            logger = logging.getLogger("omni-agent-ai")
            logger.info("Logging initialised (dev mode with Rich)")
        except ImportError:
            logging.basicConfig(level=log_level)
            logger = logging.getLogger("omni-agent-ai")
            logger.info("Logging initialised (dev mode, Rich not installed)")

#PLAN TRACE FORMATTER
STATUS_ICONS = {
    "success": "✅",
    "failed":  "❌",
    "skipped": "⏭️",
    "pending": "⏳",
}

TOOL_ICONS = {
    "pdf_extractor":"📄",
    "ocr_vision":"🔍",
    "audio_transcriber":"🎙️",
    "youtube_fetcher":"▶️",
    "intent_classifier":"🧠",
    "planner":"📋",
    "summarize":"📝",
    "sentiment":"💬",
    "code_explain":"💻",
    "qa_answer":"❓",
    "compare":"⚖️",
    "conversational":"💭",
    "passthrough":"➡️",
    "text_reader":"📃",
}

def format_plan_trace(plan_trace: list[dict]) -> str:
    """
    Convert a plan_trace list into a human-readable string for the UI.
    Example output:
        ✅ Step 1 [📄 pdf_extractor] — Extracted text from report.pdf
           Preview: The Q3 results showed a 12% increase in…

        ✅ Step 2 [🧠 intent_classifier] — Detected intent: 'summarize'(94%)
           Preview: User wants a structured summary of the document.
    """
    if not plan_trace:
        return "No steps recorded."

    lines = []
    for step in plan_trace:
        step_num=step.get("step", "?")
        tool=step.get("tool", "unknown")
        description=step.get("description", "")
        status=step.get("status", "success")
        preview=step.get("output_preview", "")

        status_icon=STATUS_ICONS.get(status, "•")
        tool_icon=TOOL_ICONS.get(tool, "🔧")

        lines.append(
            f"{status_icon} Step {step_num} [{tool_icon} {tool}] — {description}"
        )
        if preview:
            lines.append(f"   ↳ {preview}")
        lines.append("")

    return "\n".join(lines).strip()

def format_plan_trace_json(plan_trace: list[dict]) -> list[dict]:
    """
    Enrich plan trace dicts with icons — returned as JSON for the frontend
    to render as an animated step list.
    """
    enriched=[]
    for step in plan_trace:
        tool=step.get("tool", "unknown")
        status=step.get("status", "success")
        enriched.append({
            **step,
            "tool_icon":TOOL_ICONS.get(tool,"🔧"),
            "status_icon":STATUS_ICONS.get(status,"•"),
        })
    return enriched

# REQUEST LOGGER —in main.py middleware
class RequestLogger:
    """
    Lightweight request/response logger.
    Use as a dependency in FastAPI endpoints.
    """
    def __init__(self):
        self.logger=logging.getLogger("omni-agent-ai.requests")

    def log_request(
        self,
        session_id:str,
        method:str,
        path:str,
        query:str,
        file_count:int,
    ) -> float:
        """Log incoming request. Returns start timestamp."""
        start=time.time()
        self.logger.info(
            f"[{session_id}]→{method}{path}"
            f"query='{query[:50]}'files={file_count}"
        )
        return start

    def log_response(
        self,
        session_id:str,
        start_time:float,
        status:str,
        steps:int,
        tokens:int | None = None,
    ) -> None:
        """Log completed response with timing."""
        elapsed=round(time.time()-start_time, 2)
        self.logger.info(
            f"[{session_id}] ← {status} "
            f"steps={steps} tokens={tokens or '?'} "
            f"time={elapsed}s"
        )

#utils
def log_agent_summary(
    session_id:str,
    intent:str,
    tools_used:list[str],
    elapsed:float,
    token_count:int,
) -> None:
    """Log a one-line summary after each agent run — useful for monitoring."""
    logger=logging.getLogger("omni-agent-ai.summary")
    logger.info(
        f"[{session_id}]intent={intent}"
        f"tools=[{', '.join(tools_used)}]"
        f"tokens={token_count}"
        f"time={elapsed:.2f}s"
    )