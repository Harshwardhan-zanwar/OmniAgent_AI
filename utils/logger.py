"""Structured logging and plan formatting."""
import os
import logging
import time

is_prod=os.getenv("RENDER","")=="true" or os.getenv("ENV","")=="production"

def setup_logging(level:str="INFO") -> None:
    lvl=getattr(logging,level.upper(),logging.INFO)
    if is_prod:
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        logging.getLogger("omni-agent-ai").info("Logging initialised (prod)")
    else:
        try:
            from rich.logging import RichHandler
            logging.basicConfig(
                level=lvl,
                format="%(message)s",
                datefmt="[%H:%M:%S]",
                handlers=[RichHandler(rich_tracebacks=True,show_path=False)],
            )
            logging.getLogger("omni-agent-ai").info("Logging initialised (Rich)")
        except ImportError:
            logging.basicConfig(level=lvl)
            logging.getLogger("omni-agent-ai").info("Logging initialised (Standard)")

status_icons={
    "success":"✅",
    "failed":"❌",
    "skipped":"⏭️",
    "pending":"⏳",
}

tool_icons={
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

def format_trace(trace:list[dict]) -> str:
    if not trace:
        return "No steps recorded."

    lines=[]
    for step in trace:
        num=step.get("step","?")
        tool=step.get("tool","unknown")
        desc=step.get("description","")
        status=step.get("status","success")
        preview=step.get("output_preview","")

        s_icon=status_icons.get(status,"•")
        t_icon=tool_icons.get(tool,"🔧")

        lines.append(f"{s_icon} Step {num} [{t_icon} {tool}] — {desc}")
        if preview:
            lines.append(f"   ↳ {preview}")
        lines.append("")

    return "\n".join(lines).strip()

def format_trace_json(trace:list[dict]) -> list[dict]:
    res=[]
    for step in trace:
        tool=step.get("tool","unknown")
        status=step.get("status","success")
        res.append({
            **step,
            "tool_icon":tool_icons.get(tool,"🔧"),
            "status_icon":status_icons.get(status,"•"),
        })
    return res

class RequestLogger:
    def __init__(self):
        self.logger=logging.getLogger("omni-agent-ai.requests")

    def log_request(self,sid:str,method:str,path:str,query:str,f_count:int) -> float:
        start=time.time()
        self.logger.info(f"[{sid}] → {method} {path} query='{query[:50]}' files={f_count}")
        return start

    def log_response(self,sid:str,start:float,status:str,steps:int,tok:int|None=None) -> None:
        dt=round(time.time()-start,2)
        self.logger.info(f"[{sid}] ← {status} steps={steps} tokens={tok or '?'} time={dt}s")

def log_summary(sid:str,intent:str,tools:list[str],dt:float,tok_count:int) -> None:
    logger=logging.getLogger("omni-agent-ai.summary")
    logger.info(f"[{sid}] intent={intent} tools=[{', '.join(tools)}] tokens={tok_count} time={dt:.2f}s")