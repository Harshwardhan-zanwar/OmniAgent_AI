"""
Token counting and cost estimation.
  - Counts tokens using tiktoken (accurate)
  - Maps to Gemini 2.5 Flash pricing
  - Returns cost breakdown before agent runs
  - Shown in UI
"""

import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger("omni-agent-ai.utils.cost_estimator")

PRICING = {
    "input":  0.075,   #$0.075 per 1M input tokens
    "output": 0.300,   #$0.300 per 1M output tokens
}

#avg token
FILE_TOKEN_ESTIMATES={
    "pdf":3000,
    "image":500,
    "audio":2000,
    "text":800,
    "other":300,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

SYSTEM_OVERHEAD = 850

#avg output tokens
TASK_OUTPUT_TOKENS = {
    "summarize":600,
    "sentiment":200,
    "code_explain":800,
    "qa":400,
    "transcribe":1500,
    "extract":300,
    "compare":900,
    "conversational":300,
    "youtube":700,
}


@dataclass
class CostEstimate:
    input_tokens:int
    output_tokens:int
    total_tokens:int
    input_cost_usd:float
    output_cost_usd:float
    total_cost_usd:float
    model:str
    breakdown:dict
    formatted:str

def estimate_cost(
    query:str,
    file_names:list[str]|None=None,
    intent:str="qa",
) -> CostEstimate:
    """
    estimate token usage and cost before running the agent.
    called by POST /estimate in main.py.
    """
    breakdown={}
    query_tokens=_count_tokens(query)
    breakdown["query"]=query_tokens

    file_tokens=0
    for fname in(file_names or []):
        ext=Path(fname).suffix.lower()
        tokens=_file_token_estimate(ext)
        breakdown[fname]=tokens
        file_tokens+=tokens

    #sys overhead
    breakdown["system_overhead"]=SYSTEM_OVERHEAD

    input_tokens=query_tokens+file_tokens+SYSTEM_OVERHEAD
    
    output_tokens=TASK_OUTPUT_TOKENS.get(intent,400)
    breakdown["estimated_output"]=output_tokens
    
    input_cost=(input_tokens/1_000_000)*PRICING["input"]
    output_cost=(output_tokens/1_000_000)*PRICING["output"]
    total_cost=input_cost+output_cost

    estimate=CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens+output_tokens,
        input_cost_usd=round(input_cost,6),
        output_cost_usd=round(output_cost,6),
        total_cost_usd=round(total_cost,6),
        model="gemini-2.5-flash",
        breakdown=breakdown,
        formatted=_format_estimate(
            input_tokens,output_tokens,
            input_cost,output_cost,total_cost,breakdown
        ),
    )

    logger.info(
        f"Cost estimate: {input_tokens} input+ {output_tokens} output tokens "
        f"= ${total_cost:.6f}"
    )
    return estimate


def _count_tokens(text: str) -> int:
    """
    Count tokens accurately using tiktoken.
    Fallback=char/4 approx if tiktoken unavailable.
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc=tiktoken.get_encoding("cl100k_base")   # GPT-4 encoding ≈ Gemini
        return len(enc.encode(text))
    except Exception:
        return max(1,len(text)//4)


def _file_token_estimate(ext: str) -> int:
    """Return estimated token count for a file by extension."""
    if ext==".pdf":
        return FILE_TOKEN_ESTIMATES["pdf"]
    elif ext in IMAGE_EXTS:
        return FILE_TOKEN_ESTIMATES["image"]
    elif ext in AUDIO_EXTS:
        return FILE_TOKEN_ESTIMATES["audio"]
    elif ext in {".txt"," .md"," .csv"}:
        return FILE_TOKEN_ESTIMATES["text"]
    return FILE_TOKEN_ESTIMATES["other"]


def _format_estimate(
    input_tokens:int,
    output_tokens:int,
    input_cost:float,
    output_cost:float,
    total_cost:float,
    breakdown:dict,
) -> str:
    """Format a human-readable cost estimate for the UI."""
    lines = [
        "💰 **Pre-flight Cost Estimate**",
        "",
        f"| Component       | Tokens | Cost (USD) |",
        f"|----------------|--------|------------|",
        f"| Input tokens   | {input_tokens:,}  | ${input_cost:.6f} |",
        f"| Output tokens  | {output_tokens:,}  | ${output_cost:.6f} |",
        f"| **Total**      | **{input_tokens + output_tokens:,}** | **${total_cost:.6f}** |",
        "",
        "📋 **Breakdown:**",
    ]
    for item,tokens in breakdown.items():
        if item not in ("estimated_output",):
            lines.append(f"  • {item}: ~{tokens:,} tokens")
    lines.append("")
    lines.append(f"🤖 Model: gemini-2.5-flash")
    lines.append("⚠️ *Estimate only—actual cost may vary*")
    return "\n".join(lines)