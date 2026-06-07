"""Token counting and cost estimation."""
import logging
from pathlib import Path
from dataclasses import dataclass

logger=logging.getLogger("omni-agent-ai.utils.cost_estimator")

pricing={
    "input":0.075,
    "output":0.300,
}

file_estimates={
    "pdf":3000,
    "image":500,
    "audio":2000,
    "text":800,
    "other":300,
}

img_exts={".jpg",".jpeg",".png",".webp",".gif",".bmp"}
audio_exts={".mp3",".wav",".m4a",".ogg",".flac"}
sys_overhead=850

task_outputs={
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

def get_cost(query:str,names:list[str]|None=None,intent:str="qa") -> CostEstimate:
    breakdown={}
    q_tok=_tokens(query)
    breakdown["query"]=q_tok

    f_tok=0
    for name in (names or []):
        ext=Path(name).suffix.lower()
        tok=_file_tokens(ext)
        breakdown[name]=tok
        f_tok+=tok

    breakdown["system_overhead"]=sys_overhead

    in_tok=q_tok+f_tok+sys_overhead
    out_tok=task_outputs.get(intent,400)
    breakdown["estimated_output"]=out_tok

    in_cost=(in_tok/1000000)*pricing["input"]
    out_cost=(out_tok/1000000)*pricing["output"]
    tot_cost=in_cost+out_cost

    return CostEstimate(
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=in_tok+out_tok,
        input_cost_usd=round(in_cost,6),
        output_cost_usd=round(out_cost,6),
        total_cost_usd=round(tot_cost,6),
        model="gemini-2.5-flash",
        breakdown=breakdown,
        formatted=_format(in_tok,out_tok,in_cost,out_cost,tot_cost,breakdown),
    )

def _tokens(txt:str) -> int:
    if not txt:
        return 0
    try:
        import tiktoken
        enc=tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(txt))
    except Exception:
        return max(1,len(txt)//4)

def _file_tokens(ext:str) -> int:
    if ext==".pdf":
        return file_estimates["pdf"]
    elif ext in img_exts:
        return file_estimates["image"]
    elif ext in audio_exts:
        return file_estimates["audio"]
    elif ext in {".txt",".md",".csv"}:
        return file_estimates["text"]
    return file_estimates["other"]

def _format(in_tok:int,out_tok:int,in_cost:float,out_cost:float,tot_cost:float,breakdown:dict) -> str:
    lines=[
        "💰 **Pre-flight Cost Estimate**",
        "",
        "| Component | Tokens | Cost (USD) |",
        "|---|---|---|",
        f"| Input tokens | {in_tok:,} | ${in_cost:.6f} |",
        f"| Output tokens | {out_tok:,} | ${out_cost:.6f} |",
        f"| **Total** | **{in_tok+out_tok:,}** | **${tot_cost:.6f}** |",
        "",
        "📋 **Breakdown:**",
    ]
    for item,tok in breakdown.items():
        if item!="estimated_output":
            lines.append(f" • {item}: ~{tok:,} tokens")
    lines.append("")
    lines.append("🤖 Model: gemini-2.5-flash")
    lines.append("⚠️ *Estimate only—actual cost may vary*")
    return "\n".join(lines)