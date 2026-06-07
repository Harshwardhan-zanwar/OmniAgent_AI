"""Cross-input reasoning and file comparison."""
import logging
import re
from agent.config import get_client

logger=logging.getLogger("omni-agent-ai.tools.cross_input")

prompt_template="""Compare and synthesize the provided source contents to answer the user query.

User query: {query}

Sources:
{sources}

In your analysis:
- Identify common themes across sources.
- Detail key differences between them.
- Directly answer the user's query with evidence.
- Conclude with a brief synthesis.
Cite source names (e.g. "[File: filename.txt]") when citing evidence."""

chunk_size=3000

async def compare(query:str,combined:str) -> str:
    if not combined or len(combined.strip())<50:
        return "[Not enough content for comparison.]"

    logger.info(f"Cross-input comparison: query='{query[:60]}', size={len(combined)}")
    sources=_parse_sources(combined)

    if len(sources)<2:
        return await _single_qa(query,combined)

    src_text=""
    for name,content in sources.items():
        chunk=content[:chunk_size]
        src_text+=f"\n---\n**{name}**\n{chunk}\n"

    prompt=prompt_template.format(query=query or "Compare these sources.",sources=src_text)
    client=get_client()
    resp=client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    return resp.text.strip()

def _parse_sources(txt:str) -> dict[str,str]:
    sources={}
    pat=re.compile(r"\[File: ([^\]]+)\]\n(.*?)(?=\[File: |\Z)",re.DOTALL)
    for name,content in pat.findall(txt):
        sources[f"File: {name.strip()}"]=content.strip()

    yt_pat=re.compile(r"\[YouTube Transcript: ([^\]]+)\]\n(.*?)(?=\[|\Z)",re.DOTALL)
    for url,content in yt_pat.findall(txt):
        sources[f"YouTube: {url.strip()}"]=content.strip()

    if not sources:
        sources["Combined Input"]=txt
    return sources

async def _single_qa(query:str,txt:str) -> str:
    client=get_client()
    resp=client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"Answer the question based on the content:\n\nQuestion: {query}\nContent: {txt[:5000]}"
    )
    return resp.text.strip()