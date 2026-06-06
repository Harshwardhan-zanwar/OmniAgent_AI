'''
Cross-input reasoning — compare and synthesize
content from multiple uploaded files.

Handles:
  - "Do these two documents discuss the same topic?"
  - "Audio + PDF comparison"
  - "What's different between these files?"
  - Any unified query across multiple inputs
'''

import logging
import re
import os

from google import genai as google_genai
from agent.config import GEMINI_MODEL
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("omni-agent-ai.tools.cross_input")

COMPARE_PROMPT = PromptTemplate(
    input_variables=["query", "sources"],
    template='''You are an expert analyst. The user has provided multiple sources of content.
User's question: {query}
Sources:{sources}
Provide a thorough comparative analysis that:
1. **Common Themes** — What topics/ideas appear across multiple sources?
2. **Key Differences** — What is unique to each source?
3. **Direct Answer** — Directly answer the user's question using evidence from the sources
4. **Conclusion** — One-paragraph synthesis
Be specific. Reference the source names (e.g., "In [File: audio.mp3]...") when citing evidence.'''
)
CHUNK_SIZE=3000 

async def compare_inputs(query: str, combined_text: str) -> str:
    '''
    Perform cross-input reasoning across multiple files.
    Splits combined_text back into per-source sections and compares them.
    '''
    if not combined_text or len(combined_text.strip())<50:
        return "[Not enough content across inputs for comparison.]"

    logger.info(f"Cross-input comparison.Query='{query[:60]}',text={len(combined_text)} chars")

    sources=_split_into_sources(combined_text)

    if len(sources)<2:
        logger.info("Only one source found — falling back to QA")
        return await _single_source_qa(query, combined_text)

    formatted_sources=""
    for name,content in sources.items():
        trimmed=content[:CHUNK_SIZE]
        formatted_sources+=f"\n---\n**{name}**\n{trimmed}\n"

    prompt=COMPARE_PROMPT.format(query=query or "Compare these sources.",sources=formatted_sources)
    
    client=google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response=client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    
    result=response.text.strip()
    logger.info(f"Cross-input analysis generated: {len(result)} chars")
    return result


def _split_into_sources(combined_text: str) -> dict[str, str]:
    '''
    Parse the combined_text string (built by planner.py) back into
    a dict of {filename: content}.

    planner.py formats it as:
        [File: filename.ext]
        <content>

        [File: filename2.ext]
        <content>
    '''
    sources={}
    pattern=re.compile(r"\[File: ([^\]]+)\]\n(.*?)(?=\[File: |\Z)", re.DOTALL)
    matches=pattern.findall(combined_text)

    for name,content in matches:
        sources[f"File: {name.strip()}"] = content.strip()

    #capture yt transcripts
    yt_pattern=re.compile(r"\[YouTube Transcript: ([^\]]+)\]\n(.*?)(?=\[|\Z)", re.DOTALL)
    yt_matches=yt_pattern.findall(combined_text)
    for url,content in yt_matches:
        sources[f"YouTube: {url.strip()}"] = content.strip()

    # If no structure, treating as one blob
    if not sources:
        sources["Combined Input"]=combined_text

    return sources


async def _single_source_qa(query:str, text:str) -> str:
    '''Fallback when only one source is present.'''
    client=google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response=client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"Answer this question based on the content below:\n\nQuestion:{query}\nContent:{text[:5000]}"
    )
    return response.text.strip()