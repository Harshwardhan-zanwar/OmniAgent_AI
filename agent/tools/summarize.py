"""Text and YouTube summarization tools."""
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agent.config import get_client

logger=logging.getLogger("omni-agent-ai.tools.summarize")

gemini_model="gemini-2.5-flash-lite"
chunk_size=6000
overlap=200
yt_threshold=4000

summary_prompt="""Summarize this content in the following style:
- ONE-LINE SUMMARY: A single sentence overview.
- 3 KEY POINTS: Bulleted list of 3 main highlights.
- 5-SENTENCE SUMMARY: A cohesive 5-sentence paragraph.

Please do not include extra headers or sections.

Content:
{text}"""

yt_summary_prompt="""Summarize the YouTube transcript below:
- ONE-LINE SUMMARY: A single sentence capturing the core topic.
- KEY TOPICS COVERED: Bullet points of the main topics.
- 3 KEY TAKEAWAYS: Bullet points of the key takeaways.
- 5-SENTENCE SUMMARY: A cohesive 5-sentence paragraph.

Transcript:
{text}"""

def _client():
    return get_client()

def _get_yt_transcript(txt:str) -> tuple[str,str]:
    import re
    match=re.search(r"\[YouTube Transcript: (https?://[^\]]+)\]\n(.*)",txt,re.DOTALL)
    if match:
        return match.group(2).strip(),match.group(1).strip()
    return txt,""

async def summarize(txt:str) -> str:
    if not txt or len(txt.strip())<50:
        return "[Not enough content to summarize.]"

    logger.info(f"Summarizing text: {len(txt)} chars")
    if len(txt)>chunk_size:
        txt=await _summarize_chunks(txt)

    resp=_client().models.generate_content(
        model=gemini_model,
        contents=summary_prompt.format(text=txt[:6000]),
    )
    return resp.text.strip()

async def summarize_yt(content:str) -> str:
    sub,url=_get_yt_transcript(content)
    if not sub or len(sub.strip())<50:
        return "[No transcript content found to summarize.]"

    logger.info(f"YouTube summarization: {len(sub)} chars, url={url}")
    if len(sub)>yt_threshold:
        sum_content=await _reduce(sub)
    else:
        sum_content=sub

    resp=_client().models.generate_content(
        model=gemini_model,
        contents=yt_summary_prompt.format(text=sum_content[:6000]),
    )
    summary=resp.text.strip()
    preview=_preview_yt(sub)

    return f"{summary}\n\n---\n\n{preview}"

async def _reduce(sub:str) -> str:
    split=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=overlap)
    chunks=split.split_text(sub)
    logger.info(f"Map-reduce: {len(chunks)} chunks")

    client=_client()
    briefs=[]
    for i,chunk in enumerate(chunks,1):
        logger.info(f"Reducing chunk {i}/{len(chunks)}")
        resp=client.models.generate_content(
            model=gemini_model,
            contents=f"Summarize this section of a video transcript in 3-4 sentences:\n\n{chunk}",
        )
        briefs.append(resp.text.strip())

    return "\n\n".join(briefs)

def _preview_yt(sub:str) -> str:
    words=len(sub.split())
    preview=sub[:800].strip()
    truncated=len(sub)>800

    out=[
        "📝 **TRANSCRIPT**",
        f"*({words:,} words total)*",
        "",
        preview,
    ]
    if truncated:
        out.append(f"\n*... [{len(sub)-800:,} more characters — expand Extracted Text panel to view]*")

    return "\n".join(out)

async def _summarize_chunks(txt:str) -> str:
    split=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=overlap)
    chunks=split.split_text(txt)
    logger.info(f"Doc split: {len(chunks)} chunks")

    client=_client()
    briefs=[]
    for i,chunk in enumerate(chunks,1):
        logger.info(f"Summarizing chunk {i}/{len(chunks)}")
        resp=client.models.generate_content(
            model=gemini_model,
            contents=f"Summarize this section in 3-4 sentences:\n\n{chunk}",
        )
        briefs.append(resp.text.strip())

    return "\n\n".join(briefs)