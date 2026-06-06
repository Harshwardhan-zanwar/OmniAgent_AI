"""
agent/tools/summarize.py
──────────────────────────────────────────────
Text summarization — two modes:
  summarize_text()              : generic docs, PDFs, audio transcripts
  summarize_youtube_transcript(): YouTube-specific with transcript display
                                  + map-reduce for long videos
──────────────────────────────────────────────
"""

import os
import logging
from google import genai as google_genai
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("omni-agent-ai.tools.summarize")

GEMINI_MODEL  = "gemini-2.5-flash-lite"
CHUNK_SIZE    = 6000
CHUNK_OVERLAP = 200

# Threshold — transcripts longer than this use map-reduce
YT_LONG_THRESHOLD = 4000

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are an expert summarizer. Summarize the content below in exactly this format:

ONE-LINE SUMMARY: A single sentence capturing the core topic.

3 KEY POINTS:
• <Point 1>
• <Point 2>
• <Point 3>

5-SENTENCE SUMMARY:
<Sentence 1>. <Sentence 2>. <Sentence 3>. <Sentence 4>. <Sentence 5>.

Content to summarize:
{text}

Important: Follow the format exactly. Do not add extra sections."""
)

YT_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are an expert at summarizing YouTube video transcripts.

Summarize the transcript below in exactly this format:

ONE-LINE SUMMARY: A single sentence capturing what this video is about.

KEY TOPICS COVERED:
• <Topic 1>
• <Topic 2>
• <Topic 3>
• <Topic 4>

3 KEY TAKEAWAYS:
• <Takeaway 1>
• <Takeaway 2>
• <Takeaway 3>

5-SENTENCE SUMMARY:
<Sentence 1>. <Sentence 2>. <Sentence 3>. <Sentence 4>. <Sentence 5>.

Transcript:
{text}

Important: Follow the format exactly. Do not add extra sections."""
)


def _get_client() -> google_genai.Client:
    return google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _extract_transcript_text(content: str) -> tuple[str, str]:
    """
    Separates the YouTube transcript from the rest of the content.
    Returns (transcript_only, yt_url).
    """
    import re
    match = re.search(r"\[YouTube Transcript: (https?://[^\]]+)\]\n(.*)", content, re.DOTALL)
    if match:
        return match.group(2).strip(), match.group(1).strip()
    return content, ""


async def summarize_text(text: str) -> str:
    if not text or len(text.strip()) < 50:
        return "[Not enough content to summarize.]"

    logger.info(f"Summarizing text: {len(text)} chars")

    if len(text) > CHUNK_SIZE:
        text = await _chunk_and_summarize(text)

    client   = _get_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=SUMMARY_PROMPT.format(text=text[:6000]),
    )
    result = response.text.strip()
    logger.info(f"Summary generated: {len(result)} chars")
    return result


async def summarize_youtube_transcript(content: str) -> str:
    """
    YouTube-specific summarization.
      1. Extracts the raw transcript from content
      2. Decides short vs long path based on YT_LONG_THRESHOLD
      3. Short → single Gemini call with YT_SUMMARY_PROMPT
         Long  → map-reduce chunks → final YT_SUMMARY_PROMPT pass
      4. Returns: transcript display block + summary
    """
    transcript, yt_url = _extract_transcript_text(content)

    if not transcript or len(transcript.strip()) < 50:
        return "[No transcript content found to summarize.]"

    logger.info(f"YouTube summarization: {len(transcript)} chars, url={yt_url}")

    # ── Map-reduce for long transcripts ──────────────────────────────────
    if len(transcript) > YT_LONG_THRESHOLD:
        logger.info("Long transcript detected — using map-reduce")
        summarized_content = await _map_reduce(transcript)
    else:
        logger.info("Short transcript — direct summarization")
        summarized_content = transcript

    # ── Final summary pass with YouTube-specific prompt ───────────────────
    client   = _get_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=YT_SUMMARY_PROMPT.format(text=summarized_content[:6000]),
    )
    summary = response.text.strip()

    # ── Build final output: transcript preview + summary ──────────────────
    transcript_preview = _format_transcript_preview(transcript)

    return f"{summary}\n\n---\n\n{transcript_preview}"


async def _map_reduce(transcript: str) -> str:
    """
    Split transcript into chunks → summarize each → combine.
    Reduces a long transcript into a dense summary for the final pass.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(transcript)
    logger.info(f"Map-reduce: {len(chunks)} chunks")

    client          = _get_client()
    chunk_summaries = []

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Reducing chunk {i}/{len(chunks)}")
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Summarize this section of a video transcript in 3-4 sentences:\n\n{chunk}",
        )
        chunk_summaries.append(resp.text.strip())

    combined = "\n\n".join(chunk_summaries)
    logger.info(f"Map-reduce complete: {len(combined)} chars")
    return combined


def _format_transcript_preview(transcript: str) -> str:
    """
    Formats the raw transcript into a clean collapsible-ready block.
    Shows first 800 chars as preview with total length info.
    """
    total_words = len(transcript.split())
    preview     = transcript[:800].strip()
    truncated   = len(transcript) > 800

    lines = [
        "📝 **TRANSCRIPT**",
        f"*({total_words:,} words total)*",
        "",
        preview,
    ]
    if truncated:
        lines.append(f"\n*... [{len(transcript) - 800:,} more characters — expand Extracted Text panel to view full transcript]*")

    return "\n".join(lines)


async def _chunk_and_summarize(text: str) -> str:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    logger.info(f"Document split into {len(chunks)} chunks")

    client          = _get_client()
    chunk_summaries = []

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Summarizing chunk {i}/{len(chunks)}")
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Summarize this section in 3-4 sentences:\n\n{chunk}",
        )
        chunk_summaries.append(resp.text.strip())

    return "\n\n".join(chunk_summaries)