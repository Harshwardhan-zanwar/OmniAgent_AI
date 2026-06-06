"""
Text summarization.
  - Using LangChain PromptTemplate
  - Splits large documents before summarizing
  - Output format: 1-line + 3 bullets + 5-sentence summary
"""

import logging
import re

import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger("omni-agent-ai.tools.summarize")

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
Content to summarize:{text}
Important: Follow the format exactly. Do not add extra sections."""
)

CHUNK_SIZE    = 6000
CHUNK_OVERLAP = 200

async def summarize_text(text: str) -> str:
    """
    Summarize text in the required 3-format output.
    Handles large documents by chunking.
    """
    if not text or len(text.strip()) < 50:
        return "[Not enough content to summarize.]"

    logger.info(f"Summarizing text: {len(text)} chars")

    #large docs
    if len(text) > CHUNK_SIZE:
        text = await _chunk_and_summarize(text)

    #summary pass
    prompt  = SUMMARY_PROMPT.format(text=text[:6000])
    model   = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=800,
        ),
    )
    result = response.text.strip()
    logger.info(f"Summary generated: {len(result)} chars")
    return result

async def _chunk_and_summarize(text: str) -> str:
    """
    large documents: split → summarize each chunk → combine into one text.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    logger.info(f"Document split into {len(chunks)} chunks for summarization")

    model= genai.GenerativeModel("gemini-2.5-flash")
    chunk_summaries = []

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Summarizing chunk {i}/{len(chunks)}")
        resp = model.generate_content(
            f"Summarize this section in 3-4 sentences:\n{chunk}",
            generation_config=genai.GenerationConfig(temperature=0.2, max_output_tokens=300),
        )
        chunk_summaries.append(resp.text.strip())

    # Combine chunk summaries
    combined = "\n\n".join(chunk_summaries)
    logger.info(f"Chunk summaries combined: {len(combined)} chars")
    return combined