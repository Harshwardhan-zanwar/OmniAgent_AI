"""
agent/planner.py — Brain of omni-agent-ai

Flow:
  1. Extract content from all given files
  2. Detect YouTube URLs
  3. Detect intent via Gemini (classify task)
  4. Check confidence then ask follow-up if unclear
  5. Plan minimum tools needed for the task
  6. Execute tools in order, logging every step
  7. Generate final answer via Gemini
  8. Return structured result dict
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional
from google import genai as google_genai

from agent.tools.pdf import extract_pdf_text
from agent.tools.ocr import extract_image_text
from agent.tools.audio import transcribe_audio
from agent.tools.youtube import fetch_youtube_transcript
from agent.tools.summarize import summarize_text, summarize_youtube_transcript
from agent.tools.sentiment import analyze_sentiment
from agent.tools.code_explain import explain_code
from agent.tools.cross_input  import compare_inputs
from utils.state import add_turn, get_history_as_gemini_format

logger       = logging.getLogger("omni-agent-ai.planner")
GEMINI_MODEL = "gemini-2.5-flash-lite"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
PDF_EXTS   = {".pdf"}

YT_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]+"
)

KEYWORD_INTENTS = {
    "compare":        ["compare", "difference", "same topic", "similar", "contrast"],
    "code_explain":   ["explain code", "what does this code", "debug", "bug", "fix code", "code review"],
    "sentiment":      ["sentiment", "feeling", "tone", "emotion", "opinion", "positive", "negative"],
    "summarize":      ["summarize", "summary", "tldr", "tl;dr", "brief", "overview", "recap"],
    "transcribe":     ["transcribe", "transcript", "raw text", "extract text"],
    "extract":        ["action items", "key points", "extract", "list all", "find all"],
    "youtube":        ["youtube", "youtu.be", "watch?v="],
    "conversational": ["hello", "hi ", "hey", "how are you"],
}

_INTENT_SYSTEM_PROMPT = """
You are an intent classifier for an AI agent.

Given a user query and optional extracted content from files, identify:
1. intent   — one of: summarize | sentiment | code_explain | qa | transcribe | extract | compare | conversational | youtube
2. confidence — 0.0 to 1.0
3. reasoning  — one sentence explaining your choice

Rules:
- If the text contains code and user says "explain" / "what does this do" → code_explain
- If user asks "summarize" / "summary" / "tldr" → summarize
- If user asks about tone / feeling / opinion → sentiment
- If user asks a specific question about a document → qa
- If multiple files are uploaded and user asks to compare → compare
- If a YouTube URL exists and user asks to summarize / explain it → youtube
- If query is empty and only a file is uploaded → summarize (default)
- If unclear between two tasks → confidence < 0.6

Respond ONLY with valid JSON, no markdown fences:
{
  "intent": "<label>",
  "confidence": <float>,
  "reasoning": "<one sentence>"
}
""".strip()


async def _extract_all_files(
    file_paths: list[Path],
    plan_trace: list[dict],
) -> tuple[str, list[str]]:
    combined_parts: list[str] = []
    file_summaries: list[str] = []
    step = 1

    for path in file_paths:
        ext   = path.suffix.lower()
        fname = path.name

        try:
            if ext in PDF_EXTS:
                logger.info(f"Extracting PDF: {fname}")
                text      = await extract_pdf_text(path)
                tool_name = "pdf_extractor"
                desc      = f"Extracted text from PDF '{fname}'"

            elif ext in IMAGE_EXTS:
                logger.info(f"Running OCR on image: {fname}")
                text      = await extract_image_text(path)
                tool_name = "ocr_vision"
                desc      = f"OCR completed on image '{fname}'"

            elif ext in AUDIO_EXTS:
                logger.info(f"Transcribing audio: {fname}")
                text      = await transcribe_audio(path)
                tool_name = "audio_transcriber"
                desc      = f"Transcribed audio '{fname}'"

            else:
                text      = path.read_text(encoding="utf-8", errors="ignore")
                tool_name = "text_reader"
                desc      = f"Read text file '{fname}'"

            preview = text[:120].replace("\n", " ") + ("…" if len(text) > 120 else "")
            plan_trace.append({
                "step": step, "tool": tool_name, "description": desc,
                "status": "success", "output_preview": preview,
            })
            combined_parts.append(f"[File: {fname}]\n{text}")
            file_summaries.append(f"{fname}: {preview}")

        except Exception as exc:
            logger.warning(f"Failed to extract {fname}: {exc}")
            plan_trace.append({
                "step": step, "tool": "file_extractor",
                "description": f"Failed to extract '{fname}'",
                "status": "failed", "output_preview": str(exc),
            })

        step += 1

    return "\n\n".join(combined_parts), file_summaries


def _find_youtube_url(text: str) -> Optional[str]:
    match = YT_PATTERN.search(text)
    if match:
        url = match.group(0)
        if not url.startswith("http"):
            url = "https://" + url
        return url
    return None


async def _handle_youtube_if_present(
    query: str,
    extracted_text: str,
    plan_trace: list[dict],
    step: int,
) -> tuple[str, int]:
    yt_url = _find_youtube_url(query + "\n" + extracted_text)

    if not yt_url:
        return extracted_text, step

    logger.info(f"YouTube URL detected: {yt_url}")
    try:
        transcript = await fetch_youtube_transcript(yt_url)
        plan_trace.append({
            "step": step, "tool": "youtube_fetcher",
            "description": f"Fetched YouTube transcript from {yt_url}",
            "status": "success", "output_preview": transcript[:120] + "…",
        })
        extracted_text += f"\n\n[YouTube Transcript: {yt_url}]\n{transcript}"
    except Exception as exc:
        logger.warning(f"YouTube fetch failed: {exc}")
        plan_trace.append({
            "step": step, "tool": "youtube_fetcher",
            "description": f"Could not fetch transcript from {yt_url}",
            "status": "failed", "output_preview": str(exc),
        })

    return extracted_text, step + 1


def _keyword_intent(query: str, extracted_text: str) -> Optional[dict]:
    if not query.strip():
        return None

    lowq = query.lower()

    for intent, keywords in KEYWORD_INTENTS.items():
        if any(kw in lowq for kw in keywords):
            return {
                "intent": intent, "confidence": 0.95,
                "reasoning": f"Keyword match detected in user query for '{intent}'.",
            }
    return None


async def _detect_intent(query: str, extracted_text: str) -> dict:
    keyword_result = _keyword_intent(query, extracted_text)
    if keyword_result:
        logger.info(f"Intent via keyword: {keyword_result['intent']}")
        return keyword_result

    user_message = (
        f"User query: {query or '(no query provided)'}\n\n"
        f"Extracted content preview (first 1000 chars):\n"
        f"{extracted_text[:1000] if extracted_text else '(no files uploaded)'}"
    )

    try:
        client   = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{_INTENT_SYSTEM_PROMPT}\n\n{user_message}",
        )
        raw    = re.sub(r"^```json|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
        result = json.loads(raw)
        logger.info(f"Intent via Gemini: {result['intent']} ({result.get('confidence')})")
        return result

    except Exception as exc:
        logger.warning(f"Intent detection failed: {exc}")
        fallback = "summarize" if extracted_text.strip() else "qa"
        return {
            "intent": fallback, "confidence": 0.80,
            "reasoning": f"Auto-fallback to '{fallback}' based on content availability.",
        }


def _plan_tools(intent: str, has_files: bool, file_count: int) -> list[str]:
    plans = {
        "summarize":      ["summarize"],
        "sentiment":      ["sentiment"],
        "code_explain":   ["code_explain"],
        "qa":             ["qa_answer"],
        "transcribe":     [],
        "extract":        ["qa_answer"],
        "compare":        ["compare"] if file_count >= 2 else ["qa_answer"],
        "conversational": ["conversational"],
        "youtube":        ["youtube_summarize"],
    }
    return plans.get(intent, ["qa_answer"])


async def _execute_tools(
    tools: list[str],
    query: str,
    extracted_text: str,
    plan_trace: list[dict],
    start_step: int,
) -> str:
    step               = start_step
    tool_output        = extracted_text
    content_to_analyze = extracted_text if extracted_text else query

    for tool_name in tools:
        logger.info(f"Executing tool: {tool_name}")
        try:
            if tool_name == "summarize":
                tool_output = await summarize_text(content_to_analyze)
                desc        = "Generated 1-line + bullets + 5-sentence summary"

            elif tool_name == "youtube_summarize":
                tool_output = await summarize_youtube_transcript(content_to_analyze)
                desc        = "Generated YouTube transcript display + map-reduce summary"

            elif tool_name == "sentiment":
                tool_output = await analyze_sentiment(content_to_analyze)
                desc        = "Performed sentiment analysis with confidence score"

            elif tool_name == "code_explain":
                tool_output = await explain_code(content_to_analyze)
                desc        = "Explained code, detected bugs, noted time complexity"

            elif tool_name == "qa_answer":
                tool_output = await _gemini_qa(query, content_to_analyze)
                desc        = f"Answered: '{query[:60]}'"

            elif tool_name == "compare":
                tool_output = await compare_inputs(query, content_to_analyze)
                desc        = "Compared content across multiple inputs"

            elif tool_name == "conversational":
                tool_output = await _gemini_chat(query, content_to_analyze)
                desc        = "Responded conversationally"

            else:
                tool_output = extracted_text
                desc        = f"Passed through content (tool: {tool_name})"

            preview = str(tool_output)[:120].replace("\n", " ")
            plan_trace.append({
                "step": step, "tool": tool_name, "description": desc,
                "status": "success",
                "output_preview": preview + ("…" if len(str(tool_output)) > 120 else ""),
            })

        except Exception as exc:
            logger.error(f"Tool '{tool_name}' failed: {exc}", exc_info=True)
            plan_trace.append({
                "step": step, "tool": tool_name,
                "description": f"Tool '{tool_name}' encountered an error",
                "status": "failed", "output_preview": str(exc),
            })
            tool_output = await _gemini_qa(query, extracted_text)

        step += 1

    return str(tool_output)


async def _gemini_qa(query: str, context: str) -> str:
    client   = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt   = (
        f"You are a helpful AI assistant. Answer the user's question using ONLY the provided context.\n"
        f"Be concise, accurate, and well-formatted.\n\n"
        f"Context:\n{context[:6000]}\n\nQuestion: {query}\n\nAnswer:"
    )
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text.strip()


async def _gemini_chat(query: str, context: str = "", session_id: str = "") -> str:
    from utils.state import get_history_as_gemini_format
    client  = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Build conversation history for multi-turn memory
    history = get_history_as_gemini_format(session_id) if session_id else []

    contents = []
    # Add past turns
    for turn in history:
        contents.append(turn)
    # Add current message
    if context:
        contents.append({"role": "user", "parts": [f"Context:\n{context[:3000]}\n\nUser: {query}"]})
    else:
        contents.append({"role": "user", "parts": [query]})

    response = client.models.generate_content(model=GEMINI_MODEL, contents=contents)
    return response.text.strip()


def _generate_follow_up(intent: str, query: str, extracted_text: str) -> str:
    if not query and not extracted_text:
        return "It looks like nothing was provided. Could you type your question or upload a file?"

    if not query and extracted_text:
        return "I've extracted the content from your file. What would you like me to do with it — summarize, answer questions, or something else?"

    lowq = query.lower()
    if any(w in lowq for w in ["this", "it", "the file", "the document"]):
        return "Could you clarify what you'd like me to do — summarize the content, answer a specific question, or perform sentiment analysis?"

    return "I want to make sure I understand correctly. Could you clarify what outcome you're looking for?"


async def run_agent(
    session_id: str,
    query: str,
    file_paths: list[Path],
) -> dict:
    plan_trace: list[dict] = []
    logger.info(f"[{session_id}] Agent started. Files: {[p.name for p in file_paths]}")

    extracted_text, file_summaries = await _extract_all_files(file_paths, plan_trace)
    next_step = len(plan_trace) + 1

    extracted_text, next_step = await _handle_youtube_if_present(
        query, extracted_text, plan_trace, next_step
    )

    intent_result = await _detect_intent(query, extracted_text)
    intent        = intent_result.get("intent", "qa")
    confidence    = float(intent_result.get("confidence", 0.5))
    reasoning     = intent_result.get("reasoning", "")

    logger.info(f"[{session_id}] Intent={intent} confidence={confidence:.2f} | {reasoning}")

    plan_trace.append({
        "step": next_step, "tool": "intent_classifier",
        "description": f"Detected intent: '{intent}' (confidence {confidence:.0%})",
        "status": "success", "output_preview": reasoning,
    })
    next_step += 1

    CONFIDENCE_THRESHOLD = 0.6
    if confidence < CONFIDENCE_THRESHOLD:
        follow_up = _generate_follow_up(intent, query, extracted_text)
        logger.info(f"[{session_id}] Low confidence — returning follow-up question.")
        return {
            "result": "",
            "extracted_text": extracted_text or None,
            "plan_trace": plan_trace,
            "follow_up_question": follow_up,
            "total_tokens_used": _estimate_tokens(query, extracted_text),
        }

    tools = _plan_tools(intent, bool(file_paths), len(file_paths))
    logger.info(f"[{session_id}] Tool plan: {tools}")

    plan_trace.append({
        "step": next_step, "tool": "planner",
        "description": f"Planned {len(tools)} tool(s): {' → '.join(tools) if tools else 'direct answer'}",
        "status": "success", "output_preview": None,
    })
    next_step += 1

    if tools:
        result = await _execute_tools(
            tools=tools, query=query, extracted_text=extracted_text,
            plan_trace=plan_trace, start_step=next_step,
        )
    else:
        result = extracted_text if extracted_text else "No content could be extracted."
        plan_trace.append({
            "step": next_step, "tool": "passthrough",
            "description": "Returning extracted content directly (transcribe intent)",
            "status": "success", "output_preview": result[:120],
        })

    logger.info(f"[{session_id}] Agent complete. Steps: {len(plan_trace)}")
    # Save turn to Redis so conversation history persists
    add_turn(session_id=session_id, query=query, result=result)

    return {
        "result": result,
        "extracted_text": extracted_text if extracted_text else None,
        "plan_trace": plan_trace,
        "follow_up_question": None,
        "total_tokens_used": _estimate_tokens(query, extracted_text, result),
    }


def _estimate_tokens(*text_parts: str) -> int:
    total_chars = sum(len(t) for t in text_parts if t)
    return max(1, total_chars // 4)