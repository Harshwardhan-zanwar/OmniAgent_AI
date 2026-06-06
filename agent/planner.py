#Brain
"""
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

import re
import json
import logging
from pathlib import Path
from typing import Optional

import google.generativeai as genai

#Tool imports
from agent.tools.pdf import extract_pdf_text
from agent.tools.ocr import extract_image_text
from agent.tools.audio import transcribe_audio
from agent.tools.youtube import fetch_youtube_transcript
from agent.tools.summarize import summarize_text
from agent.tools.sentiment import analyze_sentiment
from agent.tools.code_explain import explain_code
from agent.tools.cross_input  import compare_inputs

logger = logging.getLogger("omni-agent-ai.planner")


# CONSTANTS

#Supported intent labels and what triggers them
INTENT_LABELS = [
    "summarize",       
    "sentiment",       
    "code_explain",    
    "qa",              
    "transcribe",      
    "extract",         
    "compare",         
    "conversational",  
    "youtube",         
]

#File extension to extractor mapping
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
PDF_EXTS   = {".pdf"}

#YouTube URL pattern
YT_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]+"
)

# SECTION 1 — EXTRACT CONTENT FROM ALL UPLOADED FILES

async def _extract_all_files(
    file_paths: list[Path],
    plan_trace: list[dict],
) -> tuple[str, list[str]]:
    """
    Loop through every uploaded file, detect its type, call the right tool.
    Returns:
        combined_text : all extracted text joined together
        file_summaries: one-liner per file (shown in UI extracted-text panel)
    """
    combined_parts: list[str] = []
    file_summaries: list[str] = []
    step = 1

    for path in file_paths:
        ext = path.suffix.lower()
        fname = path.name

        try:
            if ext in PDF_EXTS:
                logger.info(f"Extracting PDF: {fname}")
                text = await extract_pdf_text(path)
                tool_name = "pdf_extractor"
                desc = f"Extracted text from PDF '{fname}'"

            elif ext in IMAGE_EXTS:
                logger.info(f"Running OCR on image: {fname}")
                text = await extract_image_text(path)
                tool_name = "ocr_vision"
                desc = f"OCR completed on image '{fname}'"

            elif ext in AUDIO_EXTS:
                logger.info(f"Transcribing audio: {fname}")
                text = await transcribe_audio(path)
                tool_name = "audio_transcriber"
                desc = f"Transcribed audio '{fname}'"

            else:
                # Try reading as plain text
                text = path.read_text(encoding="utf-8", errors="ignore")
                tool_name = "text_reader"
                desc = f"Read text file '{fname}'"

            preview = text[:120].replace("\n", " ") + ("…" if len(text) > 120 else "")
            plan_trace.append({
                "step": step,
                "tool": tool_name,
                "description": desc,
                "status": "success",
                "output_preview": preview,
            })
            combined_parts.append(f"[File: {fname}]\n{text}")
            file_summaries.append(f"{fname}: {preview}")

        except Exception as exc:
            logger.warning(f"Failed to extract {fname}: {exc}")
            plan_trace.append({
                "step": step,
                "tool": "file_extractor",
                "description": f"Failed to extract '{fname}'",
                "status": "failed",
                "output_preview": str(exc),
            })

        step += 1

    return "\n\n".join(combined_parts), file_summaries


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DETECT YOUTUBE URLS
# ─────────────────────────────────────────────────────────────────────────────

def _find_youtube_url(text: str) -> Optional[str]:
    """Scan any text blob for a YouTube URL and return it (or None)."""
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
    """
    If a YouTube URL is found in query or extracted content,
    fetch the transcript and append it to extracted_text.
    Returns updated extracted_text and next step number.
    """
    combined_search = query + "\n" + extracted_text
    yt_url = _find_youtube_url(combined_search)

    if not yt_url:
        return extracted_text, step

    logger.info(f"YouTube URL detected: {yt_url}")
    try:
        transcript = await fetch_youtube_transcript(yt_url)
        plan_trace.append({
            "step": step,
            "tool": "youtube_fetcher",
            "description": f"Fetched YouTube transcript from {yt_url}",
            "status": "success",
            "output_preview": transcript[:120] + "…",
        })
        extracted_text += f"\n\n[YouTube Transcript: {yt_url}]\n{transcript}"
        step += 1
    except Exception as exc:
        logger.warning(f"YouTube fetch failed: {exc}")
        plan_trace.append({
            "step": step,
            "tool": "youtube_fetcher",
            "description": f"Could not fetch transcript from {yt_url}",
            "status": "failed",
            "output_preview": str(exc),
        })
        step += 1

    return extracted_text, step


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DETECT INTENT VIA GEMINI
# ─────────────────────────────────────────────────────────────────────────────

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


async def _detect_intent(query: str, extracted_text: str) -> dict:
    """Ask Gemini to classify the user's intent. Returns intent dict."""
    model = genai.GenerativeModel("gemini-2.5-flash")

    user_message = f"""
User query: {query or '(no query provided)'}

Extracted content preview (first 1000 chars):
{extracted_text[:1000] if extracted_text else '(no files uploaded)'}
""".strip()

    try:
        response = model.generate_content(
            [_INTENT_SYSTEM_PROMPT, user_message],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=200,
            ),
        )
        raw = response.text.strip()
        # Strip markdown fences if Gemini added them anyway
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)

    except Exception as exc:
        logger.warning(f"Intent detection failed: {exc}. Defaulting to 'qa'.")
        return {"intent": "qa", "confidence": 0.5, "reasoning": "Fallback due to classification error."}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PLAN TOOL SEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

def _plan_tools(intent: str, has_files: bool, file_count: int) -> list[str]:
    """
    Return the minimal ordered list of tool names to run for this intent.
    File extraction already happened in Section 1 — these are post-extraction tools.
    """
    plans = {
        "summarize":    ["summarize"],
        "sentiment":    ["sentiment"],
        "code_explain": ["code_explain"],
        "qa":           ["qa_answer"],
        "transcribe":   [],                        # extraction already done
        "extract":      ["qa_answer"],             # QA extracts specific items
        "compare":      ["compare"] if file_count >= 2 else ["qa_answer"],
        "conversational": ["conversational"],
        "youtube":      ["summarize"],             # transcript already fetched
    }
    return plans.get(intent, ["qa_answer"])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EXECUTE TOOLS IN SEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

async def _execute_tools(
    tools: list[str],
    query: str,
    extracted_text: str,
    plan_trace: list[dict],
    start_step: int,
) -> str:
    """
    Run each planned tool, log every step, and return the final tool output.
    Each tool receives the full extracted context + user query.
    """
    step = start_step
    tool_output = extracted_text  # default: pass-through

    for tool_name in tools:
        logger.info(f"Executing tool: {tool_name}")
        try:
            if tool_name == "summarize":
                tool_output = await summarize_text(extracted_text)
                desc = "Generated 1-line + bullets + 5-sentence summary"

            elif tool_name == "sentiment":
                tool_output = await analyze_sentiment(extracted_text)
                desc = "Performed sentiment analysis with confidence score"

            elif tool_name == "code_explain":
                tool_output = await explain_code(extracted_text)
                desc = "Explained code, detected bugs, noted time complexity"

            elif tool_name == "qa_answer":
                tool_output = await _gemini_qa(query, extracted_text)
                desc = f"Answered: '{query[:60]}'"

            elif tool_name == "compare":
                tool_output = await compare_inputs(query, extracted_text)
                desc = "Compared content across multiple inputs"

            elif tool_name == "conversational":
                tool_output = await _gemini_chat(query)
                desc = "Responded conversationally"

            else:
                tool_output = extracted_text
                desc = f"Passed through content (tool: {tool_name})"

            preview = str(tool_output)[:120].replace("\n", " ")
            plan_trace.append({
                "step": step,
                "tool": tool_name,
                "description": desc,
                "status": "success",
                "output_preview": preview + ("…" if len(str(tool_output)) > 120 else ""),
            })

        except Exception as exc:
            logger.error(f"Tool '{tool_name}' failed: {exc}", exc_info=True)
            plan_trace.append({
                "step": step,
                "tool": tool_name,
                "description": f"Tool '{tool_name}' encountered an error",
                "status": "failed",
                "output_preview": str(exc),
            })
            # Graceful degradation: fall back to raw QA on failure
            tool_output = await _gemini_qa(query, extracted_text)

        step += 1

    return str(tool_output)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GEMINI HELPERS (QA + CHAT)
# ─────────────────────────────────────────────────────────────────────────────

async def _gemini_qa(query: str, context: str) -> str:
    """
    General-purpose question answering over extracted context.
    Used for: qa, extract, fallback.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""You are a helpful AI assistant. Answer the user's question using ONLY the provided context.
Be concise, accurate, and well-formatted.

Context:
{context[:6000]}

Question: {query}

Answer:"""
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=1000),
    )
    return response.text.strip()


async def _gemini_chat(query: str) -> str:
    """Friendly conversational response (no document context needed)."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        query,
        generation_config=genai.GenerationConfig(temperature=0.7, max_output_tokens=800),
    )
    return response.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — FOLLOW-UP QUESTION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _generate_follow_up(intent: str, query: str, extracted_text: str) -> str:
    """
    When confidence is low, return a short clarifying question
    rather than guessing the wrong task.
    """
    if not query and not extracted_text:
        return "It looks like nothing was provided. Could you type your question or upload a file?"

    if not query and extracted_text:
        return "I've extracted the content from your file. What would you like me to do with it — summarize, answer questions, or something else?"

    # Try to give a context-aware question
    lowq = query.lower()
    if any(w in lowq for w in ["this", "it", "the file", "the document"]):
        return "Could you clarify what you'd like me to do — summarize the content, answer a specific question, or perform sentiment analysis?"

    return "I want to make sure I understand correctly. Could you clarify what outcome you're looking for?"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def run_agent(
    session_id: str,
    query: str,
    file_paths: list[Path],
) -> dict:
    """
    Master orchestrator. Called by main.py for every /chat request.

    Returns a dict with:
        result            : final text answer
        extracted_text    : all text pulled from uploaded files
        plan_trace        : list of step dicts (tool, status, preview)
        follow_up_question: set only when intent was unclear
        total_tokens_used : rough estimate
    """
    plan_trace: list[dict] = []
    logger.info(f"[{session_id}] Agent started. Files: {[p.name for p in file_paths]}")

    # ── 1. Extract content from all files ────────────────────────────────────
    extracted_text, file_summaries = await _extract_all_files(file_paths, plan_trace)
    next_step = len(plan_trace) + 1

    # ── 2. Detect and handle YouTube URLs ────────────────────────────────────
    extracted_text, next_step = await _handle_youtube_if_present(
        query, extracted_text, plan_trace, next_step
    )

    # ── 3. Detect intent ─────────────────────────────────────────────────────
    intent_result = await _detect_intent(query, extracted_text)
    intent     = intent_result.get("intent", "qa")
    confidence = float(intent_result.get("confidence", 0.5))
    reasoning  = intent_result.get("reasoning", "")

    logger.info(f"[{session_id}] Intent={intent} confidence={confidence:.2f} | {reasoning}")

    plan_trace.append({
        "step": next_step,
        "tool": "intent_classifier",
        "description": f"Detected intent: '{intent}' (confidence {confidence:.0%})",
        "status": "success",
        "output_preview": reasoning,
    })
    next_step += 1

    # ── 4. Follow-up if intent unclear ───────────────────────────────────────
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

    # ── 5. Plan tool sequence ─────────────────────────────────────────────────
    tools = _plan_tools(intent, bool(file_paths), len(file_paths))
    logger.info(f"[{session_id}] Tool plan: {tools}")

    plan_trace.append({
        "step": next_step,
        "tool": "planner",
        "description": f"Planned {len(tools)} tool(s): {' → '.join(tools) if tools else 'direct answer'}",
        "status": "success",
        "output_preview": None,
    })
    next_step += 1

    # ── 6. Execute tools ──────────────────────────────────────────────────────
    if tools:
        result = await _execute_tools(
            tools=tools,
            query=query,
            extracted_text=extracted_text,
            plan_trace=plan_trace,
            start_step=next_step,
        )
    else:
        # transcribe intent: just return the extracted text cleanly
        result = extracted_text if extracted_text else "No content could be extracted."
        plan_trace.append({
            "step": next_step,
            "tool": "passthrough",
            "description": "Returning extracted content directly (transcribe intent)",
            "status": "success",
            "output_preview": result[:120],
        })

    # ── 7. Return structured result ───────────────────────────────────────────
    logger.info(f"[{session_id}] Agent complete. Steps: {len(plan_trace)}")
    return {
        "result": result,
        "extracted_text": extracted_text if extracted_text else None,
        "plan_trace": plan_trace,
        "follow_up_question": None,
        "total_tokens_used": _estimate_tokens(query, extracted_text, result),
    }


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_tokens(*text_parts: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    total_chars = sum(len(t) for t in text_parts if t)
    return max(1, total_chars // 4)