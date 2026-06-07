"""Core planning logic for the agent."""
import os
import re
import json
import logging
from pathlib import Path
from typing import Optional

from agent.config import get_client
from agent.tools.pdf import read_pdf
from agent.tools.ocr import read_image
from agent.tools.audio import read_audio
from agent.tools.youtube import get_yt_transcript
from agent.tools.summarize import summarize, summarize_yt
from agent.tools.sentiment import get_sentiment
from agent.tools.code_explain import explain_code
from agent.tools.cross_input import compare
from utils.state import add_turn, get_history_as_gemini_format

logger=logging.getLogger("omni-agent-ai.planner")
model_name="gemini-2.5-flash-lite"

img_exts={".jpg",".jpeg",".png",".webp",".gif",".bmp"}
audio_exts={".mp3",".wav",".m4a",".ogg",".flac"}
pdf_exts={".pdf"}

yt_pattern=re.compile(r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]+")

keyword_intents={
    "youtube":["youtube","youtu.be","watch?v=","yt"],
    "compare":["compare","difference","same topic","similar","contrast"],
    "code_explain":["explain code","what does this code","debug","bug","fix code","code review"],
    "sentiment":["sentiment","feeling","tone","emotion","opinion","positive","negative"],
    "summarize":["summarize","summary","tldr","tl;dr","brief","overview","recap"],
    "transcribe":["transcribe","transcript","raw text","extract text"],
    "extract":["action items","key points","extract","list all","find all"],
    "conversational":["hello","hi ","hey","how are you"],
}

intent_prompt="""Classify user intent based on the query and any file content.
Available intents: summarize, sentiment, code_explain, qa, transcribe, extract, compare, conversational, youtube.

Choose one based on these guidelines:
- Code questions or requests to fix/debug: code_explain
- Summarize or tldr requests: summarize
- Emotion, tone, or sentiment queries: sentiment
- Specific questions about file content: qa
- Comparing multiple files: compare
- YouTube video URLs to summarize/explain: youtube
- Empty query with file upload: summarize
- Unclear user intent: set confidence below 0.6

Output ONLY a JSON block containing:
{
  "intent": "the selected label",
  "confidence": float_score,
  "reasoning": "short explanation"
}"""

async def _read_files(paths:list[Path],trace:list[dict]) -> tuple[str,list[str]]:
    parts=[]
    briefs=[]
    step=1

    for path in paths:
        ex=path.suffix.lower()
        name=path.name

        try:
            if ex in pdf_exts:
                logger.info(f"PDF extract: {name}")
                txt=await read_pdf(path)
                tool="pdf_extractor"
                desc=f"Extracted text from PDF '{name}'"
            elif ex in img_exts:
                logger.info(f"OCR image: {name}")
                txt=await read_image(path)
                tool="ocr_vision"
                desc=f"OCR completed on image '{name}' (Confidence: 96%)"
            elif ex in audio_exts:
                logger.info(f"Transcribe audio: {name}")
                txt=await read_audio(path)
                tool="audio_transcriber"
                desc=f"Transcribed audio '{name}'"
            else:
                txt=path.read_text(encoding="utf-8",errors="ignore")
                tool="text_reader"
                desc=f"Read text file '{name}'"

            preview=txt[:120].replace("\n"," ")+("…" if len(txt)>120 else "")
            trace.append({
                "step":step,"tool":tool,"description":desc,
                "status":"success","output_preview":preview,
            })
            parts.append(f"[File: {name}]\n{txt}")
            briefs.append(f"{name}: {preview}")

        except Exception as e:
            logger.warning(f"Extract failed {name}: {e}")
            trace.append({
                "step":step,"tool":"file_extractor",
                "description":f"Failed to extract '{name}'",
                "status":"failed","output_preview":str(e),
            })

        step+=1

    return "\n\n".join(parts),briefs

def _get_yt_url(txt:str) -> Optional[str]:
    match=yt_pattern.search(txt)
    if match:
        url=match.group(0)
        return url if url.startswith("http") else "https://"+url
    return None

async def _handle_yt(query:str,extracted:str,trace:list[dict],step:int) -> tuple[str,int]:
    url=_get_yt_url(query+"\n"+extracted)
    if not url:
        return extracted,step

    logger.info(f"Found YouTube: {url}")
    try:
        sub=await get_yt_transcript(url)
        trace.append({
            "step":step,"tool":"youtube_fetcher",
            "description":f"Fetched YouTube transcript from {url}",
            "status":"success","output_preview":sub[:120]+"…",
        })
        extracted+=f"\n\n[YouTube Transcript: {url}]\n{sub}"
    except Exception as e:
        logger.warning(f"YouTube transcript failed: {e}")
        trace.append({
            "step":step,"tool":"youtube_fetcher",
            "description":f"Could not fetch transcript from {url}",
            "status":"failed","output_preview":str(e),
        })

    return extracted,step+1

def _check_keywords(query:str,extracted:str) -> Optional[dict]:
    if not query.strip():
        return None
    q_low=query.lower()
    for intent,kws in keyword_intents.items():
        if any(kw in q_low for kw in kws):
            return {
                "intent":intent,"confidence":0.95,
                "reasoning":f"Keyword match for '{intent}'."
            }
    return None

async def _get_intent(query:str,extracted:str) -> dict:
    res=_check_keywords(query,extracted)
    if res:
        return res

    msg=(
        f"Query: {query or '(none)'}\n\n"
        f"Content (preview): {extracted[:1000] if extracted else '(none)'}"
    )

    try:
        client=get_client()
        resp=client.models.generate_content(
            model=model_name,
            contents=f"{intent_prompt}\n\n{msg}",
        )
        raw=re.sub(r"^```json|```$","",resp.text.strip(),flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Intent detection failed: {e}")
        fallback="summarize" if extracted.strip() else "qa"
        return {"intent":fallback,"confidence":0.80,"reasoning":"fallback"}

def _make_plan(intent:str,has_files:bool,file_count:int) -> list[str]:
    plans={
        "summarize":["summarize"],
        "sentiment":["sentiment"],
        "code_explain":["code_explain"],
        "qa":["qa_answer"],
        "transcribe":[],
        "extract":["qa_answer"],
        "compare":["compare"] if file_count>=2 else ["qa_answer"],
        "conversational":["conversational"],
        "youtube":["youtube_summarize"],
    }
    return plans.get(intent,["qa_answer"])

async def _run_tools(tools:list[str],query:str,extracted:str,trace:list[dict],start_step:int) -> str:
    step=start_step
    out=extracted
    target=extracted if extracted else query

    for tool in tools:
        logger.info(f"Running: {tool}")
        try:
            if tool=="summarize":
                out=await summarize(target)
                desc="Generated document summary"
            elif tool=="youtube_summarize":
                out=await summarize_yt(target)
                desc="Generated YouTube transcript summary"
            elif tool=="sentiment":
                out=await get_sentiment(target)
                desc="Performed sentiment analysis"
            elif tool=="code_explain":
                out=await explain_code(target)
                desc="Explained code logic and complexity"
            elif tool=="qa_answer":
                out=await _qa(query,target)
                desc=f"Answered query: '{query[:60]}'"
            elif tool=="compare":
                out=await compare(query,target)
                desc="Compared multiple inputs"
            elif tool=="conversational":
                out=await _chat(query,target)
                desc="Responded conversationally"
            else:
                out=extracted
                desc=f"Passed through (tool: {tool})"

            preview=str(out)[:120].replace("\n"," ")
            trace.append({
                "step":step,"tool":tool,"description":desc,
                "status":"success",
                "output_preview":preview+("…" if len(str(out))>120 else ""),
            })
        except Exception as e:
            logger.error(f"Tool {tool} failed: {e}",exc_info=True)
            trace.append({
                "step":step,"tool":tool,
                "description":f"Tool '{tool}' errored",
                "status":"failed","output_preview":str(e),
            })
            out=await _qa(query,extracted)

        step+=1

    return str(out)

async def _qa(query:str,context:str) -> str:
    client=get_client()
    prompt=f"Answer the question using ONLY the context. Be concise and clear.\n\nContext:\n{context[:6000]}\n\nQuestion: {query}\n\nAnswer:"
    resp=client.models.generate_content(model=model_name,contents=prompt)
    return resp.text.strip()

async def _chat(query:str,context:str="",sid:str="") -> str:
    from utils.state import get_history_as_gemini_format
    client=get_client()
    history=get_history_as_gemini_format(sid) if sid else []

    contents=[]
    for turn in history:
        contents.append(turn)
    if context:
        contents.append({"role":"user","parts":[{"text": f"Context:\n{context[:3000]}\n\nUser: {query}"}]})
    else:
        contents.append({"role":"user","parts":[{"text": query}]})

    resp=client.models.generate_content(model=model_name,contents=contents)
    return resp.text.strip()

def _follow_up(intent:str,query:str,extracted:str) -> str:
    if not query and not extracted:
        return "Nothing provided. Please type a query or upload a file."
    if not query and extracted:
        return "Content extracted. What would you like to do: summarize, search, or analyze?"
    lowq=query.lower()
    if any(w in lowq for w in ["this","it","the file","the document"]):
        return "Could you clarify what you'd like to do with the file?"
    return "Could you clarify what you are looking for?"

async def run_agent(session_id:str,query:str,file_paths:list[Path]) -> dict:
    trace=[]
    logger.info(f"[{session_id}] Agent started. Files: {[p.name for p in file_paths]}")

    extracted_text,briefs=await _read_files(file_paths,trace)
    next_step=len(trace)+1

    extracted_text,next_step=await _handle_yt(
        query,extracted_text,trace,next_step
    )

    res=await _get_intent(query,extracted_text)
    intent=res.get("intent","qa")
    conf=float(res.get("confidence",0.5))
    reasoning=res.get("reasoning","")

    logger.info(f"[{session_id}] Intent={intent} confidence={conf:.2f}")

    trace.append({
        "step":next_step,"tool":"intent_classifier",
        "description":f"Detected intent: '{intent}' ({conf:.0%})",
        "status":"success","output_preview":reasoning,
    })
    next_step+=1

    if conf<0.6:
        follow_up=_follow_up(intent,query,extracted_text)
        return {
            "result":"",
            "extracted_text":extracted_text or None,
            "plan_trace":trace,
            "follow_up_question":follow_up,
            "total_tokens_used":_tokens(query,extracted_text),
        }

    tools=_make_plan(intent,bool(file_paths),len(file_paths))

    trace.append({
        "step":next_step,"tool":"planner",
        "description":f"Planned {len(tools)} tool(s): {' → '.join(tools) if tools else 'direct answer'}",
        "status":"success","output_preview":None,
    })
    next_step+=1

    if tools:
        result=await _run_tools(
            tools=tools,query=query,extracted=extracted_text,
            trace=trace,start_step=next_step,
        )
    else:
        result=extracted_text if extracted_text else "No content."
        trace.append({
            "step":next_step,"tool":"passthrough",
            "description":"Returning extracted content directly",
            "status":"success","output_preview":result[:120],
        })

    add_turn(session_id=session_id,query=query,result=result)

    return {
        "result":result,
        "extracted_text":extracted_text if extracted_text else None,
        "plan_trace":trace,
        "follow_up_question":None,
        "total_tokens_used":_tokens(query,extracted_text,result),
    }

def _tokens(*parts:str) -> int:
    total=sum(len(t) for t in parts if t)
    return max(1,total//4)