"""Sentiment analysis tool."""
import json
import logging
import re
from agent.config import get_client

logger=logging.getLogger("omni-agent-ai.tools.sentiment")

prompt_template="""Analyze the sentiment of the text below.
Respond ONLY with a JSON block containing these keys:
- label: "Positive", "Negative", "Neutral", or "Mixed"
- confidence: float score from 0.0 to 1.0
- justification: single sentence explanation
- emotions: list of top 3 detected emotions
- tone: "formal", "informal", "aggressive", "empathetic", or "neutral"

Do not include markdown code fences in your response.

Text:
{text}"""

async def get_sentiment(txt:str) -> str:
    if not txt or len(txt.strip())<10:
        return "[Not enough content for sentiment analysis.]"

    logger.info(f"Analyzing sentiment: {len(txt)} chars")
    sample=txt[:4000]
    prompt=prompt_template.format(text=sample)
    client=get_client()

    try:
        resp=client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        raw=resp.text.strip()
        raw=re.sub(r"^```json|^```|```$","",raw,flags=re.MULTILINE).strip()
        data=json.loads(raw)
        return _format_report(data)
    except Exception as e:
        logger.warning(f"Structured sentiment failed: {e} — using plain fallback")
        return await _fallback_sentiment(txt)

def _format_report(res:dict) -> str:
    lbl=res.get("label","Unknown")
    conf=float(res.get("confidence",0.0))
    desc=res.get("justification","N/A")
    emos=res.get("emotions",[])
    tone=res.get("tone","N/A")

    n=int(conf*10)
    bar="█"*n+"░"*(10-n)

    emojis={
        "Positive":"🟢",
        "Negative":"🔴",
        "Neutral":"🟡",
        "Mixed":"🟠",
    }
    emoji=emojis.get(lbl,"⚪")

    return f"""{emoji} **Sentiment: {lbl}**
**Confidence:** {bar} {conf:.0%}
**Justification:** {desc}
**Detected Emotions:** {', '.join(emos) if emos else 'N/A'}
**Tone:** {tone.capitalize()}"""

async def _fallback_sentiment(txt:str) -> str:
    client=get_client()
    resp=client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"Analyze the sentiment of this text. Return a label (Positive/Negative/Neutral/Mixed), confidence score, and short justification.\n\nText: {txt[:2000]}",
    )
    return resp.text.strip()