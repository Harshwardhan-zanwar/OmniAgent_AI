"""
  - Label       : Positive / Negative / Neutral / Mixed
  - Confidence  : 0 - 1
  - Justification: one-line explanation
  - Emotion tags: top 3 detected emotions
"""

import json
import logging
import re

import google.generativeai as genai
from langchain.prompts import PromptTemplate

logger = logging.getLogger("omni-agent-ai.tools.sentiment")

SENTIMENT_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a sentiment analysis expert.
Analyze the sentiment of the text below and respond ONLY with valid JSON, no markdown fences:
{{
  "label":         "Positive" | "Negative" | "Neutral" | "Mixed",
  "confidence":    float between 0.0 and 1.0,
  "justification": "one sentence explaining why",
  "emotions":      ["emotion1", "emotion2", "emotion3"],
  "tone":          "formal | informal | aggressive | empathetic | neutral"
}}
Text to analyze:{text}"""
)

async def analyze_sentiment(text: str) -> str:
    """
    Perform sentiment analysis on the given text.
    Returns a formatted, human-readable sentiment report.
    """
    if not text or len(text.strip()) < 10:
        return "[Not enough content for sentiment analysis.]"

    logger.info(f"Analyzing sentiment: {len(text)} chars")

    sample  = text[:4000]
    prompt  = SENTIMENT_PROMPT.format(text=sample)
    model   = genai.GenerativeModel("gemini-2.5-flash")

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=300,
            ),
        )
        raw = response.text.strip()

        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        return _format_sentiment_report(data)

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning(f"Structured sentiment failed: {exc} — using plain response")
        return await _plain_sentiment_fallback(text)


def _format_sentiment_report(data: dict) -> str:
    """Format the JSON sentiment result into a clean readable report."""
    label= data.get("label","Unknown")
    confidence= float(data.get("confidence",0.0))
    justification= data.get("justification","N/A")
    emotions= data.get("emotions",[])
    tone= data.get("tone","N/A")

    # Confidence bar visualization
    filled= int(confidence * 10)
    bar= "█" * filled + "░" * (10 - filled)

    # Sentiment emoji
    emoji_map = {
        "Positive":"🟢",
        "Negative":"🔴",
        "Neutral":"🟡",
        "Mixed":"🟠",
    }
    emoji=emoji_map.get(label,"⚪")

    return f"""{emoji}**Sentiment:{label}**
**Confidence:**{bar}{confidence:.0%}
**Justification:**{justification}
**Detected Emotions:**{', '.join(emotions) if emotions else 'N/A'}
**Tone:**{tone.capitalize()}```"""

async def _plain_sentiment_fallback(text: str) -> str:
    """Simple fallback when JSON parsing fails."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        f"Analyze the sentiment of this text. Give: label (Positive/Negative/Neutral/Mixed), "
        f"confidence percentage, and one-line justification.\nText:{text[:2000]}"
    )
    return response.text.strip()