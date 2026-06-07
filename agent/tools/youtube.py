"""YouTube transcript fetcher."""
import re
import logging

logger=logging.getLogger("omni-agent-ai.tools.youtube")

def _get_vid(url:str) -> str | None:
    patterns=[
        r"(?:v=)([\w\-]{11})",
        r"youtu\.be\/([\w\-]{11})",
        r"shorts\/([\w\-]{11})",
        r"embed\/([\w\-]{11})",
    ]
    for p in patterns:
        match=re.search(p,url)
        if match:
            return match.group(1)
    return None

async def get_yt_transcript(url:str) -> str:
    logger.info(f"Fetching YouTube transcript: {url}")
    video_id=_get_vid(url)
    if not video_id:
        return f"[Could not extract video ID from URL: {url}]"

    try:
        from youtube_transcript_api import YouTubeTranscriptApi,NoTranscriptFound

        api=YouTubeTranscriptApi()
        try:
            sub=api.fetch(video_id,languages=["en"])
        except NoTranscriptFound:
            logger.info(f"No English transcript for {video_id} — trying other languages")
            avail=list(api.list(video_id))
            if not avail:
                return await _fallback(url,video_id,"NoTranscriptFound")
            sub=api.fetch(video_id,languages=[t.language_code for t in avail])

        parts=[]
        last_min=-1
        for entry in sub:
            start=entry.start if hasattr(entry,"start") else entry["start"]
            text=entry.text if hasattr(entry,"text") else entry["text"]
            minute=int(start//60)
            if minute!=last_min:
                parts.append(f"\n[{minute:02d}:{int(start%60):02d}]")
                last_min=minute
            parts.append(text)

        text_out=" ".join(parts).strip()
        logger.info(f"Transcript fetched: {len(text_out)} chars, video_id={video_id}")
        return text_out

    except Exception as e:
        logger.warning(f"YouTube transcript failed for {video_id}: {e}")
        return await _fallback(url,video_id,str(e))

async def _fallback(url:str,video_id:str,error:str) -> str:
    reasons={
        "transcriptsdisabled":"Transcripts are disabled for this video.",
        "videounavailable":"This video is unavailable or private.",
        "notranscriptfound":"No transcript found for this video in any language.",
        "no element found":"YouTube returned an empty response.",
        "429":"YouTube rate-limited the request.",
    }
    reason=next((msg for k,msg in reasons.items() if k in error.lower()),f"Could not fetch transcript ({error})")

    return (
        f"[YouTube Transcript Unavailable]\n"
        f"Video ID: {video_id}\n"
        f"URL: {url}\n"
        f"Reason: {reason}\n\n"
        f"Suggestion: Try a video that has auto-generated captions enabled."
    )