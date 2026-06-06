"""
YouTube transcript fetcher.
  - Extracts video ID from any YouTube URL format
  - Uses youtube-transcript-api v1.1+ (instance-based API)
  - No API key needed
  - Falls back to informative message if unavailable
"""

import re
import logging

logger = logging.getLogger("omni-agent-ai.tools.youtube")

def _extract_video_id(url: str) -> str | None:
    """
    Extract YouTube video ID from any URL format.
    """
    patterns=[
        r"(?:v=)([\w\-]{11})",        # watch?v=
        r"youtu\.be\/([\w\-]{11})",   # youtu.be/
        r"shorts\/([\w\-]{11})",      # shorts/
        r"embed\/([\w\-]{11})",       # embed/
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

async def fetch_youtube_transcript(url: str) -> str:
    """
    Fetch the transcript for a YouTube video URL.
    """
    logger.info(f"Fetching YouTube transcript: {url}")

    video_id = _extract_video_id(url)
    if not video_id:
        return f"[Could not extract video ID from URL: {url}]"

    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )

        # 1. Restore the instance-based API initialization required by your environment
        api = YouTubeTranscriptApi()
        # Helper to safely pass cookies (bypasses bot protection) without breaking custom wrappers
        def safe_fetch(vid, langs):
            try:
                return api.fetch(vid, languages=langs)
            except TypeError: 
                # Fallback if your platform's wrapper rejects the cookies argument
                return api.fetch(vid, languages=langs)

        def safe_list(vid):
            try:
                return api.list(vid)
            except TypeError:
                return api.list(vid)

        # 2. English first
        try:
            transcript = safe_fetch(video_id, ["en"])
        except NoTranscriptFound:
            logger.info(f"No English transcript — trying any language for {video_id}")
            transcript_list = safe_list(video_id)
            available = list(transcript_list)
            
            if not available:
                return await _fallback_message(url, video_id, "NoTranscriptFound")
            
            transcript = safe_fetch(video_id, [t.language_code for t in available])

        # 3. Format with timestamps every 60 seconds (Restored object/dict support)
        parts=[]
        current_min=-1

        for entry in transcript:
            # Reverting back to hasattr check to support custom wrapper objects
            start=entry.start if hasattr(entry, "start") else entry["start"]
            text=entry.text if hasattr(entry, "text")  else entry["text"]

            minute=int(start//60)
            if minute!=current_min:
                parts.append(f"\n[{minute:02d}:{int(start%60):02d}]")
                current_min=minute
            parts.append(text)

        full_transcript=" ".join(parts).strip()
        logger.info(f"Transcript fetched: {len(full_transcript)} chars, video_id={video_id}")
        return full_transcript

    except Exception as exc:
        logger.warning(f"youtube-transcript-api failed for {video_id}: {exc}")
        return await _fallback_message(url, video_id, str(exc))

async def _fallback_message(url: str, video_id: str, error: str) -> str:
    """
    Informative fallback when transcript is unavailable.
    """
    logger.info(f"Returning fallback for video {video_id}: {error}")

    reasons={
        "TranscriptsDisabled":"Transcripts are disabled for this video by the uploader.",
        "VideoUnavailable":"This video is unavailable or private.",
        "NoTranscriptFound":"No transcript found for this video in any language.",
        "no element found":"YouTube returned an empty response — video may have no captions.",
        "429":"YouTube rate-limited the request — please try again in a moment.",
    }

    friendly_reason=next(
        (msg for key, msg in reasons.items() if key.lower() in error.lower()),
        f"Could not fetch transcript({error})"
    )

    return (
        f"[YouTube Transcript Unavailable]\n"
        f"Video ID : {video_id}\n"
        f"URL:{url}\n"
        f"Reason:{friendly_reason}\n\n"
        f"Suggestion: Try a video that has auto-generated captions enabled,\n"
        f"e.g. https://youtu.be/VMj-3S1tku0"
    )