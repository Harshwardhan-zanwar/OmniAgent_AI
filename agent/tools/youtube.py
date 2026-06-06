"""
YouTube transcript fetcher.
  - Extracts video ID from any YouTube URL format
  - Fetches transcript using youtube-transcript-api (no API key needed)
  - Falls back to Gemini summarization of URL metadata if unavailable
"""

import re
import logging
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("omni-agent-ai.tools.youtube")


def _extract_video_id(url: str) -> str | None:
    """
    Extract YouTube video ID from any URL
    """
    patterns = [
        r"(?:v=)([\w\-]{11})",           # watch?v=
        r"youtu\.be\/([\w\-]{11})",      # youtu.be/
        r"shorts\/([\w\-]{11})",         # shorts/
        r"embed\/([\w\-]{11})",          # embed/
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


async def fetch_youtube_transcript(url: str) -> str:
    """
    Fetch the transcript for a YouTube video URL.
    Returns the full transcript as a single string.
    """
    logger.info(f"Fetching YouTube transcript: {url}")

    video_id = _extract_video_id(url)
    if not video_id:
        return f"[Could not extract video ID from URL: {url}]"

    # ── Primary: youtube-transcript-api ──────────────────────────────────
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

        # Try English first, then any available language
        try:
            entries = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        except NoTranscriptFound:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript      = transcript_list.find_generated_transcript(
                [t.language_code for t in transcript_list]
            )
            entries = transcript.fetch()

        # Format: join with timestamps every 60 seconds
        parts        = []
        current_min  = -1
        for entry in entries:
            minute = int(entry["start"] // 60)
            if minute != current_min:
                parts.append(f"\n[{minute:02d}:{int(entry['start'] % 60):02d}]")
                current_min = minute
            parts.append(entry["text"])

        full_transcript = " ".join(parts).strip()
        logger.info(f"Transcript fetched: {len(full_transcript)} chars, video_id={video_id}")
        return full_transcript

    except Exception as exc:
        logger.warning(f"youtube-transcript-api failed: {exc}")
        return await _fallback_message(url, video_id, str(exc))


async def _fallback_message(url: str, video_id: str, error: str) -> str:
    """
    When transcript is unavailable (disabled, private, etc.),
    return an informative fallback message instead of crashing.
    """
    logger.info(f"Returning fallback for video {video_id}: {error}")

    reasons = {
        "TranscriptsDisabled": "Transcripts are disabled for this video by the uploader.",
        "VideoUnavailable":    "This video is unavailable or private.",
        "NoTranscriptFound":   "No transcript was found for this video in any language.",
    }

    friendly_reason = next(
        (msg for key, msg in reasons.items() if key in error),
        f"Could not fetch transcript ({error})"
    )

    return (
        f"[YouTube Transcript Unavailable]\n"
        f"Video ID : {video_id}\n"
        f"URL      : {url}\n"
        f"Reason   : {friendly_reason}\n\n"
        f"Suggestion: Try a video with auto-generated captions enabled."
    )