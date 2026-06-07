import pytest
from unittest.mock import MagicMock, patch
from agent.tools.youtube import get_yt_transcript

@pytest.mark.asyncio
async def test_youtube_tool():
    url="https://www.youtube.com/watch?v=VMj-3S1tku0"

    with patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_api_cls:
        mock_api=mock_api_cls.return_value
        mock_entry=MagicMock()
        mock_entry.start=0.5
        mock_entry.text="Hello and welcome to this agentic coding tutorial."
        mock_api.fetch.return_value=[mock_entry]

        res=await get_yt_transcript(url)
        assert "Hello and welcome" in res
        assert "[00:00]" in res
