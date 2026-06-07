import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from agent.tools.audio import read_audio

@pytest.mark.asyncio
async def test_audio_tool():
    path=Path("tests/test_data/audio_lecture.mp3")
    assert path.exists()

    with patch("agent.tools.audio._gemini_transcribe",new_callable=AsyncMock) as mock_transcribe:
        mock_transcribe.return_value="This is a transcribed audio lecture talking about Agentic AI systems."

        res=await read_audio(path)
        assert "transcribed audio lecture" in res
        assert "[Audio Duration:" in res
