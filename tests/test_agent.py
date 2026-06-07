import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from agent.planner import run_agent

@pytest.fixture
def mock_genai():
    with patch("google.genai.Client") as mock_cls:
        client=MagicMock()
        
        def side_effect(model,contents):
            res_obj=MagicMock()
            contents_str=str(contents)
            
            # If this is the intent classifier call, return JSON
            if "Classify user intent" in contents_str:
                if "Explain" in contents_str:
                    res_obj.text='{"intent": "code_explain", "confidence": 0.95, "reasoning": "explain code query"}'
                elif "YT" in contents_str or "youtube" in contents_str or "summary of it" in contents_str:
                    res_obj.text='{"intent": "youtube", "confidence": 0.95, "reasoning": "youtube summary query"}'
                elif "compare" in contents_str or "discuss" in contents_str or "topic" in contents_str:
                    res_obj.text='{"intent": "compare", "confidence": 0.95, "reasoning": "compare files"}'
                elif "Summarize" in contents_str:
                    res_obj.text='{"intent": "summarize", "confidence": 0.95, "reasoning": "summarize content"}'
                else:
                    res_obj.text='{"intent": "qa", "confidence": 0.95, "reasoning": "qa query"}'
            else:
                # Tool content generation: return plain text
                res_obj.text="Mocked output: 1-line summary, 3 bullets, 5-sentence summary, duration, action items, bugs."
            return res_obj
            
        client.models.generate_content.side_effect=side_effect
        mock_cls.return_value=client
        yield client

@pytest.mark.asyncio
async def test_case_1_audio_transcription(mock_genai):
    path=Path("tests/test_data/audio_lecture.mp3")
    assert path.exists()

    with patch("agent.tools.audio._gemini_transcribe",new_callable=AsyncMock) as mock_trans:
        mock_trans.return_value="Speech about multi-modal agentic systems."
        
        res=await run_agent(
            session_id="test1",
            query="Summarize this audio lecture.",
            file_paths=[path]
        )
        assert res["result"] != ""
        assert any(step["tool"] == "summarize" for step in res["plan_trace"])

@pytest.mark.asyncio
async def test_case_2_pdf_qa(mock_genai):
    path=Path("tests/test_data/meeting_notes.pdf")
    assert path.exists()

    with patch("agent.tools.pdf.PyPDFLoader") as mock_loader_cls:
        mock_loader=mock_loader_cls.return_value
        mock_page=MagicMock()
        mock_page.page_content="Action Items:\n- Refactor helper functions (Bob)\n" + "x"*120
        mock_loader.load.return_value=[mock_page]

        res=await run_agent(
            session_id="test2",
            query="What are the action items?",
            file_paths=[path]
        )
        assert "Mocked output" in res["result"]
        assert any(step["tool"] == "qa_answer" for step in res["plan_trace"])

@pytest.mark.asyncio
async def test_case_3_image_code(mock_genai):
    path=Path("tests/test_data/code_screenshot.png")
    assert path.exists()

    mock_vision=MagicMock()
    mock_vision.error.message=""
    mock_vision.full_text_annotation.text="def calculate_total(prices):"
    mock_vision.full_text_annotation.pages=[MagicMock(confidence=0.96)]

    with patch("google.cloud.vision.ImageAnnotatorClient") as mock_client_cls:
        mock_client=MagicMock()
        mock_client.document_text_detection.return_value=mock_vision
        mock_client_cls.return_value=mock_client

        res=await run_agent(
            session_id="test3",
            query="Explain",
            file_paths=[path]
        )
        assert res["result"] != ""
        assert any(step["tool"] == "code_explain" for step in res["plan_trace"])

@pytest.mark.asyncio
async def test_case_4_youtube_chain(mock_genai):
    path=Path("tests/test_data/youtube_url.pdf")
    assert path.exists()

    with patch("agent.tools.pdf.PyPDFLoader") as mock_loader_cls, \
         patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_yt_cls:
        
        mock_loader=mock_loader_cls.return_value
        mock_page=MagicMock()
        mock_page.page_content="Review video: https://www.youtube.com/watch?v=VMj-3S1tku0\n" + "x"*120
        mock_loader.load.return_value=[mock_page]

        mock_api=mock_yt_cls.return_value
        mock_entry=MagicMock()
        mock_entry.start=10.0
        mock_entry.text="Transcript line."
        mock_api.fetch.return_value=[mock_entry]

        res=await run_agent(
            session_id="test4",
            query="Hit the YT URL in this PDF and give me a summary of it",
            file_paths=[path]
        )
        assert res["result"] != ""
        assert any(step["tool"] == "youtube_fetcher" for step in res["plan_trace"])
        assert any(step["tool"] == "youtube_summarize" for step in res["plan_trace"])

@pytest.mark.asyncio
async def test_case_5_multi_file_compare(mock_genai):
    audio_path=Path("tests/test_data/audio_lecture.mp3")
    pdf_path=Path("tests/test_data/simple_doc.pdf")
    assert audio_path.exists()
    assert pdf_path.exists()

    with patch("agent.tools.audio._gemini_transcribe",new_callable=AsyncMock) as mock_trans, \
         patch("agent.tools.pdf.PyPDFLoader") as mock_loader_cls:
        
        mock_trans.return_value="Discussing multi-modal agent tools."
        
        mock_loader=mock_loader_cls.return_value
        mock_page=MagicMock()
        mock_page.page_content="This document is about agentic systems and multi-modal tools.\n" + "x"*120
        mock_loader.load.return_value=[mock_page]

        res=await run_agent(
            session_id="test5",
            query="Do the audio and the document discuss the same topic?",
            file_paths=[audio_path,pdf_path]
        )
        assert res["result"] != ""
        assert any(step["tool"] == "compare" for step in res["plan_trace"])
