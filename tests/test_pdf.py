import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from agent.tools.pdf import read_pdf

@pytest.mark.asyncio
async def test_pdf_tool():
    path=Path("tests/test_data/meeting_notes.pdf")
    assert path.exists()

    with patch("agent.tools.pdf.PyPDFLoader") as mock_loader_cls:
        mock_loader=mock_loader_cls.return_value
        mock_page=MagicMock()
        mock_page.page_content="Action Items:\n- Refactor helper functions (Bob)\n" + "x"*120
        mock_loader.load.return_value=[mock_page]

        res=await read_pdf(path)
        assert "Action Items" in res
