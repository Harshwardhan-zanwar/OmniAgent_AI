"""PDF text extraction tool."""
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

logger=logging.getLogger("omni-agent-ai.tools.pdf")
max_len=12000

async def read_pdf(path:Path) -> str:
    logger.info(f"Extracting PDF: {path.name}")
    try:
        loader=PyPDFLoader(str(path))
        pgs=loader.load()
        text="\n\n".join(p.page_content for p in pgs if p.page_content.strip())
        if len(text.strip())>100:
            logger.info(f"PyPDFLoader extracted {len(text)} chars")
            return _clean(text)
        logger.warning("Sparse text — trying OCR")
    except Exception as e:
        logger.warning(f"PyPDFLoader failed: {e} — trying OCR")

    return await _pdf_ocr(path)

async def _pdf_ocr(path:Path) -> str:
    try:
        from PyMuPDF import fitz
        from agent.tools.ocr import ocr_bytes

        pdf=fitz.open(str(path))
        pages_text=[]
        for i,pg in enumerate(pdf,start=1):
            pixmap=pg.get_pixmap(dpi=200)
            image_data=pixmap.tobytes("png")
            text=await ocr_bytes(image_data)
            pages_text.append(f"[Page {i}]\n{text}")
            logger.info(f"OCR page {i}: {len(text)} chars")
        pdf.close()
        return _clean("\n\n".join(pages_text))
    except ImportError:
        logger.error("PyMuPDF not installed — OCR fallback unavailable")
        return "[Error: PyMuPDF not installed for scanned PDF fallback]"
    except Exception as e:
        logger.error(f"OCR fallback failed: {e}")
        return f"[Error extracting PDF: {e}]"

def _clean(txt:str) -> str:
    import re
    txt=re.sub(r"\n{3,}","\n\n",txt)
    txt=re.sub(r" {2,}"," ",txt)
    return txt[:max_len].strip()