"""
PDF text extraction.
  - Primary   : LangChain PyPDFLoader (fast, text-based PDFs)
  - Fallback  : Google Cloud Vision OCR (scanned / image-only PDFs)
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("omni-agent-ai.tools.pdf")

# Max characters, to keep token cost reasonable
MAX_CHARS = 12_000


async def extract_pdf_text(file_path: Path) -> str:
    """
    Extract text from a PDF file.
    Returns cleaned, joined text string.
    """
    logger.info(f"Extracting PDF: {file_path.name}")

    try:
        loader = PyPDFLoader(str(file_path))
        pages  = loader.load()                      
        raw    = "\n\n".join(p.page_content for p in pages if p.page_content.strip())

        if len(raw.strip()) > 100:
            logger.info(f"PyPDFLoader extracted {len(raw)} chars from {len(pages)} pages")
            return _clean_and_trim(raw)

        logger.warning("PyPDFLoader returned sparse text — trying OCR fallback")

    except Exception as exc:
        logger.warning(f"PyPDFLoader failed: {exc} — trying OCR fallback")

    # ── Fallback: Vision OCR ──────────────────────────────────────────────
    return await _ocr_fallback(file_path)

#ocr -> optical char recognisation
async def _ocr_fallback(file_path: Path) -> str:
    """
    Convert each PDF page to an image and run Google Cloud Vision OCR.
    Used when the PDF is scanned / image-only.
    """
    try:
        from PyMuPDF import fitz 
        from agent.tools.ocr import ocr_image_bytes

        doc   = fitz.open(str(file_path))
        parts = []

        for page_num, page in enumerate(doc, start=1):
            pix        = page.get_pixmap(dpi=200)
            img_bytes  = pix.tobytes("png")
            page_text  = await ocr_image_bytes(img_bytes)
            parts.append(f"[Page {page_num}]\n{page_text}")
            logger.info(f"OCR page {page_num}: {len(page_text)} chars")

        doc.close()
        return _clean_and_trim("\n\n".join(parts))

    except ImportError:
        logger.error("PyMuPDF not installed — cannot do OCR fallback on PDF")
        return "[Error: Could not extract text from this PDF. Install PyMuPDF for scanned PDF support.]"
    except Exception as exc:
        logger.error(f"OCR fallback failed: {exc}")
        return f"[Error extracting PDF: {exc}]"


def _clean_and_trim(text: str) -> str:
    """Remove excessive whitespace and trim to MAX_CHARS."""
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse blank lines
    text = re.sub(r" {2,}", " ", text)        # collapse spaces
    return text[:MAX_CHARS].strip()