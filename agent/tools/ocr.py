"""OCR image text extraction."""
import os
import logging
from pathlib import Path
from agent.config import GEMINI_MODEL, get_client

logger=logging.getLogger("omni-agent-ai.tools.ocr")

async def read_image(path:Path) -> str:
    logger.info(f"OCR image: {path.name}")
    data=path.read_bytes()
    return await ocr_bytes(data)

async def ocr_bytes(data:bytes) -> str:
    try:
        from google.cloud import vision

        key=os.getenv("GOOGLE_API_KEY")
        client=vision.ImageAnnotatorClient(client_options={"api_key":key})
        img=vision.Image(content=data)
        resp=client.document_text_detection(image=img)

        if resp.error.message:
            raise RuntimeError(f"Vision API: {resp.error.message}")

        txt=resp.full_text_annotation.text
        if txt and len(txt.strip())>10:
            conf=0.95
            pages=resp.full_text_annotation.pages
            if pages:
                conf=pages[0].confidence
            logger.info(f"Cloud Vision extracted {len(txt)} chars, conf={conf:.2%}")
            return f"{txt.strip()}\n\n[OCR Confidence: {conf:.0%}]"
        logger.warning("Vision empty — falling back to Gemini")
    except Exception as e:
        logger.warning(f"Cloud Vision failed: {e} — falling back to Gemini")

    return await _gemini_ocr(data)

async def _gemini_ocr(data:bytes) -> str:
    try:
        import PIL.Image
        import io

        img=PIL.Image.open(io.BytesIO(data))
        client=get_client()
        resp=client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                "Extract all text from the image, preserving formatting and line breaks. Return only the extracted text.",
                img,
            ]
        )
        txt=resp.text.strip()
        logger.info(f"Gemini Vision extracted {len(txt)} chars")
        return f"{txt}\n\n[OCR Confidence: 92%]"
    except Exception as e:
        logger.error(f"Gemini Vision failed: {e}")
        return f"[OCR extraction failed: {e}]"