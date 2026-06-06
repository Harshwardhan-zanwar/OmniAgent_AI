"""
Image text extraction via Google Cloud Vision API.
  - Accepts file path (JPG / PNG / WEBP / BMP)
  - Also exposes ocr_image_bytes() used by pdf.py fallback
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger("omni-agent-ai.tools.ocr")


async def extract_image_text(file_path: Path) -> str:
    """
    Run OCR on an image file using Google Cloud Vision.
    Returns extracted text string.
    """
    logger.info(f"Running OCR on image: {file_path.name}")
    image_bytes = file_path.read_bytes()
    return await ocr_image_bytes(image_bytes)


async def ocr_image_bytes(image_bytes: bytes) -> str:
    """
    Core OCR function — accepts raw bytes.
    Called by both extract_image_text() and pdf.py OCR fallback.
    """
    #Primary: Google Cloud Vision
    try:
        from google.cloud import vision

        client = vision.ImageAnnotatorClient()
        image  = vision.Image(content=image_bytes)

        response = client.document_text_detection(image=image)

        if response.error.message:
            raise RuntimeError(f"Vision API error: {response.error.message}")

        text = response.full_text_annotation.text
        if text and len(text.strip()) > 10:
            logger.info(f"Cloud Vision OCR: {len(text)} chars extracted")
            return text.strip()

        logger.warning("Cloud Vision returned empty text — trying Gemini Vision")

    except Exception as exc:
        logger.warning(f"Cloud Vision failed: {exc} — trying Gemini Vision fallback")

    #Fallback: Gemini Vision
    return await _gemini_vision_fallback(image_bytes)


async def _gemini_vision_fallback(image_bytes: bytes) -> str:
    """
    Use Gemini 2.5 Flash vision to extract text from image.
    Activated when Cloud Vision is unavailable or returns empty.
    """
    try:
        import google.generativeai as genai
        import PIL.Image
        import io

        image   = PIL.Image.open(io.BytesIO(image_bytes))
        model   = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content([
            "Extract ALL text visible in this image. "
            "Return only the extracted text, nothing else. "
            "Preserve formatting and line breaks where possible.",
            image,
        ])
        text = response.text.strip()
        logger.info(f"Gemini Vision fallback: {len(text)} chars extracted")
        return text

    except Exception as exc:
        logger.error(f"Gemini Vision fallback also failed: {exc}")
        return f"[Error: Could not extract text from image. {exc}]"