"""
agent/tools/audio.py
──────────────────────────────────────────────
Audio transcription pipeline.
  Primary   : Gemini File Upload API (native audio understanding)
  Fallback  : Google Cloud Speech-to-Text
  Requires  : ffmpeg installed on system PATH
──────────────────────────────────────────────
"""

import os
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger("omni-agent-ai.tools.audio")

SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

MIME_MAP = {
    ".mp3":  "audio/mpeg",
    ".wav":  "audio/wav",
    ".m4a":  "audio/mp4",
    ".ogg":  "audio/ogg",
    ".flac": "audio/flac",
}


async def transcribe_audio(file_path: Path) -> str:
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        return f"[Unsupported audio format: {ext}. Supported: {', '.join(SUPPORTED_FORMATS)}]"

    logger.info(f"Transcribing audio: {file_path.name}")

    # Primary: Gemini File Upload API
    try:
        result = await _transcribe_with_gemini(file_path)
        if result and len(result.strip()) > 20:
            return result
        logger.warning("Gemini returned sparse transcript — trying STT fallback")
    except Exception as exc:
        logger.warning(f"Gemini audio failed: {exc} — trying STT fallback")

    # Fallback: Google Cloud Speech-to-Text
    return await _transcribe_with_google_stt(file_path)


async def _transcribe_with_gemini(file_path: Path) -> str:
    from google import genai as google_genai

    api_key   = os.getenv("GEMINI_API_KEY")
    client    = google_genai.Client(api_key=api_key)
    mime_type = MIME_MAP.get(file_path.suffix.lower(), "audio/mpeg")

    logger.info(f"Uploading audio to Gemini Files API: {file_path.name}")

    # Read file as bytes and upload
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    # Upload using Files API
    import tempfile, os as _os
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_path.suffix,
        dir=file_path.parent,
    )
    tmp.write(audio_bytes)
    tmp.close()
    tmp_path = Path(tmp.name)

    try:
        uploaded = client.files.upload(
            file=tmp_path,
            config={"mime_type": mime_type},
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                uploaded,
                (
                    "Please transcribe this audio accurately. "
                    "Return only the spoken text. "
                    "Label speakers as [Speaker 1], [Speaker 2] if multiple. "
                    "Add timestamps like [00:30] every 30 seconds."
                ),
            ],
        )

        # Delete uploaded file from Gemini to save quota
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

        transcript = response.text.strip()
        logger.info(f"Gemini transcription: {len(transcript)} chars")
        return transcript

    finally:
        tmp_path.unlink(missing_ok=True)


async def _transcribe_with_google_stt(file_path: Path) -> str:
    try:
        from pydub import AudioSegment
        from google.cloud import speech

        logger.info("Converting audio for Google STT...")

        # Convert to WAV mono 16kHz — required by Google STT
        audio    = AudioSegment.from_file(str(file_path))
        audio    = audio.set_channels(1).set_frame_rate(16000)
        wav_path = file_path.with_suffix(".converted.wav")
        audio.export(str(wav_path), format="wav")

        client = speech.SpeechClient()
        with wav_path.open("rb") as f:
            audio_bytes = f.read()

        wav_path.unlink(missing_ok=True)

        audio_obj = speech.RecognitionAudio(content=audio_bytes)
        config    = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

        if len(audio_bytes) > 1_000_000:
            operation = client.long_running_recognize(config=config, audio=audio_obj)
            response  = operation.result(timeout=300)
        else:
            response  = client.recognize(config=config, audio=audio_obj)

        transcript = " ".join(
            result.alternatives[0].transcript
            for result in response.results
            if result.alternatives
        )
        logger.info(f"Google STT transcript: {len(transcript)} chars")
        return transcript.strip() or "[No speech detected in audio]"

    except ImportError as exc:
        logger.error(f"Missing dependency: {exc}")
        return (
            "[Audio transcription failed: missing dependency. "
            "Ensure ffmpeg is installed and pydub/google-cloud-speech are in requirements.txt]"
        )
    except Exception as exc:
        logger.error(f"Google STT fallback failed: {exc}")
        return f"[Error transcribing audio: {exc}]"