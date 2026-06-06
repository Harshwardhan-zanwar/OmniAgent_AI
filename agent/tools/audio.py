"""
Audio transcription pipeline.
  - Converts MP3 / WAV / M4A to text via Gemini Audio
  - Falls back to Google Speech-to-Text if Gemini fails
  - Returns clean transcript string
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("omni-agent-ai.tools.audio")

SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


async def transcribe_audio(file_path: Path) -> str:
    """
    Transcribe an audio file to text.
    Primary   : Gemini 2.5 Flash (native audio understanding)
    Fallback  : Google Cloud Speech-to-Text
    """
    logger.info(f"Transcribing audio: {file_path.name}")
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        return f"[Unsupported audio format: {ext}. Supported: {', '.join(SUPPORTED_FORMATS)}]"

    try:
        transcript = await _transcribe_with_gemini(file_path)
        if transcript and len(transcript.strip()) > 20:
            return transcript
        logger.warning("Gemini returned sparse transcript — trying STT fallback")
    except Exception as exc:
        logger.warning(f"Gemini audio failed: {exc} — trying STT fallback")

    #Fallback
    return await _transcribe_with_google_stt(file_path)


async def _transcribe_with_gemini(file_path: Path) -> str:
    """
    Upload audio to Gemini Files API and request transcription.
    Gemini 2.5 Flash natively understands audio.
    """
    import google.generativeai as genai

    mime_map = {
        ".mp3":  "audio/mpeg",
        ".wav":  "audio/wav",
        ".m4a":  "audio/mp4",
        ".ogg":  "audio/ogg",
        ".flac": "audio/flac",
    }
    mime_type = mime_map.get(file_path.suffix.lower(), "audio/mpeg")

    logger.info(f"Uploading audio to Gemini Files API: {file_path.name}")
    audio_file = genai.upload_file(str(file_path), mime_type=mime_type)

    model    = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([
        audio_file,
        "Please transcribe this audio accurately. "
        "Return only the spoken text. "
        "Preserve speaker changes with labels like [Speaker 1], [Speaker 2] if multiple speakers. "
        "Include timestamps every 30 seconds like [00:30] if audio is long.",
    ])

    try:
        genai.delete_file(audio_file.name)
    except Exception:
        pass

    transcript = response.text.strip()
    logger.info(f"Gemini transcription: {len(transcript)} chars")
    return transcript


async def _transcribe_with_google_stt(file_path: Path) -> str:
    """
    Google Cloud Speech-to-Text fallback.
    Converts audio to LINEAR16 WAV if needed, then transcribes.
    """
    try:
        from pydub import AudioSegment
        from google.cloud import speech

        # Convert to WAV mono 16kHz (required by Google STT)
        logger.info("Converting audio for Google STT...")
        audio  = AudioSegment.from_file(str(file_path))
        audio  = audio.set_channels(1).set_frame_rate(16000)
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

    except Exception as exc:
        logger.error(f"Google STT fallback failed: {exc}")
        return f"[Error transcribing audio: {exc}]"



        