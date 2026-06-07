"""Audio transcription tool."""
import logging
from pathlib import Path
from agent.config import get_client

logger=logging.getLogger("omni-agent-ai.tools.audio")

supported_formats={".mp3",".wav",".m4a",".ogg",".flac"}
mime_map={
    ".mp3":"audio/mpeg",
    ".wav":"audio/wav",
    ".m4a":"audio/mp4",
    ".ogg":"audio/ogg",
    ".flac":"audio/flac",
}

async def read_audio(path:Path) -> str:
    ext=path.suffix.lower()
    if ext not in supported_formats:
        return f"[Unsupported audio format: {ext}]"

    logger.info(f"Transcribing audio: {path.name}")
    
    duration_str="unknown"
    try:
        from pydub import AudioSegment
        audio=AudioSegment.from_file(str(path))
        sec=len(audio)/1000.0
        m,s=divmod(int(sec),60)
        duration_str=f"{m}:{s:02d}"
    except Exception as e:
        logger.warning(f"Duration extraction failed: {e}")

    try:
        res=await _gemini_transcribe(path)
        if res and len(res.strip())>20:
            return f"[Audio Duration: {duration_str}]\n\n{res}"
    except Exception as e:
        logger.warning(f"Gemini audio failed: {e} — using STT fallback")

    res=await _google_stt(path)
    return f"[Audio Duration: {duration_str}]\n\n{res}"

async def _gemini_transcribe(path:Path) -> str:
    import tempfile

    client=get_client()
    mime=mime_map.get(path.suffix.lower(),"audio/mpeg")

    with open(path,"rb") as f:
        data=f.read()

    tmp=tempfile.NamedTemporaryFile(
        delete=False,
        suffix=path.suffix,
        dir=path.parent,
    )
    tmp.write(data)
    tmp.close()
    tmp_path=Path(tmp.name)

    try:
        up=client.files.upload(
            file=tmp_path,
            config={"mime_type":mime},
        )
        resp=client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                up,
                "Transcribe this audio accurately. Label speakers as [Speaker 1], [Speaker 2] if multiple. Add timestamps like [00:30] every 30 seconds."
            ],
        )
        try:
            client.files.delete(name=up.name)
        except Exception:
            pass
        return resp.text.strip()
    finally:
        tmp_path.unlink(missing_ok=True)

async def _google_stt(path:Path) -> str:
    try:
        from pydub import AudioSegment
        from google.cloud import speech

        audio=AudioSegment.from_file(str(path))
        audio=audio.set_channels(1).set_frame_rate(16000)
        wav_path=path.with_suffix(".converted.wav")
        audio.export(str(wav_path),format="wav")

        client=speech.SpeechClient()
        with wav_path.open("rb") as f:
            data=f.read()

        wav_path.unlink(missing_ok=True)
        audio_obj=speech.RecognitionAudio(content=data)
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

        if len(data)>1000000:
            op=client.long_running_recognize(config=config,audio=audio_obj)
            resp=op.result(timeout=300)
        else:
            resp=client.recognize(config=config,audio=audio_obj)

        text=" ".join(
            r.alternatives[0].transcript
            for r in resp.results
            if r.alternatives
        )
        return text.strip() or "[No speech detected]"
    except ImportError as e:
        logger.error(f"Missing dependency for speech: {e}")
        return "[Audio transcription failed: missing pydub or google-cloud-speech]"
    except Exception as e:
        logger.error(f"Google STT failed: {e}")
        return f"[Error transcribing audio: {e}]"