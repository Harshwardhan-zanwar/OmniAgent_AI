import os
from google import genai as google_genai

GEMINI_MODEL = "gemini-2.5-flash-lite"
GROQ_MODEL = "openai/gpt-oss-20b"

_client = None

def get_client():
    global _client
    if _client is None:
        _client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client