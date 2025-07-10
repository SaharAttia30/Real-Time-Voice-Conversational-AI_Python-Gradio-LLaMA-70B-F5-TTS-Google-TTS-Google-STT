# services/nebius_tts.py

import requests
from app.config import NEBIUS_API_KEY

def synthesize_nebius(text: str, lang: str, voice: str) -> bytes:
    api_key = NEBIUS_API_KEY
    url = "https://api.nebius.com/ai/tts"
    payload = {"text": text, "language": lang, "voice": voice}
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.post(url, json=payload, headers=headers)
    if resp.ok:
        return resp.content  # raw audio bytes
    raise RuntimeError(f"Nebius TTS error: {resp.status_code}")

def synthesize_nebius(text, lang, voice):
    # Nebius TTS integration
    return b"audio bytes"
