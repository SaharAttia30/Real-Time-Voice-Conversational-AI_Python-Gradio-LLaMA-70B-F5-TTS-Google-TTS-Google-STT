from .speech_to_text import transcribe_audio
from .tts import synthesize_text
from .config import NEBIUS_API_KEY, NEBIUS_API_URL, USE_GOOGLE_LLM, USE_F5_TTS
from google.cloud import translate_v2 as translate
import requests
def translate_text(text: str, target: str) -> str:
    client = translate.Client()
    result = client.translate(text, target_language=target)
    return result["translatedText"]

def call_nebius(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {NEBIUS_API_KEY}"}
    payload = {"prompt": prompt}
    response = requests.post(NEBIUS_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")

def call_google_llm(prompt: str) -> str:
    return f"[Google LLM response to '{prompt}']"

def handle_chat(audio_bytes: bytes, language: str) -> bytes:
    user_text = transcribe_audio(audio_bytes, language)
    if language != "en":
        user_text_en = translate_text(user_text, target="en")
    else:
        user_text_en = user_text
    if USE_GOOGLE_LLM:
        response_en = call_google_llm(user_text_en)
    else:
        response_en = call_nebius(user_text_en)
    if language != "en":
        response_text = translate_text(response_en, target=language)
    else:
        response_text = response_en
    audio_response = synthesize_text(response_text, language, use_f5=USE_F5_TTS)
    return audio_response
