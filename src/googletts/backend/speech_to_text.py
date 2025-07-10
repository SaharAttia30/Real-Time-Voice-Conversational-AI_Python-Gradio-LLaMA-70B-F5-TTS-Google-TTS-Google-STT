import io
import wave
from google.cloud import speech_v1 as speech
from .config import *

def language_code_map(lang: str) -> str:
    codes = {"en": "en-US", "he": "he-IL"}
    return codes.get(lang, "en-US")

def transcribe_audio(audio_bytes: bytes, language: str) -> str:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code=language_code_map(language),
        sample_rate_hertz=sample_rate,
        audio_channel_count=channels,
    )
    audio = speech.RecognitionAudio(content=frames)
    response = client.recognize(config=config, audio=audio)
    if response.results:
        return response.results[0].alternatives[0].transcript
    return ""