import io
from google.cloud import texttospeech
from .config import USE_F5_TTS
try:
    from f5_tts_mlx.generate import generate as f5_generate
except ImportError:
    f5_generate = None


def language_code_map(lang: str) -> str:
    codes = {"en": "en-US", "he": "he-IL"}
    return codes.get(lang, "en-US")


def synthesize_text(text: str, language: str, use_f5: bool=False) -> bytes:
    if language == "en" and use_f5 and f5_generate:
        audio_array = f5_generate(text=text)
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio_array, samplerate=24000, format="WAV")
        return buf.getvalue()
    client = texttospeech.TextToSpeechClient()
    synth_inp = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code_map(language),
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_conf = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    resp = client.synthesize_speech(input=synth_inp, voice=voice, audio_config=audio_conf)
    return resp.audio_content
