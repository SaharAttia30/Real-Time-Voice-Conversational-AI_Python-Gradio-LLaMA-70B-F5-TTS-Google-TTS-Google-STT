# models/request_models.py

from pydantic import BaseModel

class ChatRequest(BaseModel):
    text: str
    tts_engine: str
    language: str
    voice: str
    silence_timeout: float