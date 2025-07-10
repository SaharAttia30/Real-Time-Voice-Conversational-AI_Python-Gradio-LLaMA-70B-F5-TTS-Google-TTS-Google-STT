# models/response_models.py

from pydantic import BaseModel

class ChatResponse(BaseModel):
    text: str
    audio_url: str
