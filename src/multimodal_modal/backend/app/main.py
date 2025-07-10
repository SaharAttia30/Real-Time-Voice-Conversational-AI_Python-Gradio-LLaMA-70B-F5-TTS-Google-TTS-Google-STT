# main.py

from fastapi import FastAPI
from app.api import chat, tts

app = FastAPI()

app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(tts.router, prefix="/tts", tags=["tts"])