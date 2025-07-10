import io
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from .chat import handle_chat
from .tts import synthesize_text

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(
    audio: UploadFile = File(...),
    language: str = Query("en", description="User language code")
):
    data = await audio.read()
    out = handle_chat(data, language)
    return StreamingResponse(io.BytesIO(out), media_type="audio/mp3")

@app.post("/tts")
async def tts_endpoint(
    text: str = Query(...),
    language: str = Query("en")
):
    out = synthesize_text(text, language, use_f5=False)
    return StreamingResponse(io.BytesIO(out), media_type="audio/mp3")