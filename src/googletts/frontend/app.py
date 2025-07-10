import gradio as gr
import requests

def chat_with_ai(audio, engine, lang, voice):
    if audio is None: return None
    files = {"audio": open(audio, "rb")}
    params = {"language": lang}
    res = requests.post("http://localhost:8000/chat", files=files, params=params)
    res.raise_for_status()
    path = "response_audio.mp3"
    with open(path, "wb") as f: f.write(res.content)
    return path

with gr.Blocks() as demo:
    gr.Markdown("# Voice Chatbot")
    engine = gr.Dropdown(["nebius_llm", "google_llm"], value="nebius_llm", label="Chat Model")
    tts_eng = gr.Dropdown(["google_tts", "f5_tts"], value="google_tts", label="TTS Engine")
    lang = gr.Radio(["en", "he"], value="en", label="Language")
    mic = gr.Audio(sources=["microphone"], type="filepath", label="Speak to AI")
    out = gr.Audio(type="filepath", label="AI Response")
    mic.change(chat_with_ai, inputs=[mic, engine, lang, tts_eng], outputs=out)
    demo.launch()