import gradio as gr

def chat_interface(audio, text, engine, lang, voice):
    return "response.mp3", "Chat response"

iface = gr.Interface(
    fn=chat_interface,
    inputs=["audio", "text", "dropdown", "dropdown", "dropdown"],
    outputs=["audio", "text"]
)

iface.launch()