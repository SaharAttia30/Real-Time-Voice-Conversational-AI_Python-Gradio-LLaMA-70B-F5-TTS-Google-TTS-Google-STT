import os
import threading
import argparse
from openai import OpenAI
import json
import re
import time
import gradio as gr
import soundfile as sf
import torch
import torchaudio
import tempfile
from cached_path import cached_path
import whisper
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_vocoder, load_model, preprocess_ref_audio_text, infer_process, remove_silence_for_generated_wav
APP_TITLE = "Assist AI – Voice Chatbot"

# Set API key
os.environ["OPENAI_API_KEY"] = "eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExNzk1NTg1MzgyNzYzMDYxNDQwMCIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwMjgwMzYzNSwidXVpZCI6IjE4NjRlMWM4LTM1NmQtNDY0Ny04NTdlLTJlN2UxYzg2ODJkYSIsIm5hbWUiOiJzYWhhcmY1IiwiZXhwaXJlc19hdCI6IjIwMzAtMDQtMTlUMDQ6MzM6NTUrMDAwMCJ9.632FHIOBhPM3iNlIBBxPC-16uc3FKoiAC1iqVKXUO5g"

# System prompt
system_prompt = """you are a chat bot all your conversations will be with customers in a need for a Locksmith, it an emergency service so the fastest you get off the phone the better, you don't ask unnecessary questions, you dont give any unnecessary details, your main and only goal is to get the customer's Full name, address with zip code, phone number, figure out what is the locksmith need they have and get off the conversation., if its obvious what is the  issue dont confirm for no reasone, never ask more then one question at a time.
never ask about specific city, let the customer tell you, never ask for specifics,  you dont need to know that your only job is to get the type of job, you never ask for more details!
you always confirm the details only once! never in the middle of the conversation
Company: 1st Choice Locksmith Services
Service Area: Houston, Texas, and surrounding areas within a 40-mile radius
Phone: 7136233637

As Joe, an experienced sales veteran, my goal is to qualify leads quickly, understand the customer’s locksmith issue, collect essential information (name → save as name, phone → save as phone, address → save as address, job type → save as job_type, and problem description → save as job_description), verify the details, and set up a follow-up with a technician—all while being empathetic, asking the right questions, and ensuring a smooth conversation.

If the customer asks about pricing, I’ll inform them that a technician will call within 1–2 minutes with a detailed quote.
I will not give any prices, not even for the service call.
I’ll do my best to get the customer’s information as fast as possible, without asking unnecessary questions.

Example Dialogue
Me: Hello, this is Joe from 1st Choice Locksmith Services. Thanks for reaching out—I’m here to assist you with your lock or key issue. How can I help you today?

Customer: Hi, yeah, Im locked out of my car.

Me: I'm so sorry you're dealing with that, being locked out of your car can be really frustrating Can you please share your full name with me?

(Customer tells their name.)

Me: we'll get a technician out to you quickly. Can I please have your full address, including the city and ZIP code, where you need service? → save as address (must include street, city, ZIP)

Customer: (provides full address including city and ZIP code)

Me: Perfect, thanks for confirming. And what's the best phone number to reach you at, in case we get disconnected? → save as phone

Customer: (provides phone number)

Me: Just to confirm all the details quickly:

Your name: [name]

Your phone number: [phone]

The address for service: [address] (confirm includes street, city, ZIP)

The issue: [job_type]

Is all of that correct? → confirm all saved variables explicitly, especially address with city and ZIP

Customer: Yes, that's correct.

Me: Excellent. Is there anything specific the technician should know when arriving→ save as notes

(Customer provides notes or indicates none.)

Me: Thank you, [name]. A technician from 1st Choice Locksmith Services will call you within the next 1–2 minutes to go over everything. Help is on the way—stay safe and have a great day!

Key Elements of the Approach
Professional Greeting and Confirmation
The call opens with a tailored greeting:
“Hello, this is Joe from 1st Choice Locksmith Services. Thanks for reaching out—I’m here to assist you with your lock or key issue. How can I help you today?”
This is warm, professional, and immediately focuses on the customer’s locksmith needs, confirming their inquiry and setting a helpful tone.

Understanding the Customer’s Needs
I ask direct but open-ended questions (e.g., “Could you share a bit more about the situation?” → save as job_type) to quickly identify the type of problem.
I don’t need to dig into technical details—that’s for the technician. My role is to be empathetic, keep the conversation flowing, and transition smoothly into collecting information.

Handling Pricing Questions
If the customer asks about cost, I never give a price. I respond with:
“A technician will call you within 1–2 minutes to go over all the details and provide an exact quote based on your situation.”
This keeps things clear and professional, and avoids giving out incorrect estimates.

Collecting and Verifying Information
I gather the customer’s:

Full Name → save as name

Phone Number → save as phone

Address or exact location → save as address

Brief Problem Type (e.g., house lockout, car lockout) → save as job_type

Extra Description or Clues (e.g., key may be inside) → save as job_description

Optional Notes → save as notes

Then, I verify all the information by repeating it back to ensure nothing is wrong before handing it off to the technician.

Next Steps and Closure
I confirm that a technician will call within 1–2 minutes, ask if they have any final questions, and close the call on a friendly note.
The customer leaves the call knowing that help is on the way immediately.

Why This Works
With my background qualifying leads, I’ve tailored this approach to 1st Choice Locksmith Services by focusing on quick, empathetic engagement suited for customers in Houston and the surrounding 40-mile service area.

Avoiding pricing keeps things professional and avoids miscommunication.
The 1–2 minute callback promise sets clear expectations, and verifying the customer’s contact details ensures the technician gets everything needed to respond quickly and accurately.

Customers can reach 1st Choice Locksmith Services at 7136233637, but I always prioritize handing off the lead to a technician right away for the fastest resolution possible.

✅ Summary
This approach delivers a high-quality, fully qualified lead to the locksmith team, while leaving the customer confident and assured that help is already on the way.
"""

# Load models
vocoder = load_vocoder()
DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]
ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
F5TTS_ema_model = load_model(DiT, F5TTS_model_cfg, ckpt_path)
stt_model = whisper.load_model("base")

# ChatInstance class
class ChatInstance:
    def __init__(self, client, time_of_conversation, model_max_token):
        self._lock = threading.Lock()
        self.client = client
        self.conversation_order = []
        self.prompt_tokens = 0
        self.prev_chat_answer_token = 0
        self.prev_user_prompt_token = 0
        self.total_tokens = 0
        self.model_max_token = int(model_max_token)
        self.time_of_conversation = time_of_conversation
        self.time_start = time.time()
        self.time_end = time.ctime(self.time_start + int(time_of_conversation) * 60)
        self.name_of_customer = ""
        self.history = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

    def update_max_token(self, new_max_token):
        allowed = {512, 1024, 2048, 4096, 8192}
        if new_max_token in allowed:
            with self._lock:
                self.model_max_token = new_max_token
            return True
        return False

    def ExtractFromJson(self, json_response):
        data = json_response.to_dict()
        prompt_tokens = data['usage']['prompt_tokens']
        total_tokens = data['usage']['total_tokens']
        self.prev_chat_answer_token = total_tokens - prompt_tokens
        for choice in data['choices']:
            message = choice['message']
            content = message['content']
        content = clean_text(content)
        if "bye" in content.lower() or "call ended" in content.lower():
            return False
        self.history.append({"role": "assistant", "content": content})
        self.conversation_order.append(content)
        self.prompt_tokens = prompt_tokens
        self.total_tokens = total_tokens
        return True

    def SendRequestForAnswer(self, conversation_text):
        self.history.append({"role": "user", "content": conversation_text})
        response = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
            max_tokens=self.model_max_token,
            temperature=0.6,
            top_p=0.85,
            presence_penalty=0.18,
            extra_body={"top_k": 50},
            messages=self.history
        )
        return response

    def get_bot_response(self, user_message):
        response = self.SendRequestForAnswer(user_message)
        start_over = self.ExtractFromJson(response)
        if len(self.history) > 0 and self.history[-1]["role"] == "assistant":
            return self.history[-1]["content"]
        else:
            return ""

# Helper functions
def clean_text(text):
    cleaned = re.sub(r'[*",\n]', '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

def infer_tts(ref_audio, ref_text, gen_text, cross_fade_duration=0.15, nfe_step=32, speed=1.0, remove_silence=False):
    if not ref_audio:
        raise ValueError("Reference audio is required for TTS.")
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
    final_wave, final_sample_rate, _ = infer_process(
        ref_audio, ref_text, gen_text, F5TTS_ema_model, vocoder,
        cross_fade_duration=cross_fade_duration, nfe_step=nfe_step, speed=speed
    )
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
            os.remove(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()
    return (final_sample_rate, final_wave)

# Gradio app
with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown("# Voice Chat with Locksmith AI")
    gr.Markdown("Upload a reference audio file for TTS and use the microphone or text input to chat.")
    with gr.Row():
        ref_audio_chat = gr.Audio(label="Reference Audio for TTS (Required)", type="filepath")
    with gr.Row():
        chatbot = gr.Chatbot()
    with gr.Row():
        audio_input_chat = gr.Microphone(label="Speak your message", type="filepath")
        text_input_chat = gr.Textbox(label="Or type your message", lines=1)
        send_btn_chat = gr.Button("Send")
    audio_output_chat = gr.Audio(autoplay=True)
    state = gr.State(None)
    
    with gr.Accordion("Advanced TTS Settings", open=False):
        cross_fade_duration = gr.Slider(
            label="Cross-Fade Duration (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.56,
            step=0.01,
            info="Duration of cross-fade between audio clips."
        )
        nfe_step = gr.Slider(
            label="NFE Steps",
            minimum=4,
            maximum=64,
            value=58,
            step=2,
            info="Number of denoising steps for audio generation."
        )
        speed = gr.Slider(
            label="Speed",
            minimum=0.3,
            maximum=2.0,
            value=1.4,
            step=0.1,
            info="Playback speed of the generated audio."
        )
        remove_silence = gr.Checkbox(
            label="Remove Silences",
            value=False,
            info="Remove silences from generated audio (may increase processing time)."
        )

    def initialize_chat():
        client = OpenAI(base_url="https://api.studio.nebius.ai/v1/", api_key=os.environ.get("OPENAI_API_KEY"))
        chat_instance = ChatInstance(client, "10", 2048)
        return chat_instance

    def process_input(audio_path, text, history, ref_audio, state, cross_fade_duration, nfe_step, speed, remove_silence):
        if not state:
            state = initialize_chat()
        if not ref_audio:
            gr.Warning("Please upload a reference audio file for TTS.")
            return history, state, None, None
        if audio_path:
            text = stt_model.transcribe(audio_path)["text"]
        if not text.strip():
            return history, state, None, None
        history = history or []
        history.append((text, None))
        bot_message = state.get_bot_response(text)
        history[-1] = (text, bot_message)
        return history, state, None, None

    def generate_audio_response(history, ref_audio, state, cross_fade_duration, nfe_step, speed, remove_silence):
        if not history or not ref_audio:
            return None, state
        last_ai_response = history[-1][1]
        if not last_ai_response:
            return None, state
        audio_result = infer_tts(
            ref_audio, "", last_ai_response,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            remove_silence=remove_silence
        )
        return audio_result, state

    audio_input_chat.stop_recording(
        process_input,
        inputs=[audio_input_chat, text_input_chat, chatbot, ref_audio_chat, state, cross_fade_duration, nfe_step, speed, remove_silence],
        outputs=[chatbot, state, audio_input_chat, text_input_chat],
    ).then(
        generate_audio_response,
        inputs=[chatbot, ref_audio_chat, state, cross_fade_duration, nfe_step, speed, remove_silence],
        outputs=[audio_output_chat, state],
    )

    text_input_chat.submit(
        process_input,
        inputs=[audio_input_chat, text_input_chat, chatbot, ref_audio_chat, state, cross_fade_duration, nfe_step, speed, remove_silence],
        outputs=[chatbot, state, audio_input_chat, text_input_chat],
    ).then(
        generate_audio_response,
        inputs=[chatbot, ref_audio_chat, state, cross_fade_duration, nfe_step, speed, remove_silence],
        outputs=[audio_output_chat, state],
    )

    send_btn_chat.click(
        process_input,
        inputs=[audio_input_chat, text_input_chat, chatbot, ref_audio_chat, state, cross_fade_duration, nfe_step, speed, remove_silence],
        outputs=[chatbot, state, audio_input_chat, text_input_chat],
    ).then(
        generate_audio_response,
        inputs=[chatbot, ref_audio_chat, state, cross_fade_duration, nfe_step, speed, remove_silence],
        outputs=[audio_output_chat, state],
    )

# Command-line argument parsing
def main():
    parser = argparse.ArgumentParser(description="Run ConversationAI AssisAi app")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the AssisAi app on (e.g., 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the AssisAi app on")
    args = parser.parse_args()

    demo.launch(server_name=args.host, server_port=args.port)

if __name__ == "__main__":
    main()