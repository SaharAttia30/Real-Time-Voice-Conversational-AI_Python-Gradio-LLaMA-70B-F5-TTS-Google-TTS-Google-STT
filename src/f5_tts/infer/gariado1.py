# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import gc
import json
import re
import tempfile
from collections import OrderedDict
from importlib.resources import files

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False

# Add imports for Nebius API
import os
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = "eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExNzk1NTg1MzgyNzYzMDYxNDQwMCIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwMjgwMzYzNSwidXVpZCI6IjE4NjRlMWM4LTM1NmQtNDY0Ny04NTdlLTJlN2UxYzg2ODJkYSIsIm5hbWUiOiJzYWhhcmY1IiwiZXhwaXJlc19hdCI6IjIwMzAtMDQtMTlUMDQ6MzM6NTUrMDAwMCJ9.632FHIOBhPM3iNlIBBxPC-16uc3FKoiAC1iqVKXUO5g"

# Initialize the Nebius API client
try:
    nebius_client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=os.environ.get("OPENAI_API_KEY")
    )
except Exception as e:
    raise ValueError(f"Failed to initialize Nebius API client. Ensure NEBIUS_API_KEY is set. Error: {str(e)}")

def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
APP_TITLE = "Assist AI – Voice Chatbot"

# Set API key

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

DEFAULT_TTS_MODEL = "F5-TTS_v1"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

# load models

vocoder = load_vocoder()

def load_f5tts():
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)

def load_e2tts():
    ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path)

def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    if model_cfg is None:
        model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)

F5TTS_ema_model = load_f5tts()
E2TTS_ema_model = load_e2tts() if USING_SPACES else None
custom_ema_model, pre_custom_path = None, ""

chat_model_state = None
chat_tokenizer_state = None

@gpu_decorator
def generate_response(messages, model, tokenizer):
    """Generate response using either Hugging Face models or Nebius API"""
    try:
        # Nebius API case
        if isinstance(model, str) and model == "meta-llama/Meta-Llama-3.1-70B-Instruct-fast":
            # Ensure messages are in the correct format
            formatted_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ]
            response = nebius_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
                max_tokens=512,
                temperature=0.6,
                top_p=0.9,
                extra_body={"top_k": 50},
                messages=formatted_messages
            )
            return response.choices[0].message.content

        # Hugging Face model case
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
        )
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    except Exception as e:
        gr.Warning(f"Error generating response: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request. Please try again."

@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return gr.update(), gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    if model == DEFAULT_TTS_MODEL:
        ema_model = F5TTS_ema_model
    elif model == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            show_info("Loading E2-TTS model...")
            E2TTS_ema_model = load_e2tts()
        ema_model = E2TTS_ema_model
    elif isinstance(model, list) and model[0] == "Custom":
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[1]:
            show_info("Loading Custom TTS model...")
            custom_ema_model = load_custom(model[1], vocab_path=model[2], model_cfg=model[3])
            pre_custom_path = model[1]
        ema_model = custom_ema_model

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text

with gr.Blocks() as app_tts:
    gr.Markdown("# Batched TTS")
    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    gen_text_input = gr.Textbox(label="Text to Generate", lines=10)
    generate_btn = gr.Button("Synthesize", variant="primary")
    with gr.Accordion("Advanced Settings", open=False):
        ref_text_input = gr.Textbox(
            label="Reference Text",
            info="Leave blank to automatically transcribe the reference audio. If you enter text it will override automatic transcription.",
            lines=2,
        )
        remove_silence = gr.Checkbox(
            label="Remove Silences",
            info="The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.",
            value=False,
        )
        speed = gr.Slider(
            label="Speed",
            minimum=0.3,
            maximum=2.0,
            value=1.4,
            step=0.1,
            info="Adjust the speed of the audio.",
        )
        nfe_slider = gr.Slider(
            label="NFE Steps",
            minimum=4,
            maximum=64,
            value=45,
            step=2,
            info="Set the number of denoising steps.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Cross-Fade Duration (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.38,
            step=0.01,
            info="Set the duration of the cross-fade between audio clips.",
        )

    audio_output = gr.Audio(label="Synthesized Audio")
    spectrogram_output = gr.Image(label="Spectrogram")

    @gpu_decorator
    def basic_tts(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        remove_silence,
        cross_fade_duration_slider,
        nfe_slider,
        speed,
    ):
        audio_out, spectrogram_path, ref_text_out = infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            tts_model_choice,
            remove_silence,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed ,
        )
        return audio_out, spectrogram_path, ref_text_out

    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            cross_fade_duration_slider,
            nfe_slider,
            speed ,
        ],
        outputs=[audio_output, spectrogram_output, ref_text_input],
    )

def parse_speechtypes_text(gen_text):
    pattern = r"\{(.*?)\}"
    tokens = re.split(pattern, gen_text)
    segments = []
    current_style = "Regular"
    for i in range(len(tokens)):
        if i % 2 == 0:
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            style = tokens[i].strip()
            current_style = style

    return segments

with gr.Blocks() as app_multistyle:
    gr.Markdown(
        """
    # Multiple Speech-Type Generation

    This section allows you to generate multiple speech types or multiple people's voices. Enter your text in the format shown below, and the system will generate speech using the appropriate type. If unspecified, the model will use the regular speech type. The current speech type will be used until the next speech type is specified.
    """
    )

    with gr.Row():
        gr.Markdown(
            """
            **Example Input:**                                                                      
            {Regular} Hello, I'd like to order a sandwich please.                                                         
            {Surprised} What do you mean you're out of bread?                                                                      
            {Sad} I really wanted a sandwich though...                                                              
            {Angry} You know what, darn you and your little shop!                                                                       
            {Whisper} I'll just go back home and cry now.                                                                           
            {Shouting} Why me?!                                                                         
            """
        )

        gr.Markdown(
            """
            **Example Input 2:**                                                                                
            {Speaker1_Happy} Hello, I'd like to order a sandwich please.                                                            
            {Speaker2_Regular} Sorry, we're out of bread.                                                                                
            {Speaker1_Sad} I really wanted a sandwich though...                                                                             
            {Speaker2_Whisper} I'll give you the last one I was hiding.                                                                     
            """
        )

    gr.Markdown(
        "Upload different audio clips for each speech type. The first speech type is mandatory. You can add additional speech types by clicking the 'Add Speech Type' button."
    )

    with gr.Row() as regular_row:
        with gr.Column():
            regular_name = gr.Textbox(value="Regular", label="Speech Type Name")
            regular_insert = gr.Button("Insert Label", variant="secondary")
        regular_audio = gr.Audio(label="Regular Reference Audio", type="filepath")
        regular_ref_text = gr.Textbox(label="Reference Text (Regular)", lines=2)

    # Regular speech type (max 100)
    max_speech_types = 100
    speech_type_rows = [regular_row]
    speech_type_names = [regular_name]
    speech_type_audios = [regular_audio]
    speech_type_ref_texts = [regular_ref_text]
    speech_type_delete_btns = [None]
    speech_type_insert_btns = [regular_insert]

    # Additional speech types (99 more)
    for i in range(max_speech_types - 1):
        with gr.Row(visible=False) as row:
            with gr.Column():
                name_input = gr.Textbox(label="Speech Type Name")
                delete_btn = gr.Button("Delete Type", variant="secondary")
                insert_btn = gr.Button("Insert Label", variant="secondary")
            audio_input = gr.Audio(label="Reference Audio", type="filepath")
            ref_text_input = gr.Textbox(label="Reference Text", lines=2)
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_delete_btns.append(delete_btn)
        speech_type_insert_btns.append(insert_btn)

    # Button to add speech type
    add_speech_type_btn = gr.Button("Add Speech Type")
    speech_type_count = 1
    def add_speech_type_fn():
        row_updates = [gr.update() for _ in range(max_speech_types)]
        global speech_type_count
        if speech_type_count < max_speech_types:
            row_updates[speech_type_count] = gr.update(visible=True)
            speech_type_count += 1
        else:
            gr.Warning("Exhausted maximum number of speech types. Consider restart the app.")
        return row_updates

    add_speech_type_btn.click(add_speech_type_fn, outputs=speech_type_rows)

    # Function to delete a speech type
    def delete_speech_type_fn():
        return gr.update(visible=False), None, None, None

    # Update delete button clicks
    for i in range(1, len(speech_type_delete_btns)):
        speech_type_delete_btns[i].click(
            delete_speech_type_fn,
            outputs=[speech_type_rows[i], speech_type_names[i], speech_type_audios[i], speech_type_ref_texts[i]],
        )

    # Text input for the prompt
    gen_text_input_multistyle = gr.Textbox(
        label="Text to Generate",
        lines=10,
        placeholder="Enter the script with speaker names (or emotion types) at the start of each block, e.g.:\n\n{Regular} Hello, I'd like to order a sandwich please.\n{Surprised} What do you mean you're out of bread?\n{Sad} I really wanted a sandwich though...\n{Angry} You know what, darn you and your little shop!\n{Whisper} I'll just go back home and cry now.\n{Shouting} Why me?!",
    )

    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name):
            current_text = current_text or ""
            speech_type_name = speech_type_name or "None"
            updated_text = current_text + f"{{{speech_type_name}}} "
            return updated_text

        return insert_speech_type_fn

    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i]],
            outputs=gen_text_input_multistyle,
        )

    with gr.Accordion("Advanced Settings", open=False):
        remove_silence_multistyle = gr.Checkbox(
            label="Remove Silences",
            value=True,
        )

    # Generate button
    generate_multistyle_btn = gr.Button("Generate Multi-Style Speech", variant="primary")

    # Output audio
    audio_output_multistyle = gr.Audio(label="Synthesized Audio")

    @gpu_decorator
    def generate_multistyle_speech(
        gen_text,
        *args,
    ):
        speech_type_names_list = args[:max_speech_types]
        speech_type_audios_list = args[max_speech_types : 2 * max_speech_types]
        speech_type_ref_texts_list = args[2 * max_speech_types : 3 * max_speech_types]
        remove_silence = args[3 * max_speech_types]
        # Collect the speech types and their audios into a dict
        speech_types = OrderedDict()

        ref_text_idx = 0
        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}
            else:
                speech_types[f"@{ref_text_idx}@"] = {"audio": "", "ref_text": ""}
            ref_text_idx += 1

        # Parse the gen_text into segments
        segments = parse_speechtypes_text(gen_text)

        # For each segment, generate speech
        generated_audio_segments = []
        current_style = "Regular"

        for segment in segments:
            style = segment["style"]
            text = segment["text"]

            if style in speech_types:
                current_style = style
            else:
                gr.Warning(f"Type {style} is not available, will use Regular as default.")
                current_style = "Regular"

            try:
                ref_audio = speech_types[current_style]["audio"]
            except KeyError:
                gr.Warning(f"Please provide reference audio for type {current_style}.")
                return [None] + [speech_types[style]["ref_text"] for style in speech_types]
            ref_text = speech_types[current_style].get("ref_text", "")

            audio_out, _, ref_text_out = infer(
                ref_audio, ref_text, text, tts_model_choice, remove_silence, 0, show_info=print
            )  # show_info=print no pull to top when generating
            sr, audio_data = audio_out

            generated_audio_segments.append(audio_data)
            speech_types[current_style]["ref_text"] = ref_text_out

        # Concatenate all audio segments
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return [(sr, final_audio_data)] + [speech_types[style]["ref_text"] for style in speech_types]
        else:
            gr.Warning("No audio generated.")
            return [None] + [speech_types[style]["ref_text"] for style in speech_types]

    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            gen_text_input_multistyle,
        ]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [
            remove_silence_multistyle,
        ],
        outputs=[audio_output_multistyle] + speech_type_ref_texts,
    )
    def validate_speech_types(gen_text, regular_name, *args):
        speech_type_names_list = args
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["style"] for segment in segments)

        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            return gr.update(interactive=False)
        else:
            return gr.update(interactive=True)

    gen_text_input_multistyle.change(
        validate_speech_types,
        inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
        outputs=generate_multistyle_btn,
    )
@gpu_decorator
def generate_response(messages, model, tokenizer):
    """Generate response using Nebius API"""
    try:
        formatted_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
        response = nebius_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
            max_tokens=4096,
            temperature=0.6,
            top_p=0.9,
            extra_body={"top_k": 50},
            messages=formatted_messages
        )
        return response.choices[0].message.content
    except Exception as e:
        gr.Warning(f"Error generating response: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request. Please try again."
with gr.Blocks() as app_chat:
    gr.Markdown(
        """
# Voice Chat
Have a conversation with an AI using your reference voice! 
1. Upload a reference audio clip and optionally its transcript.
2. Load the chat model.
3. Record your message through your microphone.
4. The AI will respond using the reference voice.
"""
    )

    chat_model_name_list = [
        "Llama-Fast",  # Add Nebius model
        "Qwen",
        "Phi-4",
    ]

    @gpu_decorator
    def load_chat_model(chat_model_name):
        show_info = gr.Info
        global chat_model_state, chat_tokenizer_state
        if chat_model_state is not None:
            chat_model_state = None
            chat_tokenizer_state = None
            gc.collect()
            torch.cuda.empty_cache()

        if chat_model_name == "Llama-Fast":
            show_info(f"Using Nebius API chat model: {chat_model_name}")
            chat_model_state = "meta-llama/Meta-Llama-3.1-70B-Instruct-fast"
            chat_tokenizer_state = None
            show_info(f"Chat model {chat_model_name} ready via Nebius API!")
        else:
            show_info(f"Unsupported chat model: {chat_model_name}")
            return gr.update(visible=True), gr.update(visible=False)

        return gr.update(visible=False), gr.update(visible=True)
    if USING_SPACES:
        load_chat_model(chat_model_name_list[0])
    
    chat_model_name_input = gr.Dropdown(
        choices=chat_model_name_list,
        value=chat_model_name_list[0],
        label="Chat Model Name",
        info="Enter the name of a chat model",
        allow_custom_value=not USING_SPACES,
    )
    load_chat_model_btn = gr.Button("Load Chat Model", variant="primary", visible=not USING_SPACES)
    chat_interface_container = gr.Column(visible=USING_SPACES)

    chat_model_name_input.change(
        lambda: gr.update(visible=True),
        None,
        load_chat_model_btn,
        show_progress="hidden",
    )
    load_chat_model_btn.click(
        load_chat_model, inputs=[chat_model_name_input], outputs=[load_chat_model_btn, chat_interface_container]
    )

    with chat_interface_container:
        with gr.Row():
            with gr.Column():
                ref_audio_chat = gr.Audio(label="Reference Audio", type="filepath")
            with gr.Column():
                with gr.Accordion("Advanced Settings", open=False):
                    remove_silence_chat = gr.Checkbox(
                        label="Remove Silences",
                        value=True,
                    )
                    ref_text_chat = gr.Textbox(
                        label="Reference Text",
                        info="Optional: Leave blank to auto-transcribe",
                        lines=2,
                    )
                    system_prompt_chat = gr.Textbox(
                        label="System Prompt",
                        value=system_prompt,
                        lines=10,
                    )
            with gr.Accordion("Advanced TTS Settings", open=False):
                cross_fade_duration_slider = gr.Slider(
                    label="Cross-Fade Duration (s)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.56,
                    step=0.01,
                    info="Duration of cross-fade between audio clips."
                )
                nfe_slider = gr.Slider(
                    label="NFE Steps",
                    minimum=4,
                    maximum=64,
                    value=58,
                    step=2,
                    info="Number of denoising steps for audio generation."
                )
                speed_slider = gr.Slider(
                    label="Speed video",
                    minimum=0.3,
                    maximum=2.0,
                    value=1.4,
                    step=0.1,
                    info="Playback speed of the generated audio."
                )
               
        chatbot_interface = gr.Chatbot(label="Conversation")
        with gr.Row():
            with gr.Column():
                audio_input_chat = gr.Microphone(
                    label="Speak your message",
                    type="filepath",
                )
                audio_output_chat = gr.Audio(autoplay=True)
            with gr.Column():
                text_input_chat = gr.Textbox(
                    label="Type your message",
                    lines=1,
                )
                send_btn_chat = gr.Button("Send Message")
                clear_btn_chat = gr.Button("Clear Conversation")

        conversation_state = gr.State(
            value=[
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
        )

        @gpu_decorator
        def process_audio_input(audio_path, text, history, conv_state,
                                model_choice, voice_sample,
                                crossfade, nfe, speed : gr.Slider, remove_sil):
            if not audio_path and not text.strip():
                return history, conv_state, ""

            if audio_path:
                text = preprocess_ref_audio_text(audio_path, text)[1]

            if not text.strip():
                return history, conv_state, ""
            conv_state.append({"role": "user", "content": text})
            history.append((text, None))
            response = generate_response(conv_state, model_choice, None)
            conv_state.append({"role": "assistant", "content": response})
            history[-1] = (text, response)
            tts_out, _, _ = infer(
                voice_sample,
                "",
                response,
                tts_model_choice,
                remove_sil,
                cross_fade_duration=crossfade,
                nfe_step=nfe,
                speed=speed,
            )
            return history, conv_state
        @gpu_decorator
        def generate_audio_response(history, ref_audio, ref_text, remove_silence, cross_fade_duration, nfe, speed):
            if not history or not ref_audio:
                return None, None
            last_user_message, last_ai_response = history[-1]
            if not last_ai_response:
                return None, None
            audio_result, _, ref_text_out = infer(
                ref_audio,
                ref_text,
                last_ai_response,
                tts_model_choice,
                remove_silence,
                cross_fade_duration=cross_fade_duration,  # Use the parameter value
                nfe_step=nfe,                            # Use the parameter value
                speed=speed,                             # Use the parameter value
                show_info=print,
            )
            return audio_result, ref_text_out

        def clear_conversation():
            """Reset the conversation"""
            return [], [
                {
                    "role": "system",
                    "content":system_prompt
                }
            ]

        def update_system_prompt(new_prompt):
            """Update the system prompt and reset the conversation"""
            new_conv_state = [{"role": "system", "content": new_prompt}]
            return [], new_conv_state

        audio_input_chat.stop_recording(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[
                chatbot_interface,           # history
                ref_audio_chat,             # ref_audio
                ref_text_chat,              # ref_text
                remove_silence_chat,        # remove_silence
                cross_fade_duration_slider, # Slider component (Gradio passes its value)
                nfe_slider,                # Slider component (Gradio passes its value)
                speed_slider,              # Slider component (Gradio passes its value)
            ],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            audio_input_chat,
        )

        text_input_chat.submit(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[
                chatbot_interface,
                ref_audio_chat,
                ref_text_chat,
                remove_silence_chat,
                cross_fade_duration_slider,
                nfe_slider,
                speed_slider,
            ],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )

        send_btn_chat.click(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[
                chatbot_interface,
                ref_audio_chat,
                ref_text_chat,
                remove_silence_chat,
                cross_fade_duration_slider,
                nfe_slider,
                speed_slider,
            ],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )
        clear_btn_chat.click(
            clear_conversation,
            outputs=[chatbot_interface, conversation_state],
        )

        # Handle system prompt change and reset conversation
        system_prompt_chat.change(
            update_system_prompt,
            inputs=system_prompt_chat,
            outputs=[chatbot_interface, conversation_state],
        )
      
# custom_css = """
# body {
#     font-family: 'Arial', sans-serif;
#     background: linear-gradient(135deg, #1e3c72, #2a5298);
#     color: #fff;
# }
# .gradio-container {
#     max-width: 1000px;
#     margin: 40px auto;
#     padding: 30px;
#     background: rgba(255, 255, 255, 0.15);
#     border-radius: 20px;
#     box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
# }
# h1 {
#     text-align: center;
#     font-size: 2.8em;
#     margin-bottom: 10px;
#     color: #f0f0f0;
# }
# h3 {
#     text-align: center;
#     font-size: 1.3em;
#     margin-bottom: 30px;
#     color: #d0d0d0;
# }
# input, textarea, select, .gr-button, .gr-checkbox {
  
#     padding: 10px !important;
# }
# .gr-button {
#     background: linear-gradient(90deg, #ff6f61, #ff9f80) !important;
#     transition: all 0.3s !important;
#     font-weight: bold;
# }
# .gr-button:hover {
#     transform: scale(1.05) !important;
# }
# .gr-tab-button {
#     background: rgba(255, 255, 255, 0.1) !important;
#     color: #fff !important;
#     border: 1px solid #ccc !important;
#     border-radius: 8px !important;
#     padding: 10px 20px !important;
# }
# .gr-tab-button:hover {
#     background: rgba(255, 255, 255, 0.2) !important;
# }
# .gr-tab-button[aria-selected="true"] {
#     background: linear-gradient(90deg, #ff6f61, #ff9f80) !important;
# }
# .gr-audio, .gr-image {
#     border-radius: 8px !important;
# }
# .gr-row {
#     margin-bottom: 20px !important;
# }
# """

with gr.Blocks(title=APP_TITLE) as app:
    gr.Markdown(
        """
        # ASSIST AI: Advanced AI Phone Service 
        **Features**:  
        - Customize speech speed, style, and emotion.  
        - Batch process multiple speech types or speakers.  
        - Engage in voice chats with AI using your reference voice.  

        **Note**: For best results, use reference audio clips shorter than 12 seconds in WAV or MP3 format. Reference text will be auto-transcribed if not provided.
        """
    )

    last_used_custom = files("f5_tts").joinpath("infer/.cache/last_used_custom_model_info_v1.txt")

    def load_last_used_custom():
        try:
            custom = []
            with open(last_used_custom, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            return custom
        except FileNotFoundError:
            last_used_custom.parent.mkdir(parents=True, exist_ok=True)
            return DEFAULT_TTS_MODEL_CFG

    def switch_tts_model(new_choice):
        global tts_model_choice
        if new_choice == "Custom":
            custom_ckpt_path, custom_vocab_path, custom_model_cfg = load_last_used_custom()
            tts_model_choice = ["Custom", custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg)]
            return (
                gr.update(visible=True, value=custom_ckpt_path),
                gr.update(visible=True, value=custom_vocab_path),
                gr.update(visible=True, value=custom_model_cfg),
            )
        else:
            tts_model_choice = new_choice
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def set_custom_model(custom_ckpt_path, custom_vocab_path, custom_model_cfg):
        global tts_model_choice
        tts_model_choice = ["Custom", custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg)]
        with open(last_used_custom, "w", encoding="utf-8") as f:
            f.write(custom_ckpt_path + "\n" + custom_vocab_path + "\n" + custom_model_cfg + "\n")

    with gr.Row():
        if not USING_SPACES:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS", "Custom"],
                label="Choose TTS Model",
                value=DEFAULT_TTS_MODEL
            )
        else:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS"],
                label="Choose TTS Model",
                value=DEFAULT_TTS_MODEL
            )
        custom_ckpt_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[0]],
            value=load_last_used_custom()[0],
            allow_custom_value=True,
            label="Model: local_path | hf://user_id/repo_id/model_ckpt",
            visible=False,
        )
        custom_vocab_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[1]],
            value=load_last_used_custom()[1],
            allow_custom_value=True,
            label="Vocab: local_path | hf://user_id/repo_id/vocab_file",
            visible=False,
        )
        
        custom_model_cfg = gr.Dropdown(
            choices=[
                DEFAULT_TTS_MODEL_CFG[2],
                json.dumps(
                    dict(
                        dim=1024,
                        depth=22,
                        heads=16,
                        ff_mult=2,
                        text_dim=512,
                        text_mask_padding=False,
                        conv_layers=4,
                        pe_attn_head=1,
                    )
                ),
                json.dumps(
                    dict(
                        dim=768,
                        depth=18,
                        heads=12,
                        ff_mult=2,
                        text_dim=512,
                        text_mask_padding=False,
                        conv_layers=4,
                        pe_attn_head=1,
                    )
                ),
            ],
            value=load_last_used_custom()[2],
            allow_custom_value=True,
            label="Config: in a dictionary form",
            visible=False,
        )
    
    choose_tts_model.change(
        switch_tts_model,
        inputs=[choose_tts_model],
        outputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_ckpt_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_vocab_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_model_cfg.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
  
    gr.TabbedInterface(
        [app_tts, app_multistyle, app_chat],
        ["Basic-TTS", "Multi-Speech", "Voice-Chat"],
    )

@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via assist AI share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='het',
)
@click.option(
    "--inbrowser",
    "-i",
    is_flag=True,
    default=False,
    help="Automatically launch the interface in the default web browser",
)
def main(port, host, share, api, root_path, inbrowser):
    global app
    print("Starting app...")
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_api=api,
        root_path=root_path,
        inbrowser=inbrowser,
    )

if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()