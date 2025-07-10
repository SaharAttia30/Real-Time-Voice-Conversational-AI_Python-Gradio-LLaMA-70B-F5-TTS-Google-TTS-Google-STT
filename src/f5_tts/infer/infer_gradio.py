# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import gc
import json
import re
import tempfile
from collections import OrderedDict
from importlib.resources import files
import requests # For TTS call
import base64
import traceback
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
os.environ["GOOGLE_API_KEY"] = "AIzaSyC2MC-S458QlwKKa1EFyZYH-HJJkAngSRw"
# --- Nebius API Key Handling ---
# Prefer environment variable for security
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
if not NEBIUS_API_KEY:
    # Fallback or error - replace with your key if needed, but env var is better
    print("Warning: NEBIUS_API_KEY environment variable not set. Using placeholder.")
    NEBIUS_API_KEY = "YOUR_NEBIUS_API_KEY_HERE" # Replace or ensure env var is set

# --- Google TTS API Key Handling ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("\n!!! WARNING: GOOGLE_API_KEY environment variable not set. !!!")
    print("!!! Google TTS in the Voice Chat tab will NOT work without it. !!!")
    print("!!! Set the GOOGLE_API_KEY environment variable before running. !!!\n")
    # Set a placeholder to avoid immediate crash, but function will fail
    GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"

try:
    nebius_client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=NEBIUS_API_KEY
    )
except Exception as e:
    # Don't raise ValueError immediately, allow app to load, but log error
    print(f"ERROR: Failed to initialize Nebius API client. Ensure NEBIUS_API_KEY is set correctly. Error: {str(e)}")
    nebius_client = None # Indicate client is not functional


def gpu_decorator(func):
    if USING_SPACES:
        # Assuming spaces.GPU might require specific setup not available here.
        # If spaces.GPU is critical, ensure its environment is correctly configured.
        # For now, return the function directly if spaces is available but GPU might not be configured.
        # return spaces.GPU(func) # Original line - uncomment if spaces.GPU setup is confirmed
        print("Note: Running in Spaces environment, GPU decorator bypassed in this example.")
        return func
    else:
        return func

# --- Import F5-TTS components ---
# Ensure f5_tts library is installed and accessible
try:
    from f5_tts.model import DiT, UNetT
    from f5_tts.infer.utils_infer import (
        load_vocoder,
        load_model,
        preprocess_ref_audio_text,
        infer_process,
        remove_silence_for_generated_wav,
        save_spectrogram,
    )
    F5_TTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import f5_tts components ({e}). Basic TTS and Multi-Speech tabs might not work.")
    F5_TTS_AVAILABLE = False
    # Define dummy functions or classes if needed to prevent NameErrors later
    class DiT: pass
    class UNetT: pass
    def load_vocoder(): return None
    def load_model(a, b, c, **kwargs): return None
    def preprocess_ref_audio_text(audio, text, **kwargs): return audio, text # Basic pass-through
    def infer_process(*args, **kwargs): return np.zeros(1), 16000, None # Dummy output
    def remove_silence_for_generated_wav(path): pass
    def save_spectrogram(spec, path): pass


APP_TITLE = "Assist AI – Voice Chatbot"

# System prompt (keep as is)
system_prompt = """you are a chat bot all your conversations will be with customers in a need for a Locksmith, it an emergency service so the fastest you get off the phone the better, you don't ask unnecessary questions, you dont give any unnecessary details, your main and only goal is to get the customer's Full name, address with zip code, phone number, figure out what is the locksmith need they have and get off the conversation., if its obvious what is the  issue dont confirm for no reasone, never ask more then one question at a time.
never ask about specific city, let the customer tell you, never ask for specifics,  you dont need to know that your only job is to get the type of job, you never ask for more details!
you always confirm the details only once! never in the middle of the conversation
Company: 1st Choice Locksmith Services
Service Area: Houston, Texas, and surrounding areas within a 40-mile radius
Phone: 7136233637

As Joe, an experienced sales veteran, my goal is to qualify leads quickly, understand the customer’s locksmith issue, collect essential information (name -> save as name, phone -> save as phone, address -> save as address, job type -> save as job_type, and problem description -> save as job_description), verify the details, and set up a follow-up with a technician—all while being empathetic, asking the right questions, and ensuring a smooth conversation.

If the customer asks about pricing, I’ll inform them that a technician will call within 1–2 minutes with a detailed quote.
I will not give any prices, not even for the service call.
I’ll do my best to get the customer’s information as fast as possible, without asking unnecessary questions.

Example Dialogue
Me: Hello, this is Joe from 1st Choice Locksmith Services. Thanks for reaching out—I’m here to assist you with your lock or key issue. How can I help you today?

Customer: Hi, yeah, Im locked out of my car.

Me: I'm so sorry you're dealing with that, being locked out of your car can be really frustrating Can you please share your full name with me?

(Customer tells their name.)

Me: we'll get a technician out to you quickly. Can I please have your full address, including the city and ZIP code, where you need service? -> save as address (must include street, city, ZIP)

Customer: (provides full address including city and ZIP code)

Me: Perfect, thanks for confirming. And what's the best phone number to reach you at, in case we get disconnected? -> save as phone

Customer: (provides phone number)

Me: Just to confirm all the details quickly:

Your name: [name]

Your phone number: [phone]

The address for service: [address] (confirm includes street, city, ZIP)

The issue: [job_type]

Is all of that correct? -> confirm all saved variables explicitly, especially address with city and ZIP

Customer: Yes, that's correct.

Me: Excellent. Is there anything specific the technician should know when arriving-> save as notes

(Customer provides notes or indicates none.)

Me: Thank you, [name]. A technician from 1st Choice Locksmith Services will call you within the next 1–2 minutes to go over everything. Help is on the way—stay safe and have a great day!

Key Elements of the Approach
Professional Greeting and Confirmation
The call opens with a tailored greeting:
“Hello, this is Joe from 1st Choice Locksmith Services. Thanks for reaching out—I’m here to assist you with your lock or key issue. How can I help you today?”
This is warm, professional, and immediately focuses on the customer’s locksmith needs, confirming their inquiry and setting a helpful tone.

Understanding the Customer’s Needs
I ask direct but open-ended questions (e.g., “Could you share a bit more about the situation?” -> save as job_type) to quickly identify the type of problem.
I don’t need to dig into technical details—that’s for the technician. My role is to be empathetic, keep the conversation flowing, and transition smoothly into collecting information.

Handling Pricing Questions
If the customer asks about cost, I never give a price. I respond with:
“A technician will call you within 1–2 minutes to go over all the details and provide an exact quote based on your situation.”
This keeps things clear and professional, and avoids giving out incorrect estimates.

Collecting and Verifying Information
I gather the customer’s:

Full Name -> save as name

Phone Number -> save as phone

Address or exact location -> save as address

Brief Problem Type (e.g., house lockout, car lockout) -> save as job_type

Extra Description or Clues (e.g., key may be inside) -> save as job_description

Optional Notes -> save as notes

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
tts_model_choice = DEFAULT_TTS_MODEL # Default for Basic/Multi-speech tabs

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

# --- Load models (Conditional on F5_TTS_AVAILABLE) ---
if F5_TTS_AVAILABLE:
    vocoder = load_vocoder()

    def load_f5tts():
        try:
            ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
            F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
            return load_model(DiT, F5TTS_model_cfg, ckpt_path)
        except Exception as e:
            print(f"Error loading F5-TTS model: {e}")
            return None

    def load_e2tts():
         try:
            ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
            E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)
            return load_model(UNetT, E2TTS_model_cfg, ckpt_path)
         except Exception as e:
            print(f"Error loading E2-TTS model: {e}")
            return None

    def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
        try:
            ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
            if ckpt_path.startswith("hf://"):
                ckpt_path = str(cached_path(ckpt_path))
            if vocab_path.startswith("hf://"):
                vocab_path = str(cached_path(vocab_path))
            if model_cfg is None:
                model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
            # Ensure model_cfg is a dict before passing
            if isinstance(model_cfg, str):
                 model_cfg = json.loads(model_cfg)
            return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)
        except Exception as e:
            print(f"Error loading custom TTS model: {e}")
            return None

    F5TTS_ema_model = load_f5tts()
    E2TTS_ema_model = load_e2tts() if USING_SPACES else None # Potentially load lazily later
    custom_ema_model, pre_custom_path = None, ""

else:
    # Define placeholders if F5 TTS is not available
    vocoder = None
    F5TTS_ema_model = None
    E2TTS_ema_model = None
    custom_ema_model = None
    pre_custom_path = ""
    print("F5-TTS components not loaded. Basic TTS and Multi-Speech tabs will be affected.")


# --- Chat Model State ---
chat_model_state = None
chat_tokenizer_state = None # Not used for Nebius API

# --- Google TTS Function (Keep as is) ---
def google_tts_via_api_key(text, api_key, language_code="en-US", voice_name="en-US-Wavenet-D", speaking_rate=1.0, volume_gain_db=0.0):
    """Generates TTS audio using Google Cloud TTS API and saves to a unique temp MP3 file."""
    if not text:
        print("TTS Warning: Received empty text.")
        return None
    if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
        print("TTS Error: Google API Key is missing or is a placeholder. Cannot generate audio.")
        gr.Warning("Google API Key is not configured. Cannot generate AI voice.")
        return None

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": language_code,
            "name": voice_name,
            # "ssmlGender": "NEUTRAL" # Often inferred from voice name
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": speaking_rate,
            "volumeGainDb": volume_gain_db
        }
    }
    try:
        print(f"Requesting Google TTS for text: '{text[:50]}...' (Rate: {speaking_rate})")
        resp = requests.post(url, json=payload, timeout=25) # Increased timeout
        resp.raise_for_status() # Raise exception for bad status codes (4xx, 5xx)

        response_json = resp.json()
        audio_content_base64 = response_json.get("audioContent")

        if not audio_content_base64:
            print(f"TTS Error: No audioContent received. Response: {response_json}")
            gr.Warning("Failed to get audio from Google TTS.")
            return None

        audio_bytes = base64.b64decode(audio_content_base64)

        # Save to a unique temporary file with .mp3 extension
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            output_path = tmp_file.name
            tmp_file.write(audio_bytes)

        print(f"Wrote Google TTS audio ({len(audio_bytes)} bytes) to {output_path}")
        return output_path # Return the path to the MP3 file

    except requests.exceptions.RequestException as e:
        print(f"Google TTS API Request Error: {e}")
        # Check for specific API key errors if possible
        if resp is not None and resp.status_code == 403:
             print("Google TTS Error: Received 403 Forbidden. Check your API key permissions and billing status.")
             gr.Error("Google TTS Error: Access denied. Check API key.")
        elif resp is not None and resp.status_code == 400:
             print(f"Google TTS Error: Received 400 Bad Request. Check payload/parameters. Response: {resp.text}")
             gr.Warning("Google TTS Error: Invalid request.")
        else:
             gr.Warning(f"Google TTS request failed: {e}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Google TTS Processing Error: {e}")
        gr.Warning(f"Error during Google TTS processing: {e}")
        traceback.print_exc()
        return None

# --- Nebius LLM Response Generation ---
# Simplified generate_response specifically for Nebius API
def generate_nebius_response(messages):
    """Generate response using Nebius API"""
    if not nebius_client:
         gr.Error("Nebius API client is not initialized. Cannot generate response.")
         return "I'm sorry, the chat service is currently unavailable."
    try:
        formatted_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
        print(f"Sending {len(formatted_messages)} messages to Nebius API.")
        response = nebius_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
            max_tokens=512, # Reduced from 4096, adjust if needed
            temperature=0.6,
            top_p=0.9,
            # Nebius specific parameters might go in extra_body if needed
            # extra_body={"top_k": 50}, # Keep if required by Nebius
            messages=formatted_messages
        )
        print("Received response from Nebius API.")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating Nebius response: {str(e)}")
        traceback.print_exc()
        # Provide more specific feedback if possible
        if "authentication" in str(e).lower():
             gr.Warning("Nebius API authentication failed. Check your API key.")
             return "I'm sorry, there was an issue connecting to the chat service (Auth Error)."
        else:
             gr.Warning(f"Error generating response: {str(e)}")
             return "I'm sorry, I encountered an error while processing your request. Please try again."

# --- F5-TTS infer function (Keep for Basic/Multi-speech tabs) ---
@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model_name, # Changed from 'model' which was ambiguous
    remove_silence,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
):
    if not F5_TTS_AVAILABLE:
        show_info("F5-TTS components are not available. Cannot perform inference.")
        # Return dummy data matching expected output types
        return (16000, np.zeros(1)), None, ref_text # (sample_rate, audio_array), spectrogram_path, ref_text

    if not ref_audio_orig:
        gr.Warning("Please provide reference audio for F5-TTS.")
        return gr.update(), gr.update(), ref_text

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return gr.update(), gr.update(), ref_text

    # Select the F5-TTS model based on name
    if model_name == DEFAULT_TTS_MODEL:
        ema_model = F5TTS_ema_model
    elif model_name == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            show_info("Loading E2-TTS model...")
            E2TTS_ema_model = load_e2tts()
            if E2TTS_ema_model is None:
                 gr.Error("Failed to load E2-TTS model.")
                 return (16000, np.zeros(1)), None, ref_text
        ema_model = E2TTS_ema_model
    elif isinstance(model_name, list) and model_name[0] == "Custom":
         assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
         global custom_ema_model, pre_custom_path
         custom_config = model_name # Reuse variable for clarity
         if pre_custom_path != custom_config[1]: # Check against ckpt path
              show_info("Loading Custom TTS model...")
              # Ensure config is dict, not str
              cfg_data = custom_config[3]
              if isinstance(cfg_data, str):
                  try:
                       cfg_data = json.loads(cfg_data)
                  except json.JSONDecodeError:
                       gr.Error("Invalid JSON in custom model config.")
                       return (16000, np.zeros(1)), None, ref_text
              custom_ema_model = load_custom(custom_config[1], vocab_path=custom_config[2], model_cfg=cfg_data)
              if custom_ema_model is None:
                  gr.Error("Failed to load Custom TTS model.")
                  return (16000, np.zeros(1)), None, ref_text
              pre_custom_path = custom_config[1]
         ema_model = custom_ema_model
    else:
         gr.Error(f"Unknown F5-TTS model choice: {model_name}")
         return (16000, np.zeros(1)), None, ref_text


    if ema_model is None or vocoder is None:
         gr.Error("Required F5-TTS model or vocoder is not loaded.")
         return (16000, np.zeros(1)), None, ref_text

    # Preprocess (handles transcription if ref_text is empty)
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)
    except Exception as e:
        gr.Error(f"Error during audio preprocessing/transcription: {e}")
        traceback.print_exc()
        return (16000, np.zeros(1)), None, ref_text # Return dummy data

    # F5-TTS Inference Process
    try:
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
            progress=gr.Progress(track_tqdm=True), # Enable progress tracking
        )
    except Exception as e:
        gr.Error(f"Error during F5-TTS inference: {e}")
        traceback.print_exc()
        return (16000, np.zeros(1)), None, ref_text # Return dummy data

    # Remove silence (optional)
    final_wave_np = final_wave.squeeze().cpu().numpy() # Move to numpy after inference
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave_np, final_sample_rate)
            try:
                remove_silence_for_generated_wav(f.name) # This function needs error handling
                final_wave_processed, sr_processed = torchaudio.load(f.name)
                # Ensure sample rate consistency if needed, or handle potential changes
                if sr_processed != final_sample_rate:
                     print(f"Warning: Sample rate changed after silence removal ({final_sample_rate} -> {sr_processed}). Resampling not implemented here.")
                     # Potentially resample: final_wave_processed = torchaudio.functional.resample(final_wave_processed, sr_processed, final_sample_rate)
                final_wave_np = final_wave_processed.squeeze().cpu().numpy()
            except Exception as e:
                 gr.Warning(f"Failed to remove silence: {e}. Using original audio.")
                 # Fallback to using the audio before silence removal
            finally:
                os.remove(f.name) # Clean up temp file

    # Save the spectrogram (if generated)
    spectrogram_path = None
    if combined_spectrogram is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
                spectrogram_path = tmp_spectrogram.name
                save_spectrogram(combined_spectrogram, spectrogram_path)
        except Exception as e:
            print(f"Warning: Could not save spectrogram: {e}")
            spectrogram_path = None # Ensure it's None if saving fails


    # Return audio as (sample_rate, numpy_array) tuple for Gradio Audio component
    return (final_sample_rate, final_wave_np), spectrogram_path, ref_text


# =============================================================================
# Gradio UI Definition
# =============================================================================

# --- Basic TTS Tab ---
with gr.Blocks() as app_tts:
    gr.Markdown("# Basic F5-TTS Voice Cloning")
    gr.Markdown("Synthesize speech using a reference audio clip's voice. (Uses F5-TTS/E2-TTS)")
    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    gen_text_input = gr.Textbox(label="Text to Generate", lines=10)
    generate_btn = gr.Button("Synthesize (F5-TTS)", variant="primary")
    with gr.Accordion("Advanced F5-TTS Settings", open=False):
        ref_text_input = gr.Textbox(
            label="Reference Text",
            info="Optional: Leave blank to automatically transcribe the reference audio. If you enter text it will override transcription.",
            lines=2,
        )
        remove_silence = gr.Checkbox(
            label="Remove Silences (F5-TTS Postprocessing)",
            info="Experimental feature for F5-TTS output. May increase generation time.",
            value=False,
        )
        speed = gr.Slider(
            label="Speed (F5-TTS)",
            minimum=0.3,
            maximum=2.0,
            value=1.0, # Default speed 1
            step=0.1,
            info="Adjust the speed of the F5-TTS generated audio.",
        )
        nfe_slider = gr.Slider(
            label="NFE Steps (F5-TTS)",
            minimum=4,
            maximum=64,
            value=32, # Default NFE
            step=2,
            info="Number of denoising steps for F5-TTS.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Cross-Fade Duration (s) (F5-TTS)",
            minimum=0.0,
            maximum=1.0,
            value=0.15, # Default cross-fade
            step=0.01,
            info="Cross-fade duration for F5-TTS audio chunking.",
        )

    audio_output = gr.Audio(label="Synthesized Audio (F5-TTS)")
    spectrogram_output = gr.Image(label="Spectrogram (F5-TTS)")

    # Basic TTS function using the 'infer' function
    @gpu_decorator
    def basic_tts(
        ref_audio_input_path, # Renamed for clarity
        ref_text_input_val,   # Renamed for clarity
        gen_text_input_val,   # Renamed for clarity
        remove_silence_val,
        cross_fade_duration_slider_val,
        nfe_slider_val,
        speed_val,
        selected_tts_model # Pass the globally selected F5/E2/Custom model
    ):
        # Check if F5-TTS is available before proceeding
        if not F5_TTS_AVAILABLE:
             gr.Error("F5-TTS components are not available. Cannot synthesize.")
             return None, None, ref_text_input_val # Return types match outputs

        # Call the main 'infer' function
        audio_out, spectrogram_path, ref_text_out = infer(
            ref_audio_input_path,
            ref_text_input_val,
            gen_text_input_val,
            selected_tts_model, # Use the selected model (F5/E2/Custom)
            remove_silence_val,
            cross_fade_duration=cross_fade_duration_slider_val,
            nfe_step=nfe_slider_val,
            speed=speed_val,
            show_info=gr.Info # Use Gradio's info display
        )
        # Return values assigned to the correct Gradio components
        return audio_out, spectrogram_path, ref_text_out

    # Connect button click to the basic_tts function
    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            cross_fade_duration_slider,
            nfe_slider,
            speed,
            gr.State(tts_model_choice) # Pass the current model choice as state
        ],
        outputs=[audio_output, spectrogram_output, ref_text_input], # Assign outputs
    )


# --- Multi-Speech Tab ---
# (Keeping this section largely as is, as it relies on the F5-TTS 'infer' function)
def parse_speechtypes_text(gen_text):
    # ... (keep original parse_speechtypes_text function)
    pattern = r"\{(.*?)\}"
    tokens = re.split(pattern, gen_text)
    segments = []
    current_style = "Regular" # Default style if not specified at the beginning
    for i, token in enumerate(tokens):
        token = token.strip()
        if not token: # Skip empty strings resulting from split
            continue
        if i % 2 == 0:
            # This is text content
            if token: # Only add if there's actual text
                segments.append({"style": current_style, "text": token})
        else:
            # This is a style tag
            current_style = token # Update the current style
    return segments


with gr.Blocks() as app_multistyle:
    gr.Markdown("# Multi-Speech Generation (F5-TTS)")
    gr.Markdown("Generate speech with different styles or speakers using F5-TTS voice cloning.")
    # ... (keep original Markdown examples)
    gr.Markdown(
         """
         **Example Input:**
         {Regular} Hello, I'd like to order a sandwich please.
         {Surprised} What do you mean you're out of bread?
         {Sad} I really wanted a sandwich though...
         {Angry} You know what, darn you and your little shop!
         {Whisper} I'll just go back home and cry now.
         {Shouting} Why me?!

         **Example Input 2:**
         {Speaker1_Happy} Hello, I'd like to order a sandwich please.
         {Speaker2_Regular} Sorry, we're out of bread.
         {Speaker1_Sad} I really wanted a sandwich though...
         {Speaker2_Whisper} I'll give you the last one I was hiding.
         """
     )

    gr.Markdown(
        "Upload a reference audio clip for each speech style/speaker. The first style ('Regular' by default) is mandatory."
    )

    # --- Define dynamic UI components for speech types ---
    MAX_SPEECH_TYPES = 10 # Reduced for manageability, increase if needed
    speech_type_rows = []
    speech_type_names = []
    speech_type_audios = []
    speech_type_ref_texts = []
    speech_type_delete_btns = []
    speech_type_insert_btns = []

    # Create the first mandatory row ('Regular')
    with gr.Row(visible=True) as regular_row:
        with gr.Column(scale=1):
             regular_name = gr.Textbox(value="Regular", label="Style/Speaker Name 1", interactive=True)
             regular_insert = gr.Button("Insert Label", variant="secondary", size="sm")
             regular_delete = gr.Button("Delete", variant="stop", size="sm", visible=False) # Cannot delete the first one
        with gr.Column(scale=3):
             regular_audio = gr.Audio(label="Reference Audio 1", type="filepath")
             regular_ref_text = gr.Textbox(label="Ref. Text 1 (Optional)", lines=1)

    speech_type_rows.append(regular_row)
    speech_type_names.append(regular_name)
    speech_type_audios.append(regular_audio)
    speech_type_ref_texts.append(regular_ref_text)
    speech_type_delete_btns.append(regular_delete) # Placeholder, not functional
    speech_type_insert_btns.append(regular_insert)

    # Create hidden rows for additional types
    for i in range(1, MAX_SPEECH_TYPES):
        with gr.Row(visible=False) as row:
            with gr.Column(scale=1):
                name_input = gr.Textbox(label=f"Style/Speaker Name {i+1}")
                insert_btn = gr.Button("Insert Label", variant="secondary", size="sm")
                delete_btn = gr.Button("Delete", variant="stop", size="sm")
            with gr.Column(scale=3):
                audio_input = gr.Audio(label=f"Reference Audio {i+1}", type="filepath")
                ref_text_input = gr.Textbox(label=f"Ref. Text {i+1} (Optional)", lines=1)

        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_delete_btns.append(delete_btn)
        speech_type_insert_btns.append(insert_btn)

    # Button to add speech type
    add_speech_type_btn = gr.Button("Add Another Speech Style/Speaker")
    speech_type_count_state = gr.State(1) # Start with 1 visible row

    def add_speech_type_fn(current_count):
        row_updates = [gr.update() for _ in range(MAX_SPEECH_TYPES)]
        if current_count < MAX_SPEECH_TYPES:
            row_updates[current_count] = gr.update(visible=True)
            new_count = current_count + 1
            button_update = gr.update(visible=new_count < MAX_SPEECH_TYPES) # Hide button if max reached
        else:
            gr.Warning(f"Maximum number of speech types ({MAX_SPEECH_TYPES}) reached.")
            new_count = current_count
            button_update = gr.update(visible=False)
        return [*row_updates, new_count, button_update]

    add_speech_type_btn.click(
        add_speech_type_fn,
        inputs=[speech_type_count_state],
        outputs=[*speech_type_rows, speech_type_count_state, add_speech_type_btn]
    )

    # Function generator for delete buttons
    def make_delete_fn(index_to_delete):
        def delete_speech_type_fn(current_count):
            # This function needs to return updates for ALL rows,
            # the count state, and the add button visibility.
            all_row_updates = [gr.update() for _ in range(MAX_SPEECH_TYPES)]
            # Hide the target row and clear its inputs
            all_row_updates[index_to_delete] = gr.update(visible=False)
            name_update = gr.update(value="")
            audio_update = gr.update(value=None)
            ref_text_update = gr.update(value="")

            # Note: Correctly managing the state after deletion (shifting rows up, etc.)
            # is complex with current Gradio capabilities for dynamic UI.
            # A simpler approach is just hiding and clearing. The user might need to be careful
            # about indices if they delete from the middle.
            # We won't decrement count here to avoid complex state management.
            # The user can just add new ones until the max.
            # We ensure the add button is visible if max isn't reached.
            add_btn_update = gr.update(visible=current_count < MAX_SPEECH_TYPES) # Re-evaluate add button visibility

            # Return updates for the specific row's components + all rows + count + add button
            return name_update, audio_update, ref_text_update, *all_row_updates, current_count, add_btn_update
        return delete_speech_type_fn

    # Connect delete buttons (skip the first one)
    for i in range(1, MAX_SPEECH_TYPES):
        delete_fn = make_delete_fn(i)
        speech_type_delete_btns[i].click(
            delete_fn,
            inputs=[speech_type_count_state],
            outputs=[
                speech_type_names[i], speech_type_audios[i], speech_type_ref_texts[i], # Updates for the cleared row
                *speech_type_rows, # Updates for all row visibilities
                speech_type_count_state, # Pass count state through
                add_speech_type_btn # Update add button visibility
            ],
        )


    # Text input for the prompt
    gen_text_input_multistyle = gr.Textbox(
        label="Text to Generate with {Style/Speaker} Tags",
        lines=10,
        placeholder="Example:\n{Regular} This is standard voice.\n{Happy} This sounds happy!",
    )

    # Function generator for insert buttons
    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name):
            current_text = current_text or ""
            # Use the actual name from the input box, default if empty
            speech_type_name = speech_type_name.strip() or f"Style_{index+1}"
            # Add a space after the tag if the current text isn't empty and doesn't end with space/newline
            separator = " " if current_text and not current_text.endswith(("\n", " ")) else ""
            updated_text = current_text + separator + f"{{{speech_type_name}}} "
            return gr.update(value=updated_text) # Use gr.update for direct value setting
        return insert_speech_type_fn

    # Connect insert buttons
    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i]],
            outputs=gen_text_input_multistyle,
            queue=False # Make insertion instant
        )


    with gr.Accordion("Advanced F5-TTS Settings", open=False):
        remove_silence_multistyle = gr.Checkbox(
            label="Remove Silences (F5-TTS Postprocessing)",
             info="Applies F5-TTS silence removal to each generated segment.",
            value=True,
        )
        # Add Speed, NFE, Crossfade here if segment-specific control is needed (complex)
        # For simplicity, use global settings or fixed values for multi-style generation

    # Generate button
    generate_multistyle_btn = gr.Button("Generate Multi-Style Speech (F5-TTS)", variant="primary")

    # Output audio
    audio_output_multistyle = gr.Audio(label="Synthesized Multi-Style Audio")

    # Multi-style generation function
    @gpu_decorator
    def generate_multistyle_speech(
        gen_text, # The main text with tags
        *args,    # Captures all dynamic inputs (names, audios, ref_texts, remove_silence flag)
    ):
         # Check F5-TTS availability
         if not F5_TTS_AVAILABLE:
              gr.Error("F5-TTS components are not available. Cannot generate multi-style speech.")
              # Need to return dummy data matching the expected number of outputs
              num_ref_texts = MAX_SPEECH_TYPES
              return [None] + [gr.update(value="Error") for _ in range(num_ref_texts)]

         # Unpack args based on MAX_SPEECH_TYPES
         names = args[0:MAX_SPEECH_TYPES]
         audios = args[MAX_SPEECH_TYPES : 2 * MAX_SPEECH_TYPES]
         ref_texts = args[2 * MAX_SPEECH_TYPES : 3 * MAX_SPEECH_TYPES]
         remove_silence_flag = args[3 * MAX_SPEECH_TYPES]
         # Assuming global TTS model choice applies here too
         selected_tts_model = args[3 * MAX_SPEECH_TYPES + 1]

         # Collect valid speech types provided by the user
         speech_types_map = OrderedDict()
         visible_ref_text_outputs = [] # To store updated ref texts for visible rows
         all_ref_text_outputs = [gr.update() for _ in range(MAX_SPEECH_TYPES)] # Initialize all outputs

         # Check which rows are visible (crudely, by checking if name/audio is provided)
         for i in range(MAX_SPEECH_TYPES):
             name = names[i].strip() if names[i] else None
             audio = audios[i]
             ref_text = ref_texts[i] if ref_texts[i] else ""
             # Consider a type valid if it has a name AND audio path
             if name and audio:
                 speech_types_map[name] = {"audio": audio, "ref_text": ref_text, "index": i}
                 visible_ref_text_outputs.append(ref_text) # Store initial ref text for output later
             elif i == 0 and not name: # Handle case where 'Regular' name was deleted
                  gr.Warning("The first speech type must have a name (default is 'Regular').")
                  # Provide appropriate return structure on error
                  return [None] + [gr.update() for _ in range(MAX_SPEECH_TYPES)]
             # No need to add placeholder like '@{i}@' if it's not used


         if not speech_types_map:
              gr.Warning("Please provide at least one valid speech type with a name and reference audio.")
              return [None] + [gr.update() for _ in range(MAX_SPEECH_TYPES)]

         # Parse the input text
         segments = parse_speechtypes_text(gen_text)
         if not segments:
              gr.Warning("No text segments found to generate.")
              return [None] + [gr.update() for _ in range(MAX_SPEECH_TYPES)]


         generated_audio_segments = []
         current_sr = None # Track sample rate

         default_style_name = list(speech_types_map.keys())[0] # Use the first defined style as default

         print(f"Starting multi-style generation. Styles: {list(speech_types_map.keys())}")
         print(f"Parsed segments: {segments}")

         with gr.Progress(track_tqdm=True) as progress:
             progress(0, desc="Starting Generation")
             for idx, segment in enumerate(segments):
                 style_name = segment["style"]
                 text_to_gen = segment["text"]
                 progress(idx / len(segments), desc=f"Segment {idx+1}/{len(segments)} ({style_name})")

                 # Find the style details, fallback to default if not found
                 if style_name not in speech_types_map:
                     gr.Warning(f"Style '{style_name}' not found or inactive. Using default style '{default_style_name}'.")
                     style_name = default_style_name

                 style_details = speech_types_map[style_name]
                 ref_audio_path = style_details["audio"]
                 ref_text_val = style_details["ref_text"]
                 output_index = style_details["index"] # Index for updating ref_text output

                 print(f"  Generating: Style='{style_name}', Text='{text_to_gen[:30]}...'")

                 try:
                     # Call the F5-TTS 'infer' function for this segment
                     # Using simplified settings for multi-style for now
                     # TODO: Expose NFE/Crossfade/Speed per-style if needed
                     audio_out, _, ref_text_out = infer(
                         ref_audio_path,
                         ref_text_val,
                         text_to_gen,
                         selected_tts_model, # Use the globally selected F5/E2/Custom
                         remove_silence_flag,
                         cross_fade_duration=0.1, # Fixed values for multi-style
                         nfe_step=32,             # Fixed values for multi-style
                         speed=1.0,               # Fixed values for multi-style
                         show_info=print          # Print info to console
                     )

                     if audio_out is None or audio_out[1].size == 0:
                          gr.Warning(f"Segment '{style_name}' produced no audio. Skipping.")
                          continue

                     sr, audio_data = audio_out

                     # Ensure consistent sample rate
                     if current_sr is None:
                         current_sr = sr
                     elif current_sr != sr:
                          gr.Warning(f"Sample rate mismatch ({sr} vs {current_sr}). Resampling not implemented. Audio might be inconsistent.")
                          # Potentially resample audio_data here if needed
                          # audio_data = torchaudio.functional.resample(torch.from_numpy(audio_data), sr, current_sr).numpy()

                     generated_audio_segments.append(audio_data)

                     # Update the ref_text in the map and the output list
                     speech_types_map[style_name]["ref_text"] = ref_text_out
                     all_ref_text_outputs[output_index] = gr.update(value=ref_text_out)


                 except Exception as e:
                     gr.Error(f"Error generating segment for style '{style_name}': {e}")
                     traceback.print_exc()
                     # Decide whether to stop or continue
                     # return [None] + [gr.update() for _ in range(MAX_SPEECH_TYPES)] # Option: Stop on error

         progress(1, desc="Concatenating Audio")
         # Concatenate all audio segments
         if generated_audio_segments and current_sr is not None:
             try:
                 final_audio_data = np.concatenate(generated_audio_segments)
                 final_output_audio = (current_sr, final_audio_data)
                 print("Multi-style generation complete.")
             except ValueError as e:
                  gr.Error(f"Error concatenating audio segments: {e}. Check for empty segments.")
                  final_output_audio = None
         else:
             gr.Warning("No audio segments were successfully generated.")
             final_output_audio = None

         return [final_output_audio] + all_ref_text_outputs


    # Connect the generate button
    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            gen_text_input_multistyle,
            *speech_type_names,
            *speech_type_audios,
            *speech_type_ref_texts,
            remove_silence_multistyle,
             gr.State(tts_model_choice) # Pass F5/E2/Custom choice
        ],
        outputs=[audio_output_multistyle] + speech_type_ref_texts, # Output audio + updated ref texts
    )

    # --- Validation (Optional but good) ---
    # Basic check if tags in text exist in defined styles
    def validate_speech_types(gen_text, *names):
         defined_styles = set(name.strip() for name in names if name and name.strip())
         if not defined_styles: defined_styles.add("Regular") # Assume Regular exists if nothing else defined

         segments = parse_speechtypes_text(gen_text)
         used_styles = set(segment["style"] for segment in segments)

         missing_styles = used_styles - defined_styles
         # Also check if default style is used but not defined (edge case)
         if any(seg["style"]=="Regular" for seg in segments) and "Regular" not in defined_styles and list(defined_styles)[0] != "Regular":
             # If Regular is used implicitly but not defined, and the first defined is not named Regular
             pass # Allow fallback to the first defined style implicitly

         if missing_styles:
             gr.Warning(f"Styles used in text but not defined: {missing_styles}. Will use default style.")
             # Don't disable button, just warn. Generation logic handles fallback.
             # return gr.update(interactive=False)
             return gr.update(interactive=True)
         else:
             # Clear warning if any
             return gr.update(interactive=True)

    # Connect validation to text input change
    gen_text_input_multistyle.change(
        validate_speech_types,
        inputs=[gen_text_input_multistyle, *speech_type_names],
        outputs=generate_multistyle_btn,
        queue=False # Make validation fast
    )


# --- Voice Chat Tab ---
with gr.Blocks() as app_chat:
    gr.Markdown("# Voice Chat with AI (using Google TTS)")
    gr.Markdown(
        """
        Have a conversation with the Locksmith Assistant AI.
        1. Ensure the required **Google API Key** and **Nebius API Key** are set as environment variables.
        2. Record your message or type it.
        3. The AI will respond using Google Text-to-Speech.
        """
        # Removed note about reference audio for transcription
    )

    # --- Chat Model Selection (Nebius Only for now) ---
    gr.Markdown("Using **Llama-Fast (Nebius API)** for chat responses.")
    chat_model_state = "meta-llama/Meta-Llama-3.1-70B-Instruct-fast"
    chat_tokenizer_state = None

    chat_interface_container = gr.Column(visible=True)

    with chat_interface_container:
        # REMOVED the Row containing ref_audio_chat and ref_text_chat
        # Kept the settings accordion, but removed ref_text_chat from inside it
        with gr.Accordion("Chat & AI Settings", open=False):
             system_prompt_chat = gr.Textbox(
                 label="System Prompt",
                 value=system_prompt,
                 lines=10,
                 info="Instructions for the AI assistant."
             )
             # REMOVED ref_text_chat Textbox here
             speed_slider_google = gr.Slider(
                 label="AI Voice Speed (Google TTS)",
                 minimum=0.3, maximum=2.0, value=1.1, step=0.05,
                 info="Adjust playback speed of the AI's Google voice."
             )
             google_voice_chat = gr.Dropdown(
                 label="AI Voice (Google TTS)",
                 choices=[
                     "en-US-Wavenet-D", "en-US-Wavenet-F", "en-US-Wavenet-A", "en-US-Wavenet-E",
                     "en-US-Neural2-J", "en-US-Neural2-F", "en-GB-Wavenet-B", "en-GB-Wavenet-F",
                     "en-AU-Wavenet-B", "en-AU-Wavenet-C",
                 ],
                 value="en-US-Wavenet-D",
                 info="Select the Google TTS voice for the AI."
             )

        chatbot_interface = gr.Chatbot(
            label="Conversation", height=500, bubble_full_width=False
        )
        audio_output_chat = gr.Audio(
            label="AI Response Audio", autoplay=True, interactive=False
        )

        with gr.Row():
            audio_input_chat = gr.Audio(
                label="Speak Your Message", sources=["microphone"], type="filepath", streaming=False
            )
            text_input_chat = gr.Textbox(
                label="Type Your Message", lines=3, placeholder="Type here or use the microphone above...", scale=4
            )
            with gr.Column(scale=1):
                send_btn_chat = gr.Button("Send Message", variant="primary")
                clear_btn_chat = gr.Button("Clear Conversation", variant="secondary")

        def get_initial_conversation_state(system_prompt_value):
            return [{"role": "system", "content": system_prompt_value}]

        conversation_state = gr.State(value=get_initial_conversation_state(system_prompt))

        # --- Chat Processing Logic ---

        # MODIFIED: Removed ref_audio_for_transcription, ref_text_for_transcription parameters
        def process_user_input(
            audio_path, text_input, current_history, conv_state_list
            ):

            user_message = ""
            # 1. Determine user message (from audio or text)
            if audio_path:
                print(f"Processing user audio input: {audio_path}")
                # MODIFIED: Call preprocess_ref_audio_text WITHOUT ref_audio_path/ref_text
                try:
                    _, transcribed_text = preprocess_ref_audio_text(
                        audio_path,
                        "", # Don't pass user text here, we want transcription
                        # ref_audio_path=None, # Removed
                        # ref_text=None,       # Removed
                        show_info=print
                    )
                    user_message = transcribed_text.strip()
                    print(f"Transcription result: '{user_message}'")
                    if not user_message:
                        gr.Warning("Transcription failed or produced empty text.")
                        return current_history, conv_state_list, gr.update(value="")
                except Exception as e:
                    gr.Error(f"Error during user audio transcription: {e}")
                    traceback.print_exc()
                    return current_history, conv_state_list, gr.update(value="")
            elif text_input and text_input.strip():
                user_message = text_input.strip()
                print(f"Processing user text input: '{user_message}'")
            else:
                gr.Warning("Please type a message or record audio.")
                return current_history, conv_state_list, gr.update()

            if not user_message:
                 return current_history, conv_state_list, gr.update(value="")

            # 2. Update conversation state and history
            current_conv_state = list(conv_state_list)
            current_conv_state.append({"role": "user", "content": user_message})
            updated_history = current_history + [[user_message, None]]

            # 3. Generate AI response (text only)
            ai_response_text = generate_nebius_response(current_conv_state)

            # 4. Update state and history with AI response
            current_conv_state.append({"role": "assistant", "content": ai_response_text})
            updated_history[-1][1] = ai_response_text

            # 5. Return updated history, state, and clear input textbox
            return updated_history, current_conv_state, gr.update(value="")

        # Function to generate Google TTS audio (no changes needed here)
        def generate_google_audio_response(
            current_history, google_api_key_val, google_voice_val, speed_val
            ):
            # ... (keep existing function body) ...
            print("Attempting to generate Google TTS audio for the last response.")
            if not current_history:
                print("  History is empty, no audio to generate.")
                return None # No audio output

            last_user_msg, last_ai_response = current_history[-1]

            if last_ai_response is None:
                print("  Last AI response is None, cannot generate audio.")
                return None

            if not last_ai_response.strip():
                print("  Last AI response is empty, skipping audio generation.")
                return None

            audio_file_path = google_tts_via_api_key(
                text=last_ai_response,
                api_key=google_api_key_val,
                voice_name=google_voice_val,
                speaking_rate=speed_val
            )

            if audio_file_path:
                print(f"  Successfully generated Google TTS audio: {audio_file_path}")
                return gr.update(value=audio_file_path, autoplay=True)
            else:
                print("  Failed to generate Google TTS audio.")
                gr.Warning("Could not generate AI voice response audio.")
                return None # Return None if TTS failed

        # --- Event Handling ---

        # MODIFIED: Removed ref_audio_chat, ref_text_chat from process_inputs list
        process_inputs = [
            audio_input_chat,
            text_input_chat,
            chatbot_interface,
            conversation_state,
            # ref_audio_chat, # Removed
            # ref_text_chat   # Removed
        ]
        process_outputs = [
            chatbot_interface,
            conversation_state,
            text_input_chat
        ]

        # audio_gen_inputs/outputs remain the same
        audio_gen_inputs = [
            chatbot_interface,
            gr.State(GOOGLE_API_KEY),
            google_voice_chat,
            speed_slider_google
        ]
        audio_gen_outputs = [
            audio_output_chat
        ]

        # --- Event Triggers (No changes needed in the .then structure) ---
        # Trigger processing when microphone stops recording
        audio_input_chat.stop_recording(
            process_user_input,
            inputs=process_inputs,
            outputs=process_outputs
        ).then(
            generate_google_audio_response,
            inputs=audio_gen_inputs,
            outputs=audio_gen_outputs
        ).then(
             lambda: None, None, audio_input_chat
        )

        # Trigger processing when text is submitted (Enter key)
        text_input_chat.submit(
            process_user_input,
            inputs=process_inputs,
            outputs=process_outputs
        ).then(
            generate_google_audio_response,
            inputs=audio_gen_inputs,
            outputs=audio_gen_outputs
        )

        # Trigger processing when Send button is clicked
        send_btn_chat.click(
            process_user_input,
            inputs=process_inputs,
            outputs=process_outputs
        ).then(
            generate_google_audio_response,
            inputs=audio_gen_inputs,
            outputs=audio_gen_outputs
        )

        # Clear conversation button (No changes needed)
        def clear_conversation(system_prompt_value):
            print("Clearing conversation.")
            initial_state = get_initial_conversation_state(system_prompt_value)
            return [], initial_state, None, None

        clear_btn_chat.click(
            clear_conversation,
            inputs=[system_prompt_chat],
            outputs=[chatbot_interface, conversation_state, audio_output_chat, text_input_chat],
            queue=False
        )

        # Update system prompt and reset conversation (No changes needed)
        def update_system_prompt(new_prompt):
            print("System prompt updated. Resetting conversation.")
            gr.Info("System prompt updated. Conversation has been reset.")
            new_conv_state = get_initial_conversation_state(new_prompt)
            return [], new_conv_state, None

        system_prompt_chat.change(
            update_system_prompt,
            inputs=system_prompt_chat,
            outputs=[chatbot_interface, conversation_state, audio_output_chat],
        )


# =============================================================================
# Main App Structure (Tabs and Global Settings)
# =============================================================================
# Custom CSS (Optional - Keep commented out if not needed)
# custom_css = """ ... """

with gr.Blocks(title=APP_TITLE) as app: # Removed css=custom_css
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(
        """
        **Features**:
        - **Basic TTS**: Clone voice from reference audio using F5-TTS/E2-TTS.
        - **Multi-Speech**: Generate audio with multiple speakers/styles using F5-TTS.
        - **Voice Chat**: Converse with an AI assistant using Google TTS for the AI's voice.

        **Important Notes**:
        - **Google API Key**: The Voice Chat tab requires a `GOOGLE_API_KEY` environment variable.
        - **Nebius API Key**: The Voice Chat tab requires a `NEBIUS_API_KEY` environment variable for the AI responses.
        - **F5-TTS Models**: Basic TTS and Multi-Speech tabs require F5-TTS models to be downloaded (may happen automatically on first run if `cached_path` is used).
        - **Reference Audio**: For Basic/Multi-Speech, use short (<15s) WAV/MP3 clips. For Voice Chat, reference audio helps transcribe *your* spoken input.
        """
    )

    # --- Global F5-TTS Model Selection (for Basic & Multi-Speech tabs) ---
    # This selection does NOT affect the Voice Chat tab, which uses Google TTS.
    last_used_custom_path = files("f5_tts.infer").joinpath(".cache/last_used_custom_model_info_v1.txt") if F5_TTS_AVAILABLE else None

    def load_last_used_custom():
        # Return defaults if file doesn't exist or F5-TTS is unavailable
        if not last_used_custom_path or not last_used_custom_path.exists():
            return DEFAULT_TTS_MODEL_CFG
        try:
            custom = []
            with open(last_used_custom_path, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            # Validate structure, fallback to default if incorrect
            if len(custom) == 3:
                 # Basic check if config is JSON-like
                 try:
                      json.loads(custom[2])
                      return custom
                 except json.JSONDecodeError:
                      print("Warning: Invalid JSON in cached custom config. Using defaults.")
                      return DEFAULT_TTS_MODEL_CFG
            else:
                print("Warning: Cached custom model info has incorrect format. Using defaults.")
                return DEFAULT_TTS_MODEL_CFG
        except Exception as e:
            print(f"Warning: Could not read cached custom model info: {e}. Using defaults.")
            return DEFAULT_TTS_MODEL_CFG

    # Load initial custom config values
    initial_custom_cfg = load_last_used_custom()

    with gr.Accordion("F5-TTS Model Selection (for Basic & Multi-Speech Tabs)", open=False):
         with gr.Row():
              f5_tts_choices = [DEFAULT_TTS_MODEL]
              if F5_TTS_AVAILABLE: # Only add E2 if F5 components loaded
                   f5_tts_choices.append("E2-TTS")
              if not USING_SPACES and F5_TTS_AVAILABLE: # Only add Custom if local and F5 available
                   f5_tts_choices.append("Custom")

              choose_tts_model_radio = gr.Radio(
                   choices=f5_tts_choices,
                   label="Choose F5-TTS Model",
                   value=DEFAULT_TTS_MODEL if DEFAULT_TTS_MODEL in f5_tts_choices else (f5_tts_choices[0] if f5_tts_choices else None),
                   info="This selection applies ONLY to the 'Basic TTS' and 'Multi-Speech' tabs.",
                   interactive=F5_TTS_AVAILABLE # Disable if F5-TTS isn't working
              )
         with gr.Row(visible=False) as custom_model_options_row: # Initially hidden
              custom_ckpt_path_dd = gr.Dropdown(
                   choices=[initial_custom_cfg[0]], # Start with cached/default
                   value=initial_custom_cfg[0],
                   allow_custom_value=True,
                   label="Custom Model Path/HF ID",
                   info="Local path or hf://user/repo/model.safetensors",
              )
              custom_vocab_path_dd = gr.Dropdown(
                   choices=[initial_custom_cfg[1]], # Start with cached/default
                   value=initial_custom_cfg[1],
                   allow_custom_value=True,
                   label="Custom Vocab Path/HF ID (Optional)",
                   info="Local path or hf://user/repo/vocab.txt",
              )
              custom_model_cfg_dd = gr.Dropdown(
                   choices=[initial_custom_cfg[2]], # Start with cached/default
                   value=initial_custom_cfg[2],
                   allow_custom_value=True,
                   label="Custom Model Config (JSON)",
                   info="Model configuration as a JSON string.",
              )

    def switch_f5_tts_model_visibility(choice):
        global tts_model_choice # Update the global choice for Basic/Multi tabs
        tts_model_choice = choice
        if choice == "Custom":
            # Load last used custom settings into the dropdowns when 'Custom' is selected
            ckpt, vocab, cfg = load_last_used_custom()
            return (gr.update(visible=True), # Show row
                    gr.update(value=ckpt, choices=[ckpt]), # Update dropdowns
                    gr.update(value=vocab, choices=[vocab]),
                    gr.update(value=cfg, choices=[cfg]))
        else:
            return (gr.update(visible=False), # Hide row
                    gr.update(), gr.update(), gr.update()) # No change to hidden dropdowns

    # When radio changes, update visibility and potentially dropdown values
    choose_tts_model_radio.change(
        switch_f5_tts_model_visibility,
        inputs=[choose_tts_model_radio],
        outputs=[custom_model_options_row, custom_ckpt_path_dd, custom_vocab_path_dd, custom_model_cfg_dd],
        queue=False
    )

    # Function to save custom model info when dropdowns change
    def save_custom_model_info(ckpt, vocab, cfg):
         # Update the global variable ONLY when custom is active and inputs change
         if tts_model_choice == "Custom":
              # Validate config is valid JSON before saving
              try:
                  parsed_cfg = json.loads(cfg)
                  tts_model_choice = ["Custom", ckpt, vocab, parsed_cfg] # Update global state with parsed cfg
                  if last_used_custom_path:
                      try:
                           last_used_custom_path.parent.mkdir(parents=True, exist_ok=True)
                           with open(last_used_custom_path, "w", encoding="utf-8") as f:
                                f.write(f"{ckpt}\n{vocab}\n{cfg}\n") # Save the raw JSON string
                           print("Saved custom model configuration.")
                      except Exception as e:
                           print(f"Warning: Could not save custom model info: {e}")
                  else:
                       print("Warning: F5-TTS not available, cannot save custom model info.")
              except json.JSONDecodeError:
                  gr.Warning("Invalid JSON in model config. Configuration not saved.")
              except Exception as e:
                  gr.Error(f"Error processing custom model config: {e}")


    # When custom dropdowns change, save the info
    custom_ckpt_path_dd.change(save_custom_model_info, inputs=[custom_ckpt_path_dd, custom_vocab_path_dd, custom_model_cfg_dd], queue=False)
    custom_vocab_path_dd.change(save_custom_model_info, inputs=[custom_ckpt_path_dd, custom_vocab_path_dd, custom_model_cfg_dd], queue=False)
    custom_model_cfg_dd.change(save_custom_model_info, inputs=[custom_ckpt_path_dd, custom_vocab_path_dd, custom_model_cfg_dd], queue=False)


    # --- Tabbed Interface ---
    gr.TabbedInterface(
        [app_tts, app_multistyle, app_chat],
        ["Basic TTS (F5)", "Multi-Speech (F5)", "Voice Chat (Google TTS)"],
    )

# =============================================================================
# App Launch Logic
# =============================================================================
# (Keep the click command and main function as is)
@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default="0.0.0.0", help="Host to run the app on (0.0.0.0 for public access)") # Default to 0.0.0.0
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Create a public Gradio share link (use with caution)",
)
@click.option("--api", "-a", default=False, is_flag=True, help="Enable Gradio API access (/api)") # Default API off
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='URL path prefix for running behind a reverse proxy (e.g., /myapp)',
)
@click.option(
    "--inbrowser",
    "-i",
    is_flag=True,
    default=False,
    help="Automatically launch the interface in the default web browser",
)
def main(port, host, share, api, root_path, inbrowser):
    # Check for necessary API keys before launching
    keys_missing = []
    if not NEBIUS_API_KEY or NEBIUS_API_KEY == "YOUR_NEBIUS_API_KEY_HERE":
         keys_missing.append("NEBIUS_API_KEY")
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
         keys_missing.append("GOOGLE_API_KEY")

    if keys_missing:
         print("\n" + "="*40)
         print("!!! IMPORTANT: Required API Key(s) Missing !!!")
         for key in keys_missing:
              print(f"- Set the '{key}' environment variable.")
         print("The Voice Chat tab will not function correctly without these keys.")
         print("="*40 + "\n")
         # Decide if you want to exit or just warn
         # exit(1) # Option: Exit if keys are critical

    print("Starting Gradio app...")
    # Make API configurable via the flag
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_api=api, # Show link if enabled
        root_path=root_path,
        inbrowser=inbrowser,
        # auth=("user", "password"), # Example basic auth
        # ssl_keyfile="key.pem",     # Example HTTPS
        # ssl_certfile="cert.pem",
    )

if __name__ == "__main__":
    # Check if running in Hugging Face Spaces
    if os.environ.get("SPACE_ID"):
         print("Running in Hugging Face Spaces environment.")
         # Launch for Spaces
         app.queue().launch()
    else:
         # Run locally using click arguments
         main()