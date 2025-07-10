# ruff: noqa: E402
# Allows ruff to ignore E402: module level import not at top of file

import os
import base64
import json
import tempfile
import traceback
import requests  # For TTS call
import gradio as gr
import numpy as np
from openai import OpenAI
import torch
import torchaudio
import torchaudio.transforms as T  # For resampling
import time
import soundfile as sf  # For saving buffered audio
import torch
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import gradio as gr
import traceback
import tempfile
import silero_vad
os.environ["NEBIUS_API_KEY"] = "eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExNzk1NTg1MzgyNzYzMDYxNDQwMCIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwMjgwMzYzNSwidXVpZCI6IjE4NjRlMWM4LTM1NmQtNDY0Ny04NTdlLTJlN2UxYzg2ODJkYSIsIm5hbWUiOiJzYWhhcmY1IiwiZXhwaXJlc19hdCI6IjIwMzAtMDQtMTlUMDQ6MzM6NTUrMDAwMCJ9.632FHIOBhPM3iNlIBBxPC-16uc3FKoiAC1iqVKXUO5g"
os.environ["GOOGLE_API_KEY"] = "AIzaSyC2MC-S458QlwKKa1EFyZYH-HJJkAngSRw"
# --- Silero VAD Setup ---
VAD_ENABLED = False
VAD_MODEL = None
VAD_UTILS = None
VADIterator = None
try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    # Load Silero VAD model
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    VAD_MODEL = vad_model
    VAD_UTILS = utils
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    VAD_ENABLED = True
    print("Silero VAD model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load Silero VAD model: {e}")
    print("VAD and automatic speech detection will be disabled.")
    VAD_ENABLED = False
VAD_SAMPLE_RATE = 16000
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
VAD_MODEL = vad_model
# --- Configuration & API Key Handling ---
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

missing_keys = []
if not NEBIUS_API_KEY: missing_keys.append("NEBIUS_API_KEY")
if not GOOGLE_API_KEY: missing_keys.append("GOOGLE_API_KEY")

if missing_keys:
    warning_message = (
        "\n" + "="*40 + "\n"
        "!!! WARNING: Required API Key(s) Missing !!!\n"
        f"The following environment variables are not set: {', '.join(missing_keys)}\n"
        "Please set them before running the script.\n"
        "="*40 + "\n"
    )
    print(warning_message)

# --- Nebius API Client Initialization ---
nebius_client = None
if NEBIUS_API_KEY:
    try:
        nebius_client = OpenAI(base_url="https://api.studio.nebius.com/v1/", api_key=NEBIUS_API_KEY)
        print("Nebius API client initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Nebius API client: {str(e)}")
else:
    print("ERROR: Nebius API client cannot be initialized (NEBIUS_API_KEY missing).")

# --- Transcription Function Import ---
TRANSCRIPTION_AVAILABLE = False
try:
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    TRANSCRIPTION_AVAILABLE = True
    print("Audio transcription function loaded successfully.")
except ImportError as e:
    print(f"Warning: Failed to import 'preprocess_ref_audio_text' ({e}). Transcription will fail.")
    def preprocess_ref_audio_text(audio_path, text, **kwargs):
        raise RuntimeError("Audio transcription function 'preprocess_ref_audio_text' is not available.")

# --- App Title & System Prompt ---
APP_TITLE = "Locksmith AI Assistant (VAD Enabled)"
system_prompt = """You are Joe, an AI assistant for 1st Choice Locksmith Services, specializing in the Houston, TX area (40-mile radius). Your ONLY goal is to efficiently gather essential information for a locksmith dispatch:
1. Customer's Full Name
2. Full Service Address (Street, City, ZIP Code)
3. Customer's Phone Number
4. Type of Locksmith Need (e.g., car lockout, house lockout, key replacement)
5. Brief Description/Notes (optional, e.g., 'key broke in lock')

**Conversation Rules:**
- Be empathetic but brief. This is often an emergency service.
- Ask only ONE question at a time.
- Do NOT ask unnecessary questions or provide extra details.
- Do NOT ask for city/zip specifically, let the customer provide the full address.
- Do NOT ask for technical details about the lock/key.
- Confirm ALL gathered details (Name, Address, Phone, Job Type) ONCE at the end.
- If asked about price: State clearly, "A technician will call you in 1-2 minutes with an exact quote for your situation." Never give any price estimate, not even a service call fee.
- Once details are confirmed, end the call politely: "Thank you, [Name]. A technician from 1st Choice Locksmith Services will call you within 1-2 minutes. Help is on the way!"
"""

# --- Core API Functions ---
def generate_nebius_response(messages):
    """Generate response using Nebius API"""
    if not nebius_client:
        gr.Error("Nebius API client is not initialized.")
        return "I'm sorry, the chat service is currently unavailable."
    try:
        formatted_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        print(f"Sending {len(formatted_messages)} messages to Nebius API.")
        response = nebius_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
            max_tokens=512, temperature=0.6, top_p=0.9, messages=formatted_messages
        )
        ai_response = response.choices[0].message.content
        print(f"Received Nebius response: '{ai_response[:100]}...'")
        return ai_response
    except Exception as e:
        print(f"ERROR generating Nebius response: {str(e)}"); traceback.print_exc()
        if "authentication" in str(e).lower(): gr.Warning("Nebius API auth failed. Check key.")
        else: gr.Warning(f"Nebius Error: {e}")
        return "Sorry, encountered an error generating the response."

def google_tts_via_api_key(text, api_key, language_code="en-US", voice_name="en-US-Wavenet-D", speaking_rate=1.0, volume_gain_db=0.0):
    """Generates TTS audio using Google Cloud TTS API."""
    if not text: print("TTS Warning: Empty text."); return None
    if not api_key: print("TTS Error: Google API Key missing."); gr.Warning("Google API Key missing."); return None
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
    payload = {"input": {"text": text}, "voice": {"languageCode": language_code, "name": voice_name}, "audioConfig": {"audioEncoding": "MP3", "speakingRate": speaking_rate, "volumeGainDb": volume_gain_db}}
    try:
        print(f"Requesting Google TTS: '{text[:50]}...' (Voice: {voice_name}, Rate: {speaking_rate})")
        with requests.post(url, json=payload, timeout=30, stream=False) as resp:
            resp.raise_for_status()
            response_json = resp.json()
        audio_content_base64 = response_json.get("audioContent")
        if not audio_content_base64: print(f"TTS Error: No audioContent. Resp: {response_json}"); gr.Warning("Google TTS API Error."); return None
        audio_bytes = base64.b64decode(audio_content_base64)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            output_path = tmp_file.name; tmp_file.write(audio_bytes)
        print(f"Wrote Google TTS audio to {output_path}")
        return output_path
    except requests.exceptions.RequestException as e:
        print(f"Google TTS API Request Error: {e}")
        status_code = e.response.status_code if e.response is not None else "N/A"
        if status_code == 403: gr.Error("Google TTS Error: Access Denied (403). Check API key/billing.")
        elif status_code == 400: gr.Warning("Google TTS Error: Invalid Request (400).")
        else: gr.Warning(f"Google TTS request failed (Status: {status_code}): {e}")
        traceback.print_exc(); return None
    except Exception as e:
        print(f"Google TTS Processing Error: {e}"); gr.Warning(f"Google TTS Error: {e}"); traceback.print_exc(); return None

# --- VAD Constants ---
VAD_SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_DURATION_MS = 700
VAD_SPEECH_PAD_MS = 300

# =============================================================================
# Gradio UI Definition & Logic
# =============================================================================

with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as app:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown("AI assistant listens automatically when VAD is enabled. Speak clearly.")

    # --- Display Warnings ---
    if missing_keys: gr.Markdown(f"**⚠️ Warning:** API keys missing: `{', '.join(missing_keys)}`. Set env vars and restart.")
    if not VAD_ENABLED: gr.Markdown("**⚠️ Warning:** Silero VAD failed to load. Mic input disabled.")
    if not TRANSCRIPTION_AVAILABLE: gr.Markdown("**⚠️ Warning:** Transcription function failed. Mic input may fail.")

    # --- UI Components ---
    status_display = gr.Textbox("Status: Initializing...", label="Status", interactive=False)
    chatbot_interface = gr.Chatbot(label="Conversation", height=500, bubble_full_width=False)
    audio_output_chat = gr.Audio(label="AI Response Audio", autoplay=True, interactive=False)

    with gr.Row():
        text_input_chat = gr.Textbox(label="Type Message (Alternative/Backup)", lines=1, placeholder="Use if VAD disabled/mic fails...", scale=4)
        with gr.Column(scale=1, min_width=150): send_text_btn = gr.Button("Send Text", variant="secondary")

    with gr.Accordion("⚙️ Settings & Controls", open=False):
        vad_enabled_toggle = gr.Checkbox(label="Enable Automatic Voice Detection (VAD)", value=VAD_ENABLED, interactive=VAD_ENABLED)
        system_prompt_chat = gr.Textbox(label="System Prompt", value=system_prompt, lines=10)
        google_voice_chat = gr.Dropdown(label="AI Voice (Google TTS)", choices=["en-US-Wavenet-D", "en-US-Wavenet-F", "en-US-Wavenet-A", "en-US-Wavenet-E", "en-US-Neural2-J", "en-US-Neural2-F", "en-GB-Wavenet-B", "en-GB-Wavenet-F", "en-AU-Wavenet-B", "en-AU-Wavenet-C"], value="en-US-Wavenet-D")
        speed_slider_google = gr.Slider(label="AI Voice Speed", minimum=0.5, maximum=1.5, value=1.05, step=0.05)
        clear_btn_chat = gr.Button("Clear Conversation")

    vad_trigger_path = gr.Textbox(label="VAD Trigger", visible=False)  # Hidden trigger

    # --- State Management ---
    def get_initial_conversation_state(prompt): return [{"role": "system", "content": prompt}]
    conversation_state = gr.State(value=get_initial_conversation_state(system_prompt))
    vad_internal_state = gr.State({
        "buffer": [], "is_speaking": False, "silence_start_time": None,
        "vad_iterator": None, "processing_active": False,
        "resampler": None, "resampler_orig_freq": None  # For resampling state
    })

    # --- Audio Input Component ---
    audio_input_stream = gr.Audio(
        label="Microphone Input (Live VAD when enabled)",
        sources=["microphone"], streaming=True, type="numpy",
        interactive=VAD_ENABLED,
        waveform_options=gr.WaveformOptions(waveform_color="#CCCCCC", waveform_progress_color="#CCCCCC"),
        visible=VAD_ENABLED
    )

    # --- Core Logic Functions ---
    def clear_conversation_state(current_system_prompt):
        """Resets the chat history and state (Core Logic)."""
        print("Clearing conversation state.")
        initial_state = get_initial_conversation_state(current_system_prompt)
        return [], initial_state, None, None  # history, state, audio_output, text_output

    def update_system_prompt_state(new_prompt):
        """Updates the system prompt and resets state (Core Logic)."""
        print("Updating system prompt and resetting state.")
        initial_state = get_initial_conversation_state(new_prompt)
        return [], initial_state, None  # history, state, audio_output

    def reset_vad_state_logic(current_vad_state):
        """Resets VAD buffer, flags, and iterator."""
        if VAD_ENABLED and VADIterator:
            current_vad_state["vad_iterator"] = VADIterator(VAD_MODEL, threshold=VAD_THRESHOLD, sampling_rate=VAD_SAMPLE_RATE, min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS, speech_pad_ms=VAD_SPEECH_PAD_MS)
            print("[VAD Iterator Re-initialized/Reset]")
        else:
            current_vad_state["vad_iterator"] = None
        current_vad_state["buffer"] = []
        current_vad_state["is_speaking"] = False
        current_vad_state["silence_start_time"] = None
        current_vad_state["processing_active"] = False
        print("[VAD State Reset (Buffer/Flags)]")
        status = "Status: Listening..." if VAD_ENABLED else "Status: VAD Disabled"
        return current_vad_state, status

    def reset_vad_state(current_vad_state):
        """Wrapper for Gradio state update."""
        new_state, status = reset_vad_state_logic(current_vad_state)
        return new_state, gr.update(value=status)

    def process_speech_segment(audio_segment_path, current_history, current_conv_state):
        """Transcribes audio, gets AI text response. Returns updated history, state."""
        print(f"--- Processing VAD segment: {audio_segment_path} ---")
        user_message = ""; updated_history = current_history; updated_conv_state = current_conv_state
        if not TRANSCRIPTION_AVAILABLE: gr.Error("Transcription function not available."); return updated_history, updated_conv_state
        try:
            _, transcribed_text = preprocess_ref_audio_text(audio_segment_path, "")
            user_message = transcribed_text.strip()
            print(f"Transcription result: '{user_message}'")
            if not user_message: gr.Warning("Transcription produced empty text."); return updated_history, updated_conv_state  # No change if empty
        except Exception as e: gr.Error(f"Error during transcription: {e}"); traceback.print_exc(); return updated_history, updated_conv_state
        finally:
            try:
                if audio_segment_path and os.path.exists(audio_segment_path): os.remove(audio_segment_path); print(f"Cleaned temp file: {audio_segment_path}")
            except Exception as e: print(f"Warning: Failed to delete {audio_segment_path}: {e}")

        current_conv = list(current_conv_state); current_conv.append({"role": "user", "content": user_message})
        updated_history = current_history + [[user_message, None]]
        ai_response_text = generate_nebius_response(current_conv)
        current_conv.append({"role": "assistant", "content": ai_response_text})
        updated_history[-1][1] = ai_response_text
        print("-" * 20)
        return updated_history, current_conv

    def generate_audio_for_response(current_history, google_api_key_val, google_voice_val, speed_val):
        """Generates Google TTS audio for the latest AI response."""
        print("Attempting to generate Google TTS audio...")
        if not current_history: print("  History empty."); return None
        _, last_ai_response = current_history[-1]
        if last_ai_response is None or not last_ai_response.strip(): print("  Last AI response empty."); return None
        audio_file_path = google_tts_via_api_key(text=last_ai_response, api_key=google_api_key_val, voice_name=google_voice_val, speaking_rate=speed_val)
        if audio_file_path: print(f"  Successfully generated Google TTS: {audio_file_path}"); return gr.update(value=audio_file_path, autoplay=True)
        else: print("  Failed to generate Google TTS audio."); return None

    def process_text_input_and_respond(text_input, current_history, conv_state_list):
        """Handles TEXT input, gets AI text response. Returns history, state, clears input."""
        print("-" * 20); user_message = text_input.strip() if text_input else ""
        if not user_message: gr.Warning("Please type a message."); return current_history, conv_state_list, gr.update()
        print(f"Processing text input: '{user_message}'")
        current_conv = list(conv_state_list); current_conv.append({"role": "user", "content": user_message})
        updated_history = current_history + [[user_message, None]]
        ai_response_text = generate_nebius_response(current_conv)
        current_conv.append({"role": "assistant", "content": ai_response_text})
        updated_history[-1][1] = ai_response_text
        print("-" * 20)
        return updated_history, current_conv, gr.update(value="")

    # --- VAD Streaming Callback (Fixed Version) ---
    
    def vad_stream_processor(stream_tuple, vad_state):
        update_status = "Status: Listening..."
        trigger_update = gr.update()
        chunk_size = 512  # Fixed chunk size for VAD at 16000 Hz

        # Initialize vad_state if not already done
        if "vad_buffer" not in vad_state:
            vad_state["vad_buffer"] = torch.tensor([])
        if "buffer" not in vad_state:
            vad_state["buffer"] = []
        if "is_speaking" not in vad_state:
            vad_state["is_speaking"] = False
        if "processing_active" not in vad_state:
            vad_state["processing_active"] = False
        if "vad_iterator" not in vad_state:
            from silero_vad.utils import VADIterator
            vad_state["vad_iterator"] = VADIterator(VAD_MODEL, sampling_rate=VAD_SAMPLE_RATE)

        # Handle stream end
        if stream_tuple is None:
            print("Audio stream ended.")
            if vad_state["vad_buffer"].numel() > 0:
                while vad_state["vad_buffer"].shape[0] >= chunk_size:
                    vad_chunk = vad_state["vad_buffer"][:chunk_size]
                    vad_state["vad_buffer"] = vad_state["vad_buffer"][chunk_size:]
                    process_vad_chunk(vad_chunk, vad_state)
                if vad_state["vad_buffer"].shape[0] > 0:
                    padding = chunk_size - vad_state["vad_buffer"].shape[0]
                    vad_chunk = torch.cat([vad_state["vad_buffer"], torch.zeros(padding)])
                    process_vad_chunk(vad_chunk, vad_state)
                    vad_state["vad_buffer"] = torch.tensor([])
            if vad_state["is_speaking"] and vad_state["buffer"] and not vad_state["processing_active"]:
                process_speech_buffer(vad_state)
                update_status = "Status: Processing residual..."
            else:
                update_status = "Status: Listening..."
            return vad_state, gr.update(value=update_status), trigger_update

        if not VAD_ENABLED or VAD_MODEL is None:
            return vad_state, gr.update(value="Status: VAD Disabled"), trigger_update
        if vad_state["processing_active"]:
            return vad_state, gr.update(value="Status: Processing..."), trigger_update

        # Unpack and prepare audio tensor
        try:
            sample_rate, stream_chunk = stream_tuple
            if not isinstance(stream_chunk, np.ndarray):
                raise ValueError("Invalid audio data")

            audio_tensor = torch.from_numpy(stream_chunk.astype(np.float32))
            if audio_tensor.ndim > 1:
                print(f"[VAD Stream] Converting to mono: {audio_tensor.shape}")
                audio_tensor = torch.mean(audio_tensor, dim=1)  # Convert stereo to mono
            if audio_tensor.ndim != 1:
                raise ValueError(f"Audio tensor is not 1D: {audio_tensor.shape}")

            # Resample to 16000 Hz if necessary
            if sample_rate != VAD_SAMPLE_RATE:
                if "resampler" not in vad_state or vad_state.get("resampler_orig_freq") != sample_rate:
                    print(f"Initializing resampler: {sample_rate}Hz -> {VAD_SAMPLE_RATE}Hz")
                    vad_state["resampler"] = T.Resample(orig_freq=sample_rate, new_freq=VAD_SAMPLE_RATE)
                    vad_state["resampler_orig_freq"] = sample_rate
                audio_chunk = vad_state["resampler"](audio_tensor)
            else:
                audio_chunk = audio_tensor

        except Exception as e:
            print(f"Error preparing audio: {e}")
            traceback.print_exc()
            return vad_state, gr.update(value="Status: Stream Error"), trigger_update

        # Append to vad_buffer
        vad_state["vad_buffer"] = torch.cat([vad_state["vad_buffer"], audio_chunk]) if vad_state["vad_buffer"].numel() > 0 else audio_chunk

        # Process in 512-sample chunks
        while vad_state["vad_buffer"].shape[0] >= chunk_size:
            vad_chunk = vad_state["vad_buffer"][:chunk_size]
            vad_state["vad_buffer"] = vad_state["vad_buffer"][chunk_size:]
            process_vad_chunk(vad_chunk, vad_state)
        # Process speech buffer if speech has ended
        if not vad_state["is_speaking"] and vad_state["buffer"] and not vad_state["processing_active"]:
            process_speech_buffer(vad_state)
            update_status = "Status: Processing Speech..."

        return vad_state, gr.update(value=update_status), trigger_update
    
    def process_vad_chunk(vad_chunk, vad_state):
        if vad_state["vad_iterator"] is None:
            try:
                vad_state["vad_iterator"] = VADIterator(VAD_MODEL, sampling_rate=VAD_SAMPLE_RATE)
            except Exception as e:
                print(f"Failed to initialize vad_iterator: {e}")
                return
        if vad_state["vad_iterator"] is None:
            print("vad_iterator is still None after initialization attempt")
            return
        try:
            speech_dict = vad_state["vad_iterator"](vad_chunk, return_seconds=True)
        except Exception as e:
            print(f"Error during VAD processing: {e}")
            import traceback
            traceback.print_exc()
            if vad_state["vad_iterator"] is not None:
                vad_state["vad_iterator"].reset_states()

    def process_speech_buffer(vad_state):
        vad_state["processing_active"] = True
        try:
            full_speech_tensor = torch.cat(vad_state["buffer"])
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                sf.write(tmp_f.name, full_speech_tensor.numpy(), VAD_SAMPLE_RATE)
                return gr.update(value=tmp_f.name)
        except Exception as e:
            print(f"Error saving speech buffer: {e}")
        finally:
            vad_state["processing_active"] = False
            vad_state["buffer"] = []
    # --- Event Handler Wrappers ---
    def clear_conversation_and_vad(current_system_prompt, current_vad_state):
        hist, conv_state, audio_out, text_out = clear_conversation_state(current_system_prompt)
        new_vad_state, status_update = reset_vad_state(current_vad_state)
        return hist, conv_state, audio_out, text_out, new_vad_state, status_update

    def update_system_prompt_and_vad(new_prompt, current_vad_state):
        hist, conv_state, audio_out = update_system_prompt_state(new_prompt)
        new_vad_state, status_update = reset_vad_state(current_vad_state)
        return hist, conv_state, audio_out, new_vad_state, status_update

    def handle_vad_trigger(audio_segment_path, current_history, current_conv_state, current_vad_state):
        """Called when VAD trigger path changes. Processes speech, updates state."""
        if not audio_segment_path: return current_history, current_conv_state, current_vad_state, gr.update()
        print(f"[Trigger] Processing path: {audio_segment_path}")
        updated_history, updated_conv_state = process_speech_segment(audio_segment_path, current_history, current_conv_state)
        current_vad_state["processing_active"] = False
        status_update = "Status: Generating AI Response..."
        return updated_history, updated_conv_state, current_vad_state, gr.update(value=status_update)

    def text_input_pipeline(text_input, current_history, current_conv_state):
        """Wrapper for text input processing."""
        return process_text_input_and_respond(text_input, current_history, current_conv_state)

    def toggle_vad(vad_choice, current_vad_state):
        """Handles VAD checkbox change."""
        if vad_choice and VAD_ENABLED:
            print("VAD Enabled by user.")
            new_state, status = reset_vad_state_logic(current_vad_state)
            return new_state, gr.update(interactive=True, visible=True), gr.update(value=status)
        else:
            print("VAD Disabled by user or not available.")
            new_state, status = reset_vad_state_logic(current_vad_state)
            status = "Status: VAD Disabled"
            return new_state, gr.update(interactive=False, visible=VAD_ENABLED), gr.update(value=status)

    # --- Event Handling Wiring ---
    app.load(lambda: "Status: Ready (VAD Enabled)" if VAD_ENABLED else "Status: Ready (VAD Disabled)", None, status_display)

    vad_enabled_toggle.change(
        toggle_vad,
        inputs=[vad_enabled_toggle, vad_internal_state],
        outputs=[vad_internal_state, audio_input_stream, status_display],
        queue=False
    )

    if VAD_ENABLED:
        audio_input_stream.stream(
            vad_stream_processor,
            inputs=[audio_input_stream, vad_internal_state],
            outputs=[vad_internal_state, status_display, vad_trigger_path],
        )

        vad_trigger_path.change(
            handle_vad_trigger,
            inputs=[vad_trigger_path, chatbot_interface, conversation_state, vad_internal_state],
            outputs=[chatbot_interface, conversation_state, vad_internal_state, status_display]
        ).then(
            generate_audio_for_response,
            inputs=[chatbot_interface, gr.State(GOOGLE_API_KEY), google_voice_chat, speed_slider_google],
            outputs=[audio_output_chat]
        ).then(
            reset_vad_state,
            inputs=[vad_internal_state],
            outputs=[vad_internal_state, status_display]
        )

    send_text_btn.click(
        text_input_pipeline,
        inputs=[text_input_chat, chatbot_interface, conversation_state],
        outputs=[chatbot_interface, conversation_state, text_input_chat],
        show_progress="minimal"
    ).then(
        generate_audio_for_response,
        inputs=[chatbot_interface, gr.State(GOOGLE_API_KEY), google_voice_chat, speed_slider_google],
        outputs=[audio_output_chat]
    )

    clear_btn_chat.click(
        clear_conversation_and_vad,
        inputs=[system_prompt_chat, vad_internal_state],
        outputs=[chatbot_interface, conversation_state, audio_output_chat, text_input_chat, vad_internal_state, status_display],
        queue=False
    )

    system_prompt_chat.change(
        update_system_prompt_and_vad,
        inputs=[system_prompt_chat, vad_internal_state],
        outputs=[chatbot_interface, conversation_state, audio_output_chat, vad_internal_state, status_display],
    )

# =============================================================================
# App Launch
# =============================================================================

if __name__ == "__main__":
    if not VAD_ENABLED: print("\nWARNING: Silero VAD not loaded. Microphone input disabled.\n")
    if not TRANSCRIPTION_AVAILABLE: print("\nWARNING: Transcription function not loaded. Microphone input may fail.\n")
    print("Starting Gradio app...")
    app.queue().launch(server_name="0.0.0.0", debug=True)