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
import time
import soundfile as sf # For saving buffered audio

# --- Silero VAD Setup ---
VAD_ENABLED = False
VAD_MODEL = None
VAD_UTILS = None
try:
    # Check if torch version is compatible (example check, adjust as needed)
    # Silero VAD might have specific requirements
    print(f"PyTorch version: {torch.__version__}")
    
    # Attempt to load Silero VAD model
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False, # Set to True if you want to re-download
        onnx=False # Set to True if you installed 'onnxruntime' and prefer ONNX
    )
    VAD_MODEL = vad_model
    VAD_UTILS = utils
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    VAD_ENABLED = True
    print("Silero VAD model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load Silero VAD model: {e}")
    print("VAD and automatic speech detection will be disabled.")
    print("Ensure PyTorch is installed correctly and you have internet access for download.")
    VAD_ENABLED = False

# --- Configuration & API Key Handling ---
# (Keep the API Key loading logic as before)
os.environ["NEBIUS_API_KEY"] = "eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExNzk1NTg1MzgyNzYzMDYxNDQwMCIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwMjgwMzYzNSwidXVpZCI6IjE4NjRlMWM4LTM1NmQtNDY0Ny04NTdlLTJlN2UxYzg2ODJkYSIsIm5hbWUiOiJzYWhhcmY1IiwiZXhwaXJlc19hdCI6IjIwMzAtMDQtMTlUMDQ6MzM6NTUrMDAwMCJ9.632FHIOBhPM3iNlIBBxPC-16uc3FKoiAC1iqVKXUO5g"
os.environ["GOOGLE_API_KEY"] = "AIzaSyC2MC-S458QlwKKa1EFyZYH-HJJkAngSRw"
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
missing_keys = []
if not NEBIUS_API_KEY: missing_keys.append("NEBIUS_API_KEY")
if not GOOGLE_API_KEY: missing_keys.append("GOOGLE_API_KEY")
# ... (rest of API key checking code) ...
if missing_keys:
    warning_message = (
        "\n" + "="*40 + "\n"
        "!!! WARNING: Required API Key(s) Missing !!!\n"
        f"The following environment variables are not set: {', '.join(missing_keys)}\n"
        "The application requires these keys to function correctly.\n"
        "Please set them before running the script.\n"
        "="*40 + "\n"
    )
    print(warning_message)

# --- Nebius API Client Initialization ---
# ... (keep Nebius client init code) ...
nebius_client = None
if NEBIUS_API_KEY:
    try:
        nebius_client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=NEBIUS_API_KEY
        )
        print("Nebius API client initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Nebius API client. Error: {str(e)}")
        nebius_client = None
else:
    print("ERROR: Nebius API client cannot be initialized because NEBIUS_API_KEY is not set.")


# --- Transcription Function Import ---
TRANSCRIPTION_AVAILABLE = False
try:
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    TRANSCRIPTION_AVAILABLE = True
    print("Audio transcription function loaded successfully.")
except ImportError as e:
    print(f"Warning: Failed to import 'preprocess_ref_audio_text' ({e}). Transcription might fail.")
    def preprocess_ref_audio_text(audio_path, text, **kwargs):
        raise RuntimeError("Audio transcription function 'preprocess_ref_audio_text' is not available.")

# --- App Title & System Prompt ---
# ... (keep APP_TITLE and system_prompt) ...
APP_TITLE = "Locksmith AI Assistant (VAD Enabled)"
system_prompt = """You are Joe, an AI assistant for 1st Choice Locksmith Services... [Your original prompt] ..."""


# --- Core API Functions ---
# ... (keep generate_nebius_response and google_tts_via_api_key functions) ...
def generate_nebius_response(messages):
    """Generate response using Nebius API"""
    if not nebius_client:
        gr.Error("Nebius API client is not initialized. Cannot generate response.")
        print("Attempted to generate Nebius response, but client is not available.")
        return "I'm sorry, the chat service is currently unavailable due to a configuration issue."
    try:
        formatted_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
        print(f"Sending {len(formatted_messages)} messages to Nebius API.")
        response = nebius_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
            max_tokens=512, temperature=0.6, top_p=0.9,
            messages=formatted_messages
        )
        ai_response = response.choices[0].message.content
        print(f"Received response from Nebius API: '{ai_response[:100]}...'")
        return ai_response
    except Exception as e:
        # ... (error handling as before) ...
        print(f"ERROR generating Nebius response: {str(e)}")
        traceback.print_exc()
        error_message = f"Error generating response: {str(e)}"
        if "authentication" in str(e).lower():
             gr.Warning("Nebius API authentication failed. Please check the NEBIUS_API_KEY.")
             error_message = "I'm sorry, there was an authentication issue with the chat service."
        else:
             gr.Warning(error_message)
             error_message = "I'm sorry, I encountered an error while processing your request."
        return error_message

def google_tts_via_api_key(text, api_key, language_code="en-US", voice_name="en-US-Wavenet-D", speaking_rate=1.0, volume_gain_db=0.0):
    """Generates TTS audio using Google Cloud TTS API and saves to a unique temp MP3 file."""
    # ... (keep existing function body) ...
    if not text: print("TTS Warning: Received empty text."); return None
    if not api_key: print("TTS Error: Google API Key is missing."); gr.Warning("Google API Key missing."); return None

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
    payload = { "input": {"text": text}, "voice": {"languageCode": language_code, "name": voice_name}, "audioConfig": {"audioEncoding": "MP3", "speakingRate": speaking_rate, "volumeGainDb": volume_gain_db}}
    try:
        print(f"Requesting Google TTS for text: '{text[:50]}...' (Voice: {voice_name}, Rate: {speaking_rate})")
        with requests.post(url, json=payload, timeout=30, stream=False) as resp:
            resp.raise_for_status()
            response_json = resp.json()
        audio_content_base64 = response_json.get("audioContent")
        if not audio_content_base64: print(f"TTS Error: No audioContent. Resp: {response_json}"); gr.Warning("Google TTS API Error."); return None
        audio_bytes = base64.b64decode(audio_content_base64)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            output_path = tmp_file.name
            tmp_file.write(audio_bytes)
        print(f"Wrote Google TTS audio ({len(audio_bytes)} bytes) to {output_path}")
        return output_path
    except requests.exceptions.RequestException as e:
        # ... (keep detailed error handling) ...
        print(f"Google TTS API Request Error: {e}")
        status_code = e.response.status_code if e.response is not None else "N/A"
        error_text = f"Google TTS request failed (Status: {status_code}): {e}"
        if status_code == 403: gr.Error("Google TTS Error: Access Denied (403). Check API key/billing.")
        elif status_code == 400: gr.Warning("Google TTS Error: Invalid Request (400).")
        else: gr.Warning(error_text)
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Google TTS Processing Error: {e}"); gr.Warning(f"Google TTS Error: {e}"); traceback.print_exc(); return None

# --- VAD Constants ---
VAD_SAMPLE_RATE = 16000  # Silero VAD expects 16kHz
VAD_WINDOW_SIZE = 512  # Example chunk size for VAD processing (adjust based on performance)
VAD_SPEECH_PAD_MS = 250 # Add padding before/after speech detection
VAD_THRESHOLD = 0.5    # VAD confidence threshold
VAD_MIN_SILENCE_DURATION_MS = 500 # How long silence must be to trigger end of speech


# =============================================================================
# Gradio UI Definition & Logic
# =============================================================================

with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as app:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown("AI assistant will listen automatically. Start speaking clearly. Ensure API keys are set.")

    if missing_keys:
        gr.Markdown(f"**⚠️ Warning:** API keys missing: `{', '.join(missing_keys)}`. Set environment variables and restart.")

    # --- UI Components ---
    status_display = gr.Textbox("Status: Initializing...", label="Status", interactive=False)
    chatbot_interface = gr.Chatbot(label="Conversation", height=500, bubble_full_width=False)
    audio_output_chat = gr.Audio(label="AI Response Audio", autoplay=True, interactive=False)

    with gr.Row():
         text_input_chat = gr.Textbox(
              label="Type Message (Alternative)", lines=1, placeholder="Or type here if microphone/VAD disabled...", scale=4
         )
         with gr.Column(scale=1, min_width=150):
              send_text_btn = gr.Button("Send Text", variant="secondary") # Button only for text input

    with gr.Accordion("⚙️ Settings & Controls", open=False):
        # VAD Enable/Disable Toggle
        vad_enabled_toggle = gr.Checkbox(label="Enable Automatic Voice Detection (VAD)", value=VAD_ENABLED, interactive=VAD_ENABLED)
        
        system_prompt_chat = gr.Textbox(label="System Prompt", value=system_prompt, lines=10)
        google_voice_chat = gr.Dropdown(
            label="AI Voice (Google TTS)",
            choices=["en-US-Wavenet-D", "en-US-Wavenet-F", "en-US-Wavenet-A", "en-US-Wavenet-E", "en-US-Neural2-J", "en-US-Neural2-F", "en-GB-Wavenet-B", "en-GB-Wavenet-F", "en-AU-Wavenet-B", "en-AU-Wavenet-C"],
            value="en-US-Wavenet-D",
            info="Select the Google TTS voice."
        )
        speed_slider_google = gr.Slider(label="AI Voice Speed", minimum=0.5, maximum=1.5, value=1.05, step=0.05)
        clear_btn_chat = gr.Button("Clear Conversation")

    # Hidden component to trigger processing after VAD detects end of speech
    vad_trigger_path = gr.Textbox(label="VAD Trigger", visible=False)

    # --- State Management ---
    def get_initial_conversation_state(prompt):
        return [{"role": "system", "content": prompt}]

    conversation_state = gr.State(value=get_initial_conversation_state(system_prompt))
    
    # State for VAD buffering
    # Using a dictionary to hold mutable VAD state easily within Gradio's immutable State
    vad_internal_state = gr.State({
        "buffer": [],             # Holds audio chunks (tensors) during speech
        "is_speaking": False,     # Flag if user is currently speaking
        "silence_start_time": None, # Timestamp when silence began after speech
        "vad_iterator": None,     # VAD iterator object
        "processing_active": False # Flag to prevent processing overlapping triggers
    })

    # --- Audio Input Component (Streaming) ---
    # Note: Use streaming=True, sources=['microphone']. type='numpy' might be easier for VAD.
    # Sample rate must match VAD_SAMPLE_RATE. `buffer_length` affects latency.
    audio_input_stream = gr.Audio(
        label="Microphone Input (Live VAD)",
        sources=["microphone"],
        streaming=True,
        type="numpy", # Get numpy array directly
        sample_rate=VAD_SAMPLE_RATE,
        # buffer_length=0.1, # Adjust buffer length (latency vs chunk size) - might need tuning
        waveform_options=gr.WaveformOptions(
             waveform_color="#0176dd", waveform_progress_color="#0066ba", # Optional styling
             show_duration=True
        ),
        # Only interactive if VAD was loaded and toggle is on
        interactive=VAD_ENABLED 
    )

    # --- Core Logic Functions ---

    # Function to reset the VAD state (e.g., after processing or on clear)
    def reset_vad_state_logic(current_vad_state):
        if VAD_ENABLED and VAD_UTILS:
             # Re-initialize VADIterator if needed, or just reset flags/buffer
             current_vad_state["vad_iterator"] = VADIterator(VAD_MODEL, threshold=VAD_THRESHOLD, sampling_rate=VAD_SAMPLE_RATE, min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS, speech_pad_ms=VAD_SPEECH_PAD_MS)
        current_vad_state["buffer"] = []
        current_vad_state["is_speaking"] = False
        current_vad_state["silence_start_time"] = None
        current_vad_state["processing_active"] = False
        print("[VAD State Reset]")
        return current_vad_state, "Status: Listening..." # Return state dict and status update

    # Wrapper to use with Gradio State update
    def reset_vad_state(current_vad_state):
        new_state, status = reset_vad_state_logic(current_vad_state)
        return new_state, gr.update(value=status)

    def process_speech_segment(audio_segment_path, current_history, current_conv_state):
        """Transcribes audio, gets AI response text. Returns history, state."""
        print(f"--- Processing VAD segment: {audio_segment_path} ---")
        user_message = ""
        # 1. Transcribe the saved audio segment
        if not TRANSCRIPTION_AVAILABLE:
            gr.Error("Transcription is not available.")
            return current_history, current_conv_state # Return state unchanged
        try:
            _, transcribed_text = preprocess_ref_audio_text(audio_segment_path, "")
            user_message = transcribed_text.strip()
            print(f"Transcription result: '{user_message}'")
            if not user_message:
                gr.Warning("Transcription failed or produced empty text.")
                return current_history, current_conv_state # Return state unchanged
        except Exception as e:
            gr.Error(f"Error during transcription: {e}")
            traceback.print_exc()
            return current_history, current_conv_state # Return state unchanged
        finally:
             # Clean up the temporary audio file
             try:
                  if os.path.exists(audio_segment_path):
                       os.remove(audio_segment_path)
                       print(f"Cleaned up temp file: {audio_segment_path}")
             except Exception as e:
                  print(f"Warning: Could not delete temp file {audio_segment_path}: {e}")


        # 2. Update conversation state & history (UI)
        current_conv = list(current_conv_state) # Ensure mutable
        current_conv.append({"role": "user", "content": user_message})
        updated_history = current_history + [[user_message, None]] # Show user message

        # 3. Generate AI response (text)
        ai_response_text = generate_nebius_response(current_conv)

        # 4. Update state & history (UI) with AI response
        current_conv.append({"role": "assistant", "content": ai_response_text})
        updated_history[-1][1] = ai_response_text # Add AI response to chatbot display

        print("-" * 20)
        return updated_history, current_conv

    def generate_audio_for_response(current_history, google_api_key_val, google_voice_val, speed_val):
        """Generates Google TTS audio for the latest AI response."""
        # ... (identical to previous version, just called differently) ...
        print("Attempting to generate Google TTS audio...")
        if not current_history: print("  History empty, skipping audio generation."); return None
        _, last_ai_response = current_history[-1]
        if last_ai_response is None or not last_ai_response.strip(): print("  Last AI response empty, skipping audio."); return None
        
        audio_file_path = google_tts_via_api_key(text=last_ai_response, api_key=google_api_key_val, voice_name=google_voice_val, speaking_rate=speed_val)
        
        if audio_file_path: print(f"  Successfully generated Google TTS audio: {audio_file_path}"); return gr.update(value=audio_file_path, autoplay=True)
        else: print("  Failed to generate Google TTS audio."); return None

    def process_text_input_and_respond(text_input, current_history, conv_state_list):
         """Handles TEXT input, gets AI text response. Returns history, state."""
         print("-" * 20)
         user_message = text_input.strip() if text_input else ""
         if not user_message:
              gr.Warning("Please type a message.")
              return current_history, conv_state_list, gr.update() # No change

         print(f"Processing user text input: '{user_message}'")
         # Update state/history
         current_conv = list(conv_state_list)
         current_conv.append({"role": "user", "content": user_message})
         updated_history = current_history + [[user_message, None]]

         # Get AI response
         ai_response_text = generate_nebius_response(current_conv)

         # Update state/history with AI response
         current_conv.append({"role": "assistant", "content": ai_response_text})
         updated_history[-1][1] = ai_response_text

         print("-" * 20)
         return updated_history, current_conv, gr.update(value="") # Clear input


    # --- VAD Streaming Callback ---
    def vad_stream_processor(
        stream, # The numpy audio chunk (or None if stream ends)
        current_vad_state # The VAD state dictionary from gr.State
        ):
        
        if not VAD_ENABLED or VAD_MODEL is None or VAD_UTILS is None:
             # Should not happen if interactive=False is set correctly, but good safety check
             return current_vad_state, gr.update(value="Status: VAD Disabled") , gr.update()

        if stream is None:
             print("Audio stream ended.")
             # Potentially process any remaining buffer if needed, then reset
             new_state, status = reset_vad_state_logic(current_vad_state)
             return new_state, gr.update(value=status), gr.update() # Reset trigger

        # Ensure audio is in the correct format (numpy array, float32, correct sample rate)
        # Gradio numpy output is usually float64, VAD might need float32
        if stream.dtype != np.float32:
             audio_float32 = stream.astype(np.float32)
        else:
             audio_float32 = stream
        
        # Convert numpy array to torch tensor for VAD
        audio_chunk_tensor = torch.from_numpy(audio_float32)

        if current_vad_state.get("vad_iterator") is None:
             # Initialize VAD iterator if not present (e.g., on first run or after reset)
             current_vad_state["vad_iterator"] = VADIterator(VAD_MODEL, threshold=VAD_THRESHOLD, sampling_rate=VAD_SAMPLE_RATE, min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS, speech_pad_ms=VAD_SPEECH_PAD_MS)
             print("[VAD Iterator Initialized]")
        
        try:
             speech_dict = current_vad_state["vad_iterator"](audio_chunk_tensor, return_seconds=True)
        except Exception as e:
             print(f"Error during VAD processing: {e}")
             # Attempt to reset iterator on error?
             current_vad_state["vad_iterator"].reset_states()
             return current_vad_state, gr.update(value="Status: VAD Error"), gr.update()

        new_status = "Status: Listening..." # Default status
        trigger_update = gr.update() # Default: no trigger update

        if speech_dict: # If VAD found speech start or end in this chunk
            if "start" in speech_dict:
                if not current_vad_state["is_speaking"]:
                    print(f"[VAD] Speech Start detected at {speech_dict['start']:.2f}s")
                    current_vad_state["is_speaking"] = True
                    current_vad_state["silence_start_time"] = None # Reset silence timer
                    # Add chunk to buffer (potentially with padding adjustment later)
                    current_vad_state["buffer"].append(audio_chunk_tensor)
                    new_status = "Status: User Speaking..."
                else:
                     # Still speaking, just add chunk
                     current_vad_state["buffer"].append(audio_chunk_tensor)

            elif "end" in speech_dict:
                if current_vad_state["is_speaking"]:
                     print(f"[VAD] Speech End detected at {speech_dict['end']:.2f}s")
                     # Add the final chunk containing the end
                     current_vad_state["buffer"].append(audio_chunk_tensor)
                     
                     # Check if processing isn't already active from a previous trigger
                     if not current_vad_state["processing_active"]:
                         current_vad_state["processing_active"] = True # Set flag
                         print("[VAD] Triggering processing...")
                         new_status = "Status: Processing Speech..."
                         
                         # --- Process Buffered Audio ---
                         try:
                              # Concatenate buffered tensors
                              if current_vad_state["buffer"]:
                                   full_speech_tensor = torch.cat(current_vad_state["buffer"])
                                   
                                   # Save the full tensor to a temporary WAV file
                                   with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                                        wav_path = tmp_file.name
                                        # Use soundfile to save tensor (convert back to numpy)
                                        sf.write(wav_path, full_speech_tensor.numpy(), VAD_SAMPLE_RATE)
                                        print(f"Saved speech segment to {wav_path} (Duration: {len(full_speech_tensor)/VAD_SAMPLE_RATE:.2f}s)")
                                        # Update the hidden textbox to trigger the processing chain
                                        trigger_update = gr.update(value=wav_path)
                              else:
                                   print("[VAD] Buffer was empty, nothing to process.")
                                   # Reset processing flag if buffer empty
                                   current_vad_state["processing_active"] = False 
                                   new_status = "Status: Listening..."

                         except Exception as e:
                              print(f"ERROR saving/processing buffered audio: {e}")
                              traceback.print_exc()
                              # Reset processing flag on error
                              current_vad_state["processing_active"] = False 
                              new_status = "Status: Error processing audio"
                         # --- End Process Buffered Audio ---

                     # Reset speaking flag and buffer regardless of processing success/failure for next utterance
                     current_vad_state["is_speaking"] = False
                     current_vad_state["buffer"] = []
                     current_vad_state["silence_start_time"] = time.time() # Mark when silence truly began

                else:
                     # VAD triggered 'end' but we weren't marked as 'speaking' (should be rare)
                     pass # Ignore, likely just noise trigger

        elif current_vad_state["is_speaking"]:
             # No speech detected in this chunk, but we were previously speaking
             # Keep adding to buffer (might be brief pause within speech)
             current_vad_state["buffer"].append(audio_chunk_tensor)
             new_status = "Status: User Speaking... (Brief Pause?)"


        # Return updated state and status. Trigger update might contain a file path.
        return current_vad_state, gr.update(value=new_status), trigger_update

    # --- Event Handling Wiring ---

    # When VAD is enabled/disabled via checkbox
    def toggle_vad(vad_choice, current_vad_state):
        global VAD_ENABLED # Allow modification of global flag based on UI? Risky. Better to pass state.
        if vad_choice and VAD_MODEL: # Check if VAD model loaded successfully too
             print("VAD Enabled by user.")
             # Reset state when enabling
             new_state, status = reset_vad_state_logic(current_vad_state)
             # Make microphone interactive
             return new_state, gr.update(interactive=True), gr.update(value=status)
        else:
             print("VAD Disabled by user or not available.")
             # Reset state and make microphone non-interactive
             new_state, status = reset_vad_state_logic(current_vad_state)
             status = "Status: VAD Disabled" # Overwrite status
             return new_state, gr.update(interactive=False), gr.update(value=status)

    vad_enabled_toggle.change(
        toggle_vad,
        inputs=[vad_enabled_toggle, vad_internal_state],
        outputs=[vad_internal_state, audio_input_stream, status_display] # Update state, mic interactivity, status
    )

    # When the audio input stream receives data (if VAD enabled)
    audio_input_stream.stream(
        vad_stream_processor,
        inputs=[audio_input_stream, vad_internal_state],
        outputs=[vad_internal_state, status_display, vad_trigger_path], # Update state, status, and potentially the hidden trigger
    )

    # When the hidden trigger textbox changes (i.e., VAD saved a file)
    # This acts as the link between the streaming callback and the main processing logic
    def handle_vad_trigger(
         audio_segment_path, current_history, current_conv_state, current_vad_state
         ):
        if not audio_segment_path: # Ignore empty updates
            return current_history, current_conv_state, current_vad_state, gr.update() # No audio update

        print(f"[Trigger] Processing path: {audio_segment_path}")
        
        # Call the function that handles transcription and AI response generation
        updated_history, updated_conv_state = process_speech_segment(
            audio_segment_path, current_history, current_conv_state
        )
        
        # Generate audio for the AI response (this happens *after* process_speech_segment)
        # Note: This relies on generate_audio_for_response being called separately
        # We need to trigger that next. This function only returns the state needed.
        
        # We don't reset the VAD state here, that should happen after AI speaks ideally,
        # but for simplicity, we reset the 'processing_active' flag now.
        # A better approach might involve JS or estimating audio duration.
        current_vad_state["processing_active"] = False 
        
        status_update = "Status: Generating AI Response..." # Update status
        
        # Return updated history/state to be used by the next step (.then)
        # Also return the vad_state with processing flag reset
        return updated_history, updated_conv_state, current_vad_state, gr.update(value=status_update)


    # Chain after vad_trigger_path changes: Process Segment -> Generate Audio -> Reset VAD
    vad_trigger_path.change(
         handle_vad_trigger,
         inputs=[vad_trigger_path, chatbot_interface, conversation_state, vad_internal_state],
         outputs=[chatbot_interface, conversation_state, vad_internal_state, status_display] # Update chat, state, vad_state, status
    ).then(
         # Now generate the audio for the response that was just added to history
         generate_audio_for_response,
         inputs=[chatbot_interface, gr.State(GOOGLE_API_KEY), google_voice_chat, speed_slider_google],
         outputs=[audio_output_chat]
    ).then(
         # After attempting audio generation, reset VAD state to listen again
         # This is imperfect timing but the simplest approach here.
         reset_vad_state,
         inputs=[vad_internal_state],
         outputs=[vad_internal_state, status_display]
    )


    # Event handler for the TEXT input button
    def text_input_pipeline(text_input, current_history, current_conv_state):
         # 1. Process text input to get AI text response
         updated_history, updated_conv_state, text_clear_update = process_text_input_and_respond(
              text_input, current_history, current_conv_state
         )
         # Return immediately, audio generation triggered by .then()
         return updated_history, updated_conv_state, text_clear_update

    send_text_btn.click(
         text_input_pipeline,
         inputs=[text_input_chat, chatbot_interface, conversation_state],
         outputs=[chatbot_interface, conversation_state, text_input_chat],
         show_progress="minimal"
    ).then(
         # 2. Generate audio for the response added by text_input_pipeline
         generate_audio_for_response,
         inputs=[chatbot_interface, gr.State(GOOGLE_API_KEY), google_voice_chat, speed_slider_google],
         outputs=[audio_output_chat]
         # No need to reset VAD state after text input
    )


    # Clear button clicked - also resets VAD state
    def clear_conversation_and_vad(current_system_prompt, current_vad_state):
        # Clear chat history/state
        hist, conv_state, audio_out, text_out = clear_conversation_state(current_system_prompt)
        # Reset VAD state
        new_vad_state, status_update = reset_vad_state(current_vad_state)
        return hist, conv_state, audio_out, text_out, new_vad_state, status_update

    clear_btn_chat.click(
        clear_conversation_and_vad,
        inputs=[system_prompt_chat, vad_internal_state],
        outputs=[chatbot_interface, conversation_state, audio_output_chat, text_input_chat, vad_internal_state, status_display],
        queue=False
    )

    # System prompt changed - also resets VAD state
    def update_system_prompt_and_vad(new_prompt, current_vad_state):
         hist, conv_state, audio_out = update_system_prompt_state(new_prompt)
         new_vad_state, status_update = reset_vad_state(current_vad_state)
         return hist, conv_state, audio_out, new_vad_state, status_update

    system_prompt_chat.change(
        update_system_prompt_and_vad,
        inputs=[system_prompt_chat, vad_internal_state],
        outputs=[chatbot_interface, conversation_state, audio_output_chat, vad_internal_state, status_display],
    )

    # Set initial status on load
    app.load(lambda: "Status: Ready (VAD Enabled)" if VAD_ENABLED else "Status: Ready (VAD Disabled)", None, status_display)


# =============================================================================
# App Launch
# =============================================================================

if __name__ == "__main__":
    # Ensure VAD model is loaded before launch if possible
    if not VAD_ENABLED:
         print("\nWARNING: Silero VAD failed to load. Automatic speech detection disabled.\n")
    
    print("Starting Gradio app...")
    app.queue().launch(server_name="0.0.0.0", debug=True)