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
import soundfile as sf
import base64
import requests
import os
# --- Configuration & API Key Handling ---
os.environ["NEBIUS_API_KEY"] = "eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExNzk1NTg1MzgyNzYzMDYxNDQwMCIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwMjgwMzYzNSwidXVpZCI6IjE4NjRlMWM4LTM1NmQtNDY0Ny04NTdlLTJlN2UxYzg2ODJkYSIsIm5hbWUiOiJzYWhhcmY1IiwiZXhwaXJlc19hdCI6IjIwMzAtMDQtMTlUMDQ6MzM6NTUrMDAwMCJ9.632FHIOBhPM3iNlIBBxPC-16uc3FKoiAC1iqVKXUO5g"
os.environ["GOOGLE_API_KEY"] = "AIzaSyC2MC-S458QlwKKa1EFyZYH-HJJkAngSRw"
# Attempt to get API keys from environment variables
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Check if keys are actually set
missing_keys = []
if not NEBIUS_API_KEY:
    missing_keys.append("NEBIUS_API_KEY")
if not GOOGLE_API_KEY:
    missing_keys.append("GOOGLE_API_KEY")

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
    # Optionally, exit if keys are absolutely critical for any operation
    # exit(1)

# --- Nebius API Client Initialization ---
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
        nebius_client = None # Ensure client is None if init fails
else:
    print("ERROR: Nebius API client cannot be initialized because NEBIUS_API_KEY is not set.")

# --- Transcription Function (Import from F5-TTS utils) ---
# Attempt to import the transcription utility.
# This assumes the f5_tts library structure is available,
# but doesn't load the actual F5 TTS models.
TRANSCRIPTION_AVAILABLE = False
try:
    # Adjust the import path based on your actual f5_tts library structure
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    TRANSCRIPTION_AVAILABLE = True
    print("Audio transcription function loaded successfully.")
except ImportError as e:
    print(f"Warning: Failed to import 'preprocess_ref_audio_text' ({e}). Audio input via microphone will not work.")
    # Define a dummy function if import fails, so the app doesn't crash
    def preprocess_ref_audio_text(audio_path, text, **kwargs):
        print("ERROR: Transcription unavailable. Returning empty text.")
        # Returns format expected by calling code: (None, "") -> (audio_data, text)
        # We only care about the text part here for transcription.
        raise RuntimeError("Audio transcription component is not available.") # Raise error to signal issue

# --- App Title & System Prompt ---
APP_TITLE = "Locksmith AI Assistant"

# System prompt defining the AI's role
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

**Example Flow:**
Me: "Hello, this is Joe from 1st Choice Locksmith Services. How can I help you with your lock or key issue today?"
Customer: "I'm locked out of my house."
Me: "I understand, that's frustrating. Can you please provide your full name?"
Customer: [Provides Name]
Me: "Thank you. Can I get the full service address, including city and ZIP code?"
Customer: [Provides Address]
Me: "Got it. And what's the best phone number to reach you?"
Customer: [Provides Phone]
Me: "Okay, just to quickly confirm: Your name is [Name], the address is [Address with City/Zip], phone number is [Phone], and the issue is a house lockout. Is that all correct?"
Customer: "Yes."
Me: "Excellent. Is there anything specific the technician should know when they arrive?"
Customer: [Optional Notes or "No"]
Me: "Thank you, [Name]. A technician from 1st Choice Locksmith Services will call you within 1-2 minutes. Help is on the way!"
"""

# Define system prompts for each language
system_prompts = {
    "English": system_prompt,
"Hebrew": """אתה עוזר AI עבור משרד עורכי הדין זמירה צדוק בנהריה. תחילה, שאל את הלקוח אם הוא גבר או אישה כדי להשתמש בצורות המגדר הנכונות בעברית. לאחר מכן, אסוף את המידע הבא:
1. שם מלא של הלקוח
2. מספר טלפון ליצירת קשר
3. סוג השירות המשפטי הנדרש (למשל, דיני משפחה, דיני עבודה, נדל"ן)
4. תיאור קצר של הבעיה או השאלה המשפטית

**כללי שיחה:**
- היה מקצועי ואדיב.
- שאל שאלה אחת בכל פעם.
- אל תספק ייעוץ משפטי; המטרה היא רק לאסוף מידע.
- אשר את הפרטים שנאספו בסוף השיחה.
- סיים את השיחה באומר: "תודה, [שם]. נציג ממשרד עורכי הדין זמירה צדוק ייצור איתך קשר בהקדם האפשרי."

**דוגמת זרימה:**
אני: "שלום, זה עוזר AI ממשרד עורכי הדין זמירה צדוק. כדי לפנות אליך בצורה המתאימה, האם אתה גבר או אישה?"
לקוח: "אישה."
אני: "תודה. איך אוכל לעזור לך היום?"
לקוח: "אני צריכה עזרה עם גירושין."
אני: "אני מבין. כדי שנוכל להעביר את פנייתך לעורך הדין המתאים, אוכל לקבל את שמך המלא?"
לקוח: [מספקת שם]
אני: "תודה. מהו מספר הטלפון הכי טוב ליצירת קשר איתך?"
לקוח: [מספקת טלפון]
אני: "בנוגע לשירות המשפטי, האם תוכלי לציין את התחום המשפטי הספציפי? למשל, דיני משפחה, דיני עבודה, וכו'."
לקוח: "דיני משפחה, גירושין."
אני: "הבנתי. האם תוכלי לספק תיאור קצר של המצב או השאלה שלך?"
לקוח: [מספקת תיאור]
אני: "תודה. רק כדי לאשר: שמך הוא [שם], מספר הטלפון הוא [טלפון], התחום הוא דיני משפחה - גירושין, והתיאור הוא [תיאור]. הכל נכון?"
לקוח: "כן."
אני: "תודה, [שם]. נציג ממשרד עורכי הדין זמירה צדוק ייצור איתך קשר בהקדם האפשרי."
"""
}

# Voice options for each language
voice_options = {
    "English": [
        "en-US-Wavenet-D", "en-US-Wavenet-F", "en-US-Wavenet-A", "en-US-Wavenet-E",
        "en-US-Neural2-J", "en-US-Neural2-F", "en-GB-Wavenet-B", "en-GB-Wavenet-F",
        "en-AU-Wavenet-B", "en-AU-Wavenet-C",
    ],
    "Hebrew": [
        "he-IL-Wavenet-A", "he-IL-Wavenet-B", "he-IL-Wavenet-C", "he-IL-Wavenet-D"
    ]
}

default_voices = {
    "English": "en-US-Wavenet-D",
    "Hebrew": "he-IL-Wavenet-A"
}

# --- Core API Functions ---
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
            max_tokens=512,
            temperature=0.6,
            top_p=0.9,
            messages=formatted_messages
        )
        ai_response = response.choices[0].message.content
        print(f"Received response from Nebius API: '{ai_response[:100]}...'")
        return ai_response
    except Exception as e:
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

def google_tts_via_api_key(text, api_key, voice_name, speaking_rate=1.0, volume_gain_db=0.0):
    """Generates TTS audio using Google Cloud TTS API and saves to a unique temp MP3 file."""
    if not text:
        print("TTS Warning: Received empty text.")
        return None
    if not api_key:
        print("TTS Error: Google API Key is missing. Cannot generate audio.")
        gr.Warning("Google API Key is not configured. Cannot generate AI voice.")
        return None

    language_code = '-'.join(voice_name.split('-')[:2])  # Derive from voice_name, e.g., "he-IL"
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": language_code, "name": voice_name},
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": speaking_rate,
            "volumeGainDb": volume_gain_db
        }
    }
    try:
        print(f"Requesting Google TTS for text: '{text[:50]}...' (Voice: {voice_name}, Rate: {speaking_rate})")
        # Using a context manager for the request
        with requests.post(url, json=payload, timeout=30, stream=False) as resp: # Use stream=False unless handling large responses differently
            resp.raise_for_status() # Raise exception for bad status codes (4xx, 5xx)
            response_json = resp.json()

        audio_content_base64 = response_json.get("audioContent")
        if not audio_content_base64:
            print(f"TTS Error: No audioContent received. Response: {response_json}")
            gr.Warning("Failed to get audio from Google TTS API.")
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
        status_code = e.response.status_code if e.response is not None else "N/A"
        print(f"Status Code: {status_code}")
        error_text = f"Google TTS request failed (Status: {status_code}): {e}"
        if status_code == 403:
             print("Check Google API key permissions and billing status.")
             gr.Error("Google TTS Error: Access Denied (403). Check API key/billing.")
        elif status_code == 400:
             print(f"Check payload/parameters. Response: {e.response.text if e.response is not None else 'No response body'}")
             gr.Warning("Google TTS Error: Invalid Request (400).")
        else:
            gr.Warning(error_text)
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Google TTS Processing Error: {e}")
        gr.Warning(f"Error during Google TTS processing: {e}")
        traceback.print_exc()
        return None

# --- Function to Handle Language Change ---
def update_language_settings(selected_language):
    new_system_prompt = system_prompts[selected_language]
    new_voice_choices = voice_options[selected_language]
    new_default_voice = default_voices[selected_language]
    # Update system prompt textbox
    updated_system_prompt = gr.update(value=new_system_prompt)
    # Update voice dropdown
    updated_voice_dropdown = gr.update(choices=new_voice_choices, value=new_default_voice)
    # Reset conversation state
    initial_state = get_initial_conversation_state(new_system_prompt)
    return (
        updated_system_prompt,
        updated_voice_dropdown,
        [],  # Clear chatbot
        initial_state,  # Reset conversation state
        None  # Clear audio output
    )

# =============================================================================
# Gradio UI Definition & Logic
# =============================================================================

with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as app:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown("Speak or type your request to the locksmith assistant. Ensure API keys are set.")
    
    # Display warnings if keys were missing during startup
    if missing_keys:
        gr.Markdown(f"**⚠️ Warning:** The following API keys were not detected: `{', '.join(missing_keys)}`. Please set them as environment variables and restart the application for full functionality.")

    # Main Chat Interface Elements
    chatbot_interface = gr.Chatbot(
        label="Conversation", height=550, bubble_full_width=False
    )
    audio_output_chat = gr.Audio(
        label="AI Response Audio", autoplay=True, interactive=False
    )

    # User Input Area
    with gr.Row():
        audio_input_chat = gr.Audio(
            label="Speak Your Message (English only, requires working transcription)",
            sources=["microphone"],
            type="filepath",
            # Disable if transcription function failed to load
            interactive=TRANSCRIPTION_AVAILABLE
        )
    with gr.Row():
         text_input_chat = gr.Textbox(
             label="Type Your Message",
             lines=3,
             placeholder="Type your request here, or use the microphone above...",
             scale=4 # Make text input wider
         )
         with gr.Column(scale=1, min_width=150):
              send_btn_chat = gr.Button("Send Message", variant="primary")
              clear_btn_chat = gr.Button("Clear Conversation")

    # Settings Area
    with gr.Accordion("⚙️ Settings", open=False):
        gr.Markdown("Adjust AI behavior and voice.")
        system_prompt_chat = gr.Textbox(
             label="System Prompt",
             value=system_prompt,
             lines=10,
             info="Defines the AI's role and rules."
        )
        language_chat = gr.Dropdown(
            label="Language",
            choices=["English", "Hebrew"],
            value="English",
            info="Select the language for the conversation."
        )
        google_voice_chat = gr.Dropdown(
            label="AI Voice (Google TTS)",
            choices=[
                "en-US-Wavenet-D", "en-US-Wavenet-F", "en-US-Wavenet-A", "en-US-Wavenet-E",
                "en-US-Neural2-J", "en-US-Neural2-F", "en-GB-Wavenet-B", "en-GB-Wavenet-F",
                "en-AU-Wavenet-B", "en-AU-Wavenet-C",
            ],
            value="en-US-Wavenet-D", # Default voice
            info="Select the Google TTS voice for the AI response."
        )
        speed_slider_google = gr.Slider(
            label="AI Voice Speed",
            minimum=0.5, maximum=1.5, value=1.05, step=0.05, # Adjusted range/default
            info="Adjust playback speed of the AI's voice (1.0 is normal)."
        )

    # --- State Management ---
    def get_initial_conversation_state(prompt):
        return [{"role": "system", "content": prompt}]

    conversation_state = gr.State(value=get_initial_conversation_state(system_prompt))

    # --- Core Logic Functions for Gradio ---

    def process_user_input(
        audio_path, text_input, current_history, conv_state_list
        ):
        """Handles audio/text input, gets AI text response."""
        print("-" * 20) # Separator for logs
        user_message = ""

        # 1. Get user message from audio or text
        if audio_path:
            if not TRANSCRIPTION_AVAILABLE:
                 gr.Error("Audio transcription is not available. Please type your message.")
                 return current_history, conv_state_list, gr.update(value="") # Return state unchanged

            print(f"Processing user audio input: {audio_path}")
            try:
                # Call transcription function (imported or dummy)
                # Assumes it returns (audio_data, transcribed_text)
                _, transcribed_text = preprocess_ref_audio_text(audio_path, "")
                user_message = transcribed_text.strip()
                print(f"Transcription result: '{user_message}'")
                if not user_message:
                    gr.Warning("Transcription failed or produced empty text.")
                    return current_history, conv_state_list, gr.update(value="")
            except Exception as e:
                 gr.Error(f"Error during audio transcription: {e}")
                 traceback.print_exc()
                 return current_history, conv_state_list, gr.update(value="")

        elif text_input and text_input.strip():
            user_message = text_input.strip()
            print(f"Processing user text input: '{user_message}'")
        else:
            # No valid input
            gr.Warning("Please type a message or record audio.")
            return current_history, conv_state_list, gr.update() # Return state unchanged

        # 2. Update conversation state and history (UI)
        current_conv = list(conv_state_list) # Ensure mutable
        current_conv.append({"role": "user", "content": user_message})
        updated_history = current_history + [[user_message, None]] # Show user message immediately

        # 3. Generate AI response (text)
        ai_response_text = generate_nebius_response(current_conv) # Pass current state

        # 4. Update state and history (UI) with AI response
        current_conv.append({"role": "assistant", "content": ai_response_text})
        updated_history[-1][1] = ai_response_text # Add AI response to chatbot display

        print("-" * 20) # Separator for logs
        # Return updated history, state, and clear input textbox
        return updated_history, current_conv, gr.update(value="")

    def generate_audio_for_response(
        current_history, google_api_key_val, google_voice_val, speed_val
        ):
        """Generates Google TTS audio for the latest AI response."""
        print("Attempting to generate Google TTS audio...")
        if not current_history:
            print("  History empty, skipping audio generation.")
            return None

        # Get the last AI response text from history
        _, last_ai_response = current_history[-1]

        if last_ai_response is None or not last_ai_response.strip():
            print("  Last AI response is empty or None, skipping audio generation.")
            return None # No text to synthesize

        # Call Google TTS function
        audio_file_path = google_tts_via_api_key(
            text=last_ai_response,
            api_key=google_api_key_val,
            voice_name=google_voice_val,
            speaking_rate=speed_val
        )

        if audio_file_path:
            print(f"  Successfully generated Google TTS audio: {audio_file_path}")
            # Return path for the gr.Audio component
            return gr.update(value=audio_file_path, autoplay=True)
        else:
            print("  Failed to generate Google TTS audio.")
            # No explicit warning here as google_tts_via_api_key should show one
            return None # Return None if TTS failed


    def clear_conversation_state(current_system_prompt):
        """Resets the chat history and state."""
        print("Clearing conversation.")
        initial_state = get_initial_conversation_state(current_system_prompt)
        # Clear chatbot UI, reset state, clear audio player, clear text input
        return [], initial_state, None, None

    def update_system_prompt_state(new_prompt):
        """Updates the system prompt and resets the conversation."""
        print("System prompt updated. Resetting conversation.")
        gr.Info("System prompt updated. Conversation has been reset.")
        initial_state = get_initial_conversation_state(new_prompt)
        # Clear chatbot UI, set new state, clear audio player
        return [], initial_state, None


    # --- Event Handling Wiring ---

    # Define shared inputs/outputs for processing text/audio -> AI text response
    process_inputs = [
        audio_input_chat,
        text_input_chat,
        chatbot_interface,
        conversation_state
    ]
    process_outputs = [
        chatbot_interface, # Update chatbot display
        conversation_state, # Update internal state
        text_input_chat # Clear text input
    ]

    # Define shared inputs/outputs for generating audio from AI text response
    audio_gen_inputs = [
        chatbot_interface, # Pass history to get the last AI response
        gr.State(GOOGLE_API_KEY), # Pass key securely
        google_voice_chat,
        speed_slider_google
    ]
    audio_gen_outputs = [
        audio_output_chat # Target the audio player
    ]

    # Chain events: Input -> Process Text -> Generate Audio
    # 1. Microphone stops recording
    audio_input_chat.stop_recording(
        process_user_input,
        inputs=process_inputs,
        outputs=process_outputs,
        show_progress="minimal"
    ).then(
        generate_audio_for_response,
        inputs=audio_gen_inputs,
        outputs=audio_gen_outputs,
        show_progress=False # Audio generation is usually fast
    ).then(
         lambda: None, None, audio_input_chat # Clear microphone waveform viz
    )

    # 2. Text submitted via Enter key
    text_input_chat.submit(
        process_user_input,
        inputs=process_inputs,
        outputs=process_outputs,
        show_progress="minimal"
    ).then(
        generate_audio_for_response,
        inputs=audio_gen_inputs,
        outputs=audio_gen_outputs,
        show_progress=False
    )

    # 3. Send button clicked
    send_btn_chat.click(
        process_user_input,
        inputs=process_inputs,
        outputs=process_outputs,
        show_progress="minimal"
    ).then(
        generate_audio_for_response,
        inputs=audio_gen_inputs,
        outputs=audio_gen_outputs,
        show_progress=False
    )

    # 4. Clear button clicked
    clear_btn_chat.click(
        clear_conversation_state,
        inputs=[system_prompt_chat], # Pass current system prompt to reset correctly
        outputs=[chatbot_interface, conversation_state, audio_output_chat, text_input_chat],
        queue=False # Make clearing instant
    )

    # 5. System prompt changed
    system_prompt_chat.change(
        update_system_prompt_state,
        inputs=system_prompt_chat,
        outputs=[chatbot_interface, conversation_state, audio_output_chat],
    )

    # 6. Language dropdown changed
    language_chat.change(
        update_language_settings,
        inputs=language_chat,
        outputs=[system_prompt_chat, google_voice_chat, chatbot_interface, conversation_state, audio_output_chat]
    )

# =============================================================================
# App Launch
# =============================================================================

if __name__ == "__main__":
    print("Starting Gradio app...")
    app.queue().launch(server_name="0.0.0.0",debug=True) # Add debug=True for more detailed Gradio logs