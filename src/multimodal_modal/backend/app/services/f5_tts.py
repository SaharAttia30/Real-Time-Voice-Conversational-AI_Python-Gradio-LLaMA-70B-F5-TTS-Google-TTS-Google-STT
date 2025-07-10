# services/f5_tts.py

import subprocess

def synthesize_f5(text: str, ref_audio_path: str, model: str="F5TTS_v1_Base") -> bytes:
    cmd = [
        "f5-tts_infer-cli", "--model", model,
        "--ref_audio", ref_audio_path, "--ref_text", text
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError("F5-TTS synthesis failed")
    # Assume the CLI writes output to a file; load and return it:
    with open("f5_output.wav", "rb") as f:
        return f.read()
