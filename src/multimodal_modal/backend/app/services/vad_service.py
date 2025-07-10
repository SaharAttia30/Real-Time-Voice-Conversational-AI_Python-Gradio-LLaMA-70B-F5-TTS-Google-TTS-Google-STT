import webrtcvad
import webrtcvad, wave

def is_speech_frame(frame: bytes, sample_rate: int) -> bool:
    vad = webrtcvad.Vad(1)  # set aggressiveness (0-3)
    return vad.is_speech(frame, sample_rate)
