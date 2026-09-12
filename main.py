
import librosa
import numpy as np


# ==========================================
# VAANI-SHIELD
# AUDIO LOADING TEST
# ==========================================

audio_file = "audio/test.wav"

print("\n==========================================")
print("        VAANI-SHIELD AUDIO TEST")
print("==========================================")

print("\nLoading audio...")

audio, sample_rate = librosa.load(
    audio_file,
    sr=None
)

print("Audio loaded successfully! ✅")


# ==========================================
# BASIC AUDIO INFORMATION
# ==========================================

duration = len(audio) / sample_rate

print("\n===== AUDIO INFORMATION =====")

print(f"File        : {audio_file}")
print(f"Sample Rate : {sample_rate} Hz")
print(f"Samples     : {len(audio)}")
print(f"Duration    : {duration:.2f} seconds")
print(f"Max Volume  : {np.max(np.abs(audio)):.4f}")
print(f"Mean Volume : {np.mean(np.abs(audio)):.4f}")


# ==========================================
# SUCCESS
# ==========================================

print("\n==========================================")
print("       AUDIO TEST COMPLETED ✅")
print("==========================================")

print("\nVAANI-SHIELD can successfully read")
print("your voice recording.")

print("\nNext step: MFCC + Pitch + Energy +")
print("Spectral Feature Extraction 🚀")
