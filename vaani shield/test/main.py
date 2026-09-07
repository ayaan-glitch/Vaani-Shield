import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load audio
audio_file = "audio/test.wav"

audio, sample_rate = librosa.load(audio_file, sr=None)

# Basic information
duration = len(audio) / sample_rate

print("===== VAANI-SHIELD AUDIO ANALYSIS =====")
print(f"Sample Rate : {sample_rate} Hz")
print(f"Samples     : {len(audio)}")
print(f"Duration    : {duration:.2f} seconds")
print(f"Max Volume  : {np.max(np.abs(audio)):.4f}")
print(f"Mean Volume : {np.mean(np.abs(audio)):.4f}")

# Plot waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(audio, sr=sample_rate)

plt.title("VAANI-SHIELD - Voice Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.show()