from faster_whisper import WhisperModel

model = WhisperModel("large-v2", device="cuda", compute_type="float16")

segments, info = model.transcribe("audio.wav")

print("Idioma detectado:", info.language)

for segment in segments:
    print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}")