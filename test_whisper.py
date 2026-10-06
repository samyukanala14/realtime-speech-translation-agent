import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
from pathlib import Path

print("=" * 60)
print("WHISPER ASR TEST")
print("=" * 60)

print("\nThis will download the Whisper model (~1.4GB) on first run.")
print("Subsequent runs will use the cached model.\n")

print("Loading Whisper model...")
model = whisper.load_model("base")
print("✓ Model loaded\n")

print("Recording 10 seconds of audio...")
print("Speak now!\n")
audio = sd.rec(int(10 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()

print("\nTranscribing audio...")
tmp_file = Path(tempfile.gettempdir()) / "whisper_test.wav"
write(str(tmp_file), 16000, (audio * 32767).astype(np.int16))

result = model.transcribe(str(tmp_file), language="en", fp16=False)

print("\n" + "=" * 60)
print("RESULT:")
print("=" * 60)
print(f"\nTranscribed Text: {result['text']}")
print(f"Language: {result['language']}")
print(f"Duration: {result['duration']:.2f}s")

if result.get('segments'):
    print(f"\nSegments:")
    for i, seg in enumerate(result['segments'][:3], 1):
        print(f"  {i}. [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")

print("\n✓ Whisper ASR test passed!")
