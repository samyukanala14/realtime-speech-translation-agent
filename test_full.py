import asyncio
import tempfile
from pathlib import Path
from scipy.io.wavfile import write
import numpy as np
import sounddevice as sd
import whisper
from transformers import pipeline
import edge_tts

print("=" * 60)
print("FULL INTEGRATION TEST")
print("=" * 60)

# Step 1: Record Audio
print("\n[1/5] Recording 5 seconds of speech...")
print("Start speaking now...")
audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
print(f"✓ Recorded {len(audio)} samples")

# Step 2: Load Whisper
print("\n[2/5] Loading Whisper model...")
model = whisper.load_model("base")
print("✓ Loaded")

# Step 3: Transcribe
print("\n[3/5] Transcribing...")
tmp_file = Path(tempfile.gettempdir()) / "test.wav"
write(str(tmp_file), 16000, (audio * 32767).astype(np.int16))
result = model.transcribe(str(tmp_file), language="en")
english_text = result['text'].strip()
if not english_text:
    english_text = "[No speech detected]"
print(f"✓ English: {english_text}")

# Step 4: Translate
if english_text != "[No speech detected]":
    print("\n[4/5] Translating to Hindi...")
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi", device=-1)
    translated = translator(english_text, max_length=512)[0]["translation_text"]
    print(f"✓ Hindi: {translated}")

    # Step 5: Synthesize
    print("\n[5/5] Generating Hindi speech...")

    async def speak():
        output_path = Path(tempfile.gettempdir()) / "test_output.mp3"
        communicate = edge_tts.Communicate(text=translated, voice="hi-IN-SwaraNeural")
        await communicate.save(str(output_path))
        print(f"✓ Saved to: {output_path}")
        return str(output_path)

    output_file = asyncio.run(speak())

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
else:
    print("\n✗ No speech detected. Please try again with louder speech.")
