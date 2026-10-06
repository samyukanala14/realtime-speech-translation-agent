import time
import numpy as np
from scipy.io.wavfile import write
import tempfile
from pathlib import Path
import whisper
from transformers import pipeline
import edge_tts
import asyncio

print("=" * 60)
print("PERFORMANCE BENCHMARK")
print("=" * 60)

# Create dummy audio (5 seconds of silence with slight noise)
audio = np.random.randn(16000 * 5).astype(np.float32) * 0.05
tmp_file = Path(tempfile.gettempdir()) / "bench.wav"
write(str(tmp_file), 16000, (audio * 32767).astype(np.int16))
print(f"\nCreated test audio: {tmp_file}")
print(f"Audio duration: 5 seconds")

# Benchmark Whisper
print("\n" + "=" * 60)
print("[1/3] ASR (Whisper)")
print("=" * 60)
print("Loading model...")
start = time.time()
model = whisper.load_model("base")
load_time = time.time() - start
print(f"Load time: {load_time:.2f}s")

print("Transcribing audio...")
start = time.time()
result = model.transcribe(str(tmp_file), language="en", fp16=False)
asr_time = time.time() - start
print(f"Transcription time: {asr_time:.2f}s")
print(f"Transcribed text: {result['text'][:50]}..." if result['text'] else "[No speech detected]")

# Benchmark Translation
print("\n" + "=" * 60)
print("[2/3] Translation")
print("=" * 60)
print("Loading model...")
start = time.time()
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-hi",
    device=-1
)
load_time = time.time() - start
print(f"Load time: {load_time:.2f}s")

test_text = "Hello, this is a test of the translation system."
print(f"Test text: {test_text}")

print("Translating...")
start = time.time()
result = translator(test_text, max_length=512)
trans_time = time.time() - start
print(f"Translation time: {trans_time:.4f}s")
print(f"Translated: {result[0]['translation_text']}")

# Benchmark TTS
print("\n" + "=" * 60)
print("[3/3] TTS (Edge TTS)")
print("=" * 60)

async def tts_bench():
    output = Path(tempfile.gettempdir()) / "bench.mp3"
    print("Generating Hindi speech...")
    start_tts = time.time()
    comm = edge_tts.Communicate(text="नमस्कार", voice="hi-IN-SwaraNeural")
    await comm.save(str(output))
    tts_time = time.time() - start_tts
    print(f"TTS time: {tts_time:.2f}s")
    return tts_time

tts_time = asyncio.run(tts_bench())

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)
print(f"\nASR (Whisper):   {asr_time:>8.2f}s")
print(f"Translation:     {trans_time:>8.4f}s")
print(f"TTS (Edge):      {tts_time:>8.2f}s")
print(f"\nTotal:           {asr_time + trans_time + tts_time:>8.2f}s")
print(f"\nNote: First run includes model download time.")
print(f"Subsequent runs will be faster (cached models).")
print("=" * 60)
