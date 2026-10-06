import asyncio
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import whisper
import edge_tts
from scipy.io.wavfile import write
from transformers import pipeline

SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
VOLUME_THRESHOLD = 0.02


class RealTimeSpeechTranslationAgent:
    def __init__(self):
        print("Loading models...")
        self.asr_model = whisper.load_model("base", device="cpu")
        self.translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-hi",
            device=-1,
        )
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.processor = threading.Thread(target=self._process_audio, daemon=True)
        self.processor.start()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        audio = indata[:, 0].astype(np.float32)
        if np.abs(audio).mean() < VOLUME_THRESHOLD:
            return
        self.audio_queue.put(audio.copy())

    def _process_audio(self):
        buffer = np.array([], dtype=np.float32)

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            buffer = np.concatenate((buffer, chunk))
            target_samples = SAMPLE_RATE * CHUNK_SECONDS

            while buffer.size >= target_samples:
                segment = buffer[:target_samples]
                buffer = buffer[target_samples:]
                self._translate_segment(segment)

    def _translate_segment(self, segment):
        wav_path = Path(tempfile.gettempdir()) / f"segment_{time.time_ns()}.wav"
        segmented = np.clip(segment, -1.0, 1.0)
        write(str(wav_path), SAMPLE_RATE, (segmented * 32767).astype(np.int16))

        result = self.asr_model.transcribe(
            str(wav_path),
            fp16=False,
            language="en",
            beam_size=5,
        )
        text = (result.get("text") or "").strip()
        if not text:
            return

        try:
            translated = self.translator(text, max_length=512)[0]["translation_text"]
        except Exception as exc:
            print(f"Translation failed: {exc}")
            return

        print(f"\n[EN] {text}")
        print(f"[HI] {translated}")

        try:
            asyncio.run(self.speak_hindi(translated))
        except Exception as exc:
            print(f"TTS failed: {exc}")

    async def speak_hindi(self, text: str):
        output_path = Path(tempfile.gettempdir()) / f"translated_{time.time_ns()}.mp3"
        communicate = edge_tts.Communicate(text=text, voice="hi-IN-SwaraNeural")
        await communicate.save(str(output_path))

        if sys.platform.startswith("win"):
            os.startfile(str(output_path))
        elif shutil.which("ffplay"):
            subprocess.run(["ffplay", "-nodisp", "-autoexit", str(output_path)], check=False)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["afplay", str(output_path)], check=False)
        else:
            print(f"[TTS] Saved Hindi speech to: {output_path}")

    def run(self):
        print("Listening for English speech... Press Ctrl+C to stop.")
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        ):
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nStopping the agent...")
                self.stop_event.set()


if __name__ == "__main__":
    agent = RealTimeSpeechTranslationAgent()
    agent.run()
