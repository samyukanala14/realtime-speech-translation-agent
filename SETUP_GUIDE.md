# Real-Time Streaming Speech Translation Agent - Step-by-Step Build Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step 1: Environment Setup](#step-1-environment-setup)
3. [Step 2: Install Dependencies](#step-2-install-dependencies)
4. [Step 3: Verify Installation](#step-3-verify-installation)
5. [Step 4: Project Structure](#step-4-project-structure)
6. [Step 5: Core Components](#step-5-core-components)
7. [Step 6: Testing](#step-6-testing)
8. [Step 7: Troubleshooting](#step-7-troubleshooting)
9. [Step 8: Deployment & Optimization](#step-8-deployment--optimization)

---

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows 10+
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (4GB for models)
- **Microphone**: Working audio input device
- **Storage**: 2GB free space (for model downloads)

### Check Python Version
```bash
python --version
# Output should be Python 3.10.0 or higher
```

### Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y python3-dev python3-venv ffmpeg portaudio19-dev
```

#### macOS
```bash
brew install python@3.11 ffmpeg portaudio
```

#### Windows
1. Download Python 3.11 from [python.org](https://www.python.org/downloads/)
2. During installation, **check "Add Python to PATH"**
3. Install ffmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) or:
   ```bash
   choco install ffmpeg  # if using Chocolatey
   ```

---

## Step 1: Environment Setup

### 1.1 Clone the Repository
```bash
cd ~
git clone https://github.com/samyukanala14/realtime-speech-translation-agent.git
cd realtime-speech-translation-agent
```

### 1.2 Create a Virtual Environment

#### Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

### 1.3 Verify Virtual Environment is Active
You should see `(.venv)` prefix in your terminal:
```bash
(.venv) $ _
```

---

## Step 2: Install Dependencies

### 2.1 Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

### 2.2 Install Python Packages
```bash
pip install -r requirements.txt
```

### 2.3 Expected Installation Time
- `numpy`, `scipy`: ~2 minutes
- `sounddevice`: ~1 minute
- `torch`: ~10-15 minutes (CPU version, larger if GPU)
- `openai-whisper`: ~5 minutes
- `transformers`: ~5 minutes
- `edge-tts`: ~1 minute

**Total**: ~25-35 minutes on first install

### 2.4 Verify Installation
```bash
pip list
```

You should see all these packages listed.

---

## Step 3: Verify Installation

### 3.1 Test Python Imports
Create `test_imports.py`:
```python
import sys
print(f"Python: {sys.version}")

try:
    import numpy
    print("✓ numpy OK")
except ImportError as e:
    print(f"✗ numpy FAILED: {e}")

try:
    import sounddevice
    print("✓ sounddevice OK")
except ImportError as e:
    print(f"✗ sounddevice FAILED: {e}")

try:
    import whisper
    print("✓ whisper OK")
except ImportError as e:
    print(f"✗ whisper FAILED: {e}")

try:
    import transformers
    print("✓ transformers OK")
except ImportError as e:
    print(f"✗ transformers FAILED: {e}")

try:
    import edge_tts
    print("✓ edge_tts OK")
except ImportError as e:
    print(f"✗ edge_tts FAILED: {e}")

print("\nAll imports successful!")
```

Run it:
```bash
python test_imports.py
```

### 3.2 Test Audio Input
Create `test_audio.py`:
```python
import sounddevice as sd
import numpy as np

print("Available audio devices:")
print(sd.query_devices())

print("\nDefault input device:")
print(sd.default.device)

print("\nRecording 3 seconds...")
recording = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()

print(f"Recorded {len(recording)} samples")
print(f"Audio level: {np.abs(recording).mean():.4f}")
print("✓ Audio input OK")
```

Run it:
```bash
python test_audio.py
```

### 3.3 Test FFmpeg
```bash
ffmpeg -version
ffplay -h
```

---

## Step 4: Project Structure

```
realtime-speech-translation-agent/
├── .venv/                          # Virtual environment
├── app.py                          # Main application
├── requirements.txt                # Dependencies
├── README.md                       # Main documentation
├── SETUP_GUIDE.md                  # This file
├── test_imports.py                 # Import verification
├── test_audio.py                   # Audio device test
├── test_whisper.py                 # Whisper ASR test
├── test_translation.py             # Translation test
├── test_tts.py                     # TTS test
├── config.py                       # Configuration (optional)
├── utils.py                        # Utility functions (optional)
└── logs/                           # Log files (created at runtime)
```

---

## Step 5: Core Components

### Component 1: Audio Input (Microphone)

Create `test_whisper.py`:
```python
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
from pathlib import Path

print("Loading Whisper model (first time downloads ~1.4GB)...")
model = whisper.load_model("base")

print("Recording 5 seconds of audio...")
audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()

print("Transcribing...")
tmp_file = Path(tempfile.gettempdir()) / "test.wav"
write(str(tmp_file), 16000, (audio * 32767).astype(np.int16))

result = model.transcribe(str(tmp_file), language="en")
print(f"\nTranscribed text: {result['text']}")
print("✓ Whisper ASR OK")
```

Run it:
```bash
python test_whisper.py
```

### Component 2: Translation

Create `test_translation.py`:
```python
from transformers import pipeline

print("Loading translation model...")
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-hi",
    device=-1
)

text = "Hello, how are you?"
print(f"English: {text}")

result = translator(text, max_length=512)
translated = result[0]["translation_text"]

print(f"Hindi: {translated}")
print("✓ Translation OK")
```

Run it:
```bash
python test_translation.py
```

### Component 3: Text-to-Speech

Create `test_tts.py`:
```python
import asyncio
import edge_tts
from pathlib import Path
import tempfile
import os

async def test_tts():
    text = "नमस्कार, आप कैसे हैं?"
    print(f"Text to speak: {text}")
    
    output_path = Path(tempfile.gettempdir()) / "test_hi.mp3"
    print(f"Saving to: {output_path}")
    
    communicate = edge_tts.Communicate(text=text, voice="hi-IN-SwaraNeural")
    await communicate.save(str(output_path))
    
    print(f"✓ TTS OK - File saved")
    
    if os.name != 'nt':
        os.system(f"ffplay -nodisp -autoexit {output_path} 2>/dev/null")
    else:
        os.startfile(str(output_path))

asyncio.run(test_tts())
```

Run it:
```bash
python test_tts.py
```

---

## Step 6: Testing

### 6.1 Full Integration Test

Create `test_full.py`:
```python
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
audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
print(f"✓ Recorded {len(audio)} samples")

# Step 2: Transcribe
print("\n[2/5] Loading Whisper model...")
model = whisper.load_model("base")
print("✓ Loaded")

print("\n[3/5] Transcribing...")
tmp_file = Path(tempfile.gettempdir()) / "test.wav"
write(str(tmp_file), 16000, (audio * 32767).astype(np.int16))
result = model.transcribe(str(tmp_file), language="en")
english_text = result['text'].strip()
print(f"✓ English: {english_text}")

# Step 3: Translate
print("\n[4/5] Translating to Hindi...")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi", device=-1)
translated = translator(english_text, max_length=512)[0]["translation_text"]
print(f"✓ Hindi: {translated}")

# Step 4: Synthesize
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
print(f"\nFinal Output: {output_file}")
```

Run it:
```bash
python test_full.py
```

### 6.2 Run the Main Application

```bash
python app.py
```

Expected output:
```
Loading models...
Listening for English speech... Press Ctrl+C to stop.
```

Speak into your microphone. You should see:
```
[EN] Hello, how are you?
[HI] नमस्कार, आप कैसे हैं?
```

---

## Step 7: Troubleshooting

### Issue 1: "No module named 'sounddevice'"
**Solution:**
```bash
pip install --upgrade sounddevice
```

### Issue 2: "ALSA lib confmisc.c: Cannot connect to server"
**Cause:** ALSA warnings on Linux (harmless)
**Solution:** Ignore or suppress:
```bash
export ALSA_CARD=default
python app.py 2>/dev/null
```

### Issue 3: "No audio devices found"
**Solution:** Check available devices:
```python
import sounddevice as sd
print(sd.query_devices())
print(sd.default.device)
```

If empty, check audio:
- Linux: `arecord -l`
- macOS: `System Preferences > Sound > Input`
- Windows: Sound Settings

### Issue 4: "Whisper model download fails"
**Solution:**
```bash
pip install --upgrade openai-whisper
# Then try again - first run will download ~1.4GB
```

### Issue 5: "Translation is very slow"
**Cause:** First run downloads the model (~1.4GB)
**Solution:** Wait for first run to complete. Subsequent translations will be fast.

### Issue 6: "TTS voice not available"
**Solution:** Check available voices:
```python
import edge_tts
async def list_voices():
    voices = await edge_tts.list_voices()
    for voice in voices:
        if 'Hindi' in voice['Locale'] or 'hi' in voice['Locale']:
            print(voice)

import asyncio
asyncio.run(list_voices())
```

### Issue 7: "ffplay not found on macOS"
**Solution:**
```bash
brew install ffmpeg --with-tools
```

---

## Step 8: Deployment & Optimization

### 8.1 Optimize for Performance

Create `config.py`:
```python
# Configuration
SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
VOLUME_THRESHOLD = 0.02
WHISPER_MODEL = "base"  # or "tiny" for faster, less accurate
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-hi"
TTS_VOICE = "hi-IN-SwaraNeural"
DEVICE = "cpu"  # or "cuda" if you have GPU
```

### 8.2 Create a Config Management Script

Create `utils.py`:
```python
import logging
from pathlib import Path

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def get_available_audio_devices():
    import sounddevice as sd
    return sd.query_devices()

def format_output(english, hindi):
    return f"""
╔════════════════════════════════════════════════╗
║           SPEECH TRANSLATION RESULT            ║
╠════════════════════════════════════════════════╣
║ English: {english:40} ║
║ Hindi:   {hindi:40} ║
╚════════════════════════════════════════════════╝
"""
```

### 8.3 Docker Deployment (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t speech-translation .
docker run --device /dev/snd speech-translation
```

### 8.4 Create a Requirements-Dev File

Create `requirements-dev.txt`:
```
-r requirements.txt
pytest
pytest-cov
black
flake8
mypy
```

### 8.5 Performance Benchmarking

Create `benchmark.py`:
```python
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

# Create dummy audio
audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1
tmp_file = Path(tempfile.gettempdir()) / "bench.wav"
write(str(tmp_file), 16000, (audio * 32767).astype(np.int16))

# Benchmark Whisper
print("\n[ASR] Loading Whisper...")
start = time.time()
model = whisper.load_model("base")
print(f"Load time: {time.time() - start:.2f}s")

print("Transcribing...")
start = time.time()
result = model.transcribe(str(tmp_file), language="en")
asr_time = time.time() - start
print(f"Transcription time: {asr_time:.2f}s")

# Benchmark Translation
print("\n[Translation] Loading model...")
start = time.time()
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi", device=-1)
print(f"Load time: {time.time() - start:.2f}s")

text = "Hello world"
print("Translating...")
start = time.time()
translator(text, max_length=512)
trans_time = time.time() - start
print(f"Translation time: {trans_time:.4f}s")

# Benchmark TTS
print("\n[TTS] Testing...")
start = time.time()
async def tts_bench():
    output = Path(tempfile.gettempdir()) / "bench.mp3"
    comm = edge_tts.Communicate(text="नमस्कार", voice="hi-IN-SwaraNeural")
    await comm.save(str(output))

asyncio.run(tts_bench())
tts_time = time.time() - start
print(f"TTS time: {tts_time:.2f}s")

print("\n" + "=" * 60)
print(f"Total Pipeline Time: {asr_time + trans_time + tts_time:.2f}s")
print("=" * 60)
```

Run it:
```bash
python benchmark.py
```

### 8.6 Final Checklist

- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] Audio devices detected
- [ ] Whisper model works
- [ ] Translation model works
- [ ] TTS voice works
- [ ] Main app runs successfully
- [ ] Microphone captures audio
- [ ] Transcription is accurate
- [ ] Translation is correct
- [ ] Hindi speech plays

---

## Next Steps

1. **Customize for other languages:**
   - Change `Helsinki-NLP/opus-mt-en-hi` to other language pairs
   - Update voice in Edge TTS

2. **Add GUI:**
   - Create web interface with Flask/FastAPI
   - Build React frontend

3. **Production deployment:**
   - Add Docker support
   - Deploy to cloud (AWS/GCP/Azure)
   - Add logging and monitoring

4. **Optimize performance:**
   - Use GPU acceleration (CUDA)
   - Switch to "tiny" Whisper model
   - Implement streaming ASR

---

## Support & Resources

- **Whisper Docs:** https://github.com/openai/whisper
- **Hugging Face:** https://huggingface.co/models
- **Edge TTS:** https://github.com/rany2/edge-tts
- **Sounddevice:** https://python-sounddevice.readthedocs.io/

---

**Happy building! 🎤🌍📝**
