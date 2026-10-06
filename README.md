# Real-Time Streaming Speech Translation Agent

This project is a Python prototype that listens to microphone input, uses Whisper to transcribe spoken English, translates the text to Hindi, and speaks the translated output through Microsoft Edge TTS.

## Architecture

```text
Microphone -> Audio Stream -> Whisper ASR -> English Text -> Translation Model -> Hindi Text -> Edge TTS -> Hindi Voice Output
```

## Features

- Real-time microphone capture using `sounddevice`
- Chunked audio processing for low-latency translation
- English ASR with `openai-whisper`
- English-to-Hindi translation using a Hugging Face MarianMT model
- Hindi speech generation using Microsoft Edge TTS

## Requirements

- Python 3.10+
- `ffmpeg` installed and available in your PATH
- `ffplay` recommended for playback on Linux/macOS

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are on Ubuntu/Debian, install `ffmpeg` with:

```bash
sudo apt update && sudo apt install ffmpeg
```

## Run

```bash
python app.py
```

Then speak in English into your microphone. The app will print the transcript and translated Hindi sentence, and play the Hindi voice response.

## Notes

This is a prototype that works in short audio chunks. For production-grade real-time translation, you would typically add:

- streaming ASR such as FasterWhisper or Azure Speech SDK
- WebSocket or gRPC streaming pipeline
- proper buffering and concurrency controls
- multilingual voice selection and fallback handling
- queueing for translated speech playback

## Example output

```text
[EN] Hello, how are you?
[HI] नमस्कार, आप कैसे हैं?
```
