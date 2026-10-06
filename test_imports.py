import sys

print("Checking Python dependencies...\n")

dependencies = [
    ("numpy", "NumPy"),
    ("sounddevice", "sounddevice"),
    ("scipy", "SciPy"),
    ("whisper", "OpenAI Whisper"),
    ("transformers", "Hugging Face Transformers"),
    ("torch", "PyTorch"),
    ("edge_tts", "Edge TTS"),
]

failed = []

for module_name, display_name in dependencies:
    try:
        __import__(module_name)
        print(f"✓ {display_name:30} OK")
    except ImportError as e:
        print(f"✗ {display_name:30} FAILED")
        failed.append((display_name, str(e)))

print(f"\nPython Version: {sys.version}")

if failed:
    print(f"\n⚠️  {len(failed)} dependencies missing:")
    for name, error in failed:
        print(f"  - {name}: {error}")
    print("\nRun: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\n✓ All dependencies installed successfully!")
    sys.exit(0)
