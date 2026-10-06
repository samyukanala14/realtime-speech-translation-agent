import asyncio
import edge_tts
from pathlib import Path
import tempfile
import subprocess
import sys
import os

print("=" * 60)
print("EDGE TTS TEST")
print("=" * 60)

test_texts = [
    ("नमस्कार, आप कैसे हैं?", "Hi, how are you?"),
    ("यह एक परीक्षण है।", "This is a test."),
]

async def test_tts():
    for hindi_text, english_meaning in test_texts:
        print(f"\nEnglish meaning: {english_meaning}")
        print(f"Hindi text: {hindi_text}")
        
        output_path = Path(tempfile.gettempdir()) / f"tts_test_{hash(hindi_text)}.mp3"
        print(f"Saving to: {output_path}")
        
        try:
            communicate = edge_tts.Communicate(text=hindi_text, voice="hi-IN-SwaraNeural")
            await communicate.save(str(output_path))
            print("✓ Audio generated successfully")
            
            file_size = output_path.stat().st_size
            print(f"✓ File size: {file_size / 1024:.2f} KB")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    return True

print("\nGenerating Hindi speech samples...")
success = asyncio.run(test_tts())

if success:
    print("\n" + "=" * 60)
    print("✓ Edge TTS test passed!")
    print("=" * 60)
else:
    print("\n✗ Edge TTS test failed!")
    sys.exit(1)
