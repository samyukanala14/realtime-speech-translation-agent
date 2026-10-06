import sounddevice as sd
import numpy as np

print("=" * 60)
print("AUDIO DEVICE TEST")
print("=" * 60)

print("\nAvailable audio devices:")
print(sd.query_devices())

print("\nDefault input device:")
default_input = sd.default.device[0]
print(f"Device {default_input}: {sd.query_devices(default_input)['name']}")

print("\nDefault output device:")
default_output = sd.default.device[1]
print(f"Device {default_output}: {sd.query_devices(default_output)['name']}")

print("\n" + "=" * 60)
print("Recording 3 seconds (speak into microphone)...")
print("=" * 60)

recording = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()

print(f"\n✓ Recorded {len(recording)} samples")
print(f"✓ Audio level (RMS): {np.abs(recording).mean():.4f}")
print(f"✓ Peak level: {np.abs(recording).max():.4f}")

if np.abs(recording).mean() < 0.01:
    print("\n⚠️  Low audio level detected. Check microphone volume.")
else:
    print("\n✓ Audio level OK")

print("\n✓ Audio device test passed!")
