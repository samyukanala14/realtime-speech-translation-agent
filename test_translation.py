from transformers import pipeline

print("=" * 60)
print("TRANSLATION TEST")
print("=" * 60)

print("\nLoading translation model...")
print("(First run will download ~1.4GB on first use)\n")

translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-hi",
    device=-1
)

print("✓ Model loaded\n")

test_sentences = [
    "Hello, how are you?",
    "Welcome to the real-time speech translation system.",
    "This is a test of the translation model.",
]

print("=" * 60)
print("TEST TRANSLATIONS:")
print("=" * 60)

for i, sentence in enumerate(test_sentences, 1):
    print(f"\n[Test {i}]")
    print(f"English: {sentence}")
    
    result = translator(sentence, max_length=512)
    translated = result[0]["translation_text"]
    
    print(f"Hindi:   {translated}")

print("\n" + "=" * 60)
print("✓ Translation test passed!")
print("=" * 60)
