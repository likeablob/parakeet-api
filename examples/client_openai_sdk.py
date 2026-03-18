# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "openai>=2.29.0",
# ]
# ///
import os

from openai import OpenAI

# Configuration
# Points to the local Parakeet API by setting base_url
# The 'openai' library expects the base URL to end with /v1
BASE_URL = os.getenv("PARAKEET_BASE_URL", "http://localhost:8816/v1")
# If SERVER__API_KEY is set on the server, provide it here
API_KEY = os.getenv("PARAKEET_API_KEY", "your-secret-api-key")


def transcribe_with_openai_sdk(audio_path):
    print(f"Transcribing {audio_path} using OpenAI SDK...")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="parakeet",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

    print("Transcription success!")
    print(f"Text: {transcription.text}")
    print(f"Duration: {transcription.duration}s")

    if transcription.words:
        print(f"First 5 words: {[w.word for w in transcription.words[:5]]}")


if __name__ == "__main__":
    # Note: Replace with a real audio file path for testing
    TEST_AUDIO = "test.wav"
    if os.path.exists(TEST_AUDIO):
        transcribe_with_openai_sdk(TEST_AUDIO)
    else:
        print(f"Please provide a valid audio file at {TEST_AUDIO} to run this example.")
