# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "requests>=2.32.5",
# ]
# ///
import os

import requests

# Configuration
# By default, points to the local Parakeet API
API_URL = os.getenv("PARAKEET_API_URL", "http://localhost:8816/v1/audio/transcriptions")
# If SERVER__API_KEY is set on the server, provide it here
API_KEY = os.getenv("PARAKEET_API_KEY", "your-secret-api-key")


def transcribe_with_requests(audio_path):
    print(f"Transcribing {audio_path} using requests...")

    headers = {"Authorization": f"Bearer {API_KEY}"}

    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
        data = {"model": "parakeet", "response_format": "json"}

        response = requests.post(API_URL, headers=headers, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        print("Transcription success!")
        print(f"Text: {result['text']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    # Note: Replace with a real audio file path for testing
    TEST_AUDIO = "test.wav"
    if os.path.exists(TEST_AUDIO):
        transcribe_with_requests(TEST_AUDIO)
    else:
        print(f"Please provide a valid audio file at {TEST_AUDIO} to run this example.")
