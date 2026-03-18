import io
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from parakeet_api.main import app


@pytest.fixture
def mock_stt_engine():
    with patch("parakeet_api.main.get_stt_engine") as mock_get:
        engine = MagicMock()
        engine.transcribe.return_value = {
            "text": "this is a test transcription.",
            "duration": 5.0,
        }
        mock_get.return_value = engine
        yield engine


@pytest.fixture
async def openai_client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as http_client:
        yield AsyncOpenAI(api_key="fake-key", http_client=http_client)


@pytest.mark.anyio
async def test_sdk_transcribe_json(mock_stt_engine, openai_client):
    # Test default 'json' response format
    audio_file = io.BytesIO(b"fake-audio-content")
    audio_file.name = "test.wav"

    response = await openai_client.audio.transcriptions.create(
        model="parakeet", file=audio_file, response_format="json"
    )
    assert response.text == "this is a test transcription."


@pytest.mark.anyio
async def test_sdk_transcribe_verbose_json_default(mock_stt_engine, openai_client):
    audio_file = io.BytesIO(b"fake-audio-content")
    audio_file.name = "test.wav"

    response = await openai_client.audio.transcriptions.create(
        model="parakeet", file=audio_file, response_format="verbose_json"
    )

    assert response.task == "transcribe"
    assert response.duration == 5.0
    assert response.text == "this is a test transcription."
    assert response.segments is not None
    assert len(response.segments) == 1
    assert response.words is None


@pytest.mark.anyio
async def test_sdk_transcribe_verbose_json_with_words(mock_stt_engine, openai_client):
    audio_file = io.BytesIO(b"fake-audio-content")
    audio_file.name = "test.wav"

    response = await openai_client.audio.transcriptions.create(
        model="parakeet",
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )

    assert response.words is not None
    assert len(response.words) > 0
    assert response.segments is None


@pytest.mark.anyio
async def test_sdk_transcribe_verbose_json_with_both(mock_stt_engine, openai_client):
    audio_file = io.BytesIO(b"fake-audio-content")
    audio_file.name = "test.wav"

    response = await openai_client.audio.transcriptions.create(
        model="parakeet",
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
    )

    assert response.words is not None
    assert response.segments is not None


@pytest.mark.anyio
async def test_sdk_transcribe_text(mock_stt_engine, openai_client):
    audio_file = io.BytesIO(b"fake-audio-content")
    audio_file.name = "test.wav"

    response = await openai_client.audio.transcriptions.with_raw_response.create(
        model="parakeet", file=audio_file, response_format="text"
    )
    assert response.status_code == 200
    assert response.text == "this is a test transcription."


def test_raw_endpoint_compatibility(mock_stt_engine):
    client = TestClient(app)
    response = client.post(
        "/v1/audio/transcriptions/raw?response_format=verbose_json&timestamp_granularities[]=word",
        content=b"fake-audio-binary-content",
        headers={"Content-Type": "audio/wav"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "this is a test transcription."
    assert "words" in data and data["words"] is not None
    assert "segments" not in data or data["segments"] is None
