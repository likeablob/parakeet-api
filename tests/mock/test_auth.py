import io
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import AsyncOpenAI, AuthenticationError

from parakeet_api.main import app


@pytest.fixture
def mock_stt_engine():
    with patch("parakeet_api.main.get_stt_engine") as mock_get:
        engine = MagicMock()
        engine.transcribe.return_value = {
            "text": "authenticated transcription.",
            "duration": 2.0,
        }
        mock_get.return_value = engine
        yield engine


@pytest.mark.anyio
async def test_api_key_enforcement(mock_stt_engine, monkeypatch):
    # Set API key via monkeypatch (pydantic-settings should pick it up if re-instantiated)
    # Actually, main.py already imported config.settings.
    # We should patch config.settings.server.api_key directly for the test.
    from parakeet_api import config

    # Save original settings
    original_api_key = config.settings.server.api_key
    config.settings.server.api_key = "test-secret-key"

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as http_client:
            # 1. Test with missing key
            # Actually, AsyncOpenAI always sends a key if provided.
            # If we want to test "missing", we should use raw httpx.
            resp = await http_client.post(
                "/v1/audio/transcriptions", data={"model": "p"}
            )
            assert resp.status_code == 401

            # 2. Test with wrong key
            client_wrong_key = AsyncOpenAI(api_key="wrong-key", http_client=http_client)
            with pytest.raises(AuthenticationError):
                audio_file = io.BytesIO(b"audio")
                audio_file.name = "test.wav"
                await client_wrong_key.audio.transcriptions.create(
                    model="p", file=audio_file
                )

            # 3. Test with correct key
            client_correct_key = AsyncOpenAI(
                api_key="test-secret-key", http_client=http_client
            )
            audio_file = io.BytesIO(b"audio")
            audio_file.name = "test.wav"
            response = await client_correct_key.audio.transcriptions.create(
                model="p", file=audio_file
            )
            assert response.text == "authenticated transcription."

    finally:
        # Restore original settings
        config.settings.server.api_key = original_api_key


@pytest.mark.anyio
async def test_api_key_disabled_by_default(mock_stt_engine):
    from parakeet_api import config

    original_api_key = config.settings.server.api_key
    config.settings.server.api_key = None

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as http_client:
            # No key should be accepted
            resp = await http_client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", b"audio", "audio/wav")},
                data={"model": "p"},
            )
            assert resp.status_code == 200
    finally:
        config.settings.server.api_key = original_api_key
