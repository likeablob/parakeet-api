import io
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest
from openai import AsyncOpenAI

from parakeet_api.config import settings
from parakeet_api.main import app


@pytest.fixture
def silent_wav():
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    samples = np.zeros(num_samples, dtype=np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    buf.name = "test.wav"
    return buf


def has_models():
    models_dir = Path(settings.stt.models_dir)
    return any(models_dir.glob("**/*.onnx")) or any(models_dir.glob("**/weights.npz"))


@pytest.fixture
async def openai_client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as http_client:
        yield AsyncOpenAI(api_key="fake-key", http_client=http_client)


@pytest.mark.inference
@pytest.mark.anyio
async def test_e2e_api_inference(openai_client, silent_wav):
    """
    SDK -> API -> STT Engine (Actual) -> API -> SDK の全経路を確認
    """
    if not has_models():
        pytest.skip("No models found. Skipping E2E inference test.")

    # NOTE: get_stt_engine をモックせず、実際の初期化を走らせる
    response = await openai_client.audio.transcriptions.create(
        model="parakeet",
        file=silent_wav,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
    )

    assert response.duration == pytest.approx(1.0, rel=0.1)
    assert response.task == "transcribe"
    assert response.text is not None
    assert response.segments is not None
    assert response.words is not None

    print(f"E2E Inference Success: Text='{response.text}'")
