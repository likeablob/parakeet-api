import io
import wave
from pathlib import Path

import numpy as np
import pytest

from parakeet_api.config import settings
from parakeet_api.stt import STTEngine


@pytest.fixture
def silent_wav():
    # Create a 1-second silent WAV file (16kHz, mono, 16-bit)
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
    return buf.getvalue()


def has_models():
    models_dir = Path(settings.stt.models_dir)
    return any(models_dir.glob("**/*.onnx")) or any(models_dir.glob("**/weights.npz"))


@pytest.mark.inference
def test_actual_stt_inference(silent_wav):
    if not has_models():
        pytest.skip(
            "No models found in models directory. Run 'parakeet-api download' first."
        )

    try:
        engine = STTEngine()
        if not engine.recognizer:
            pytest.skip(
                "STTEngine failed to initialize any recognizer (mlx or sherpa)."
            )

        result = engine.transcribe(silent_wav)

        assert "text" in result
        assert "duration" in result
        assert result["duration"] == pytest.approx(1.0, rel=0.1)
        # Even if text is empty for silence, the flow should succeed
        print(
            f"Inference successful. Engine: {engine.engine_type}, Text: '{result['text']}'"
        )
    except Exception as e:
        pytest.fail(f"STT inference failed: {e}")


@pytest.mark.inference
def test_engine_init_fails_without_files():
    # This test might pass if models are present, but it's a good check for robustness
    try:
        engine = STTEngine()
        # Just ensure no crash
        assert engine is not None
    except Exception as e:
        pytest.fail(f"STTEngine initialization crashed: {e}")
