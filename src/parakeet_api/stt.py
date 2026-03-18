import io
import logging
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx
from pydub import AudioSegment

from .config import settings

try:
    from parakeet_mlx import from_pretrained as from_pretrained_mlx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class STTEngine:
    def __init__(self):
        self.recognizer = None
        self.engine_type = None

        logging.info(f"STTEngine CWD: {Path.cwd()}")
        logging.info(f"HAS_MLX: {HAS_MLX}")

        if HAS_MLX:
            self._init_mlx()

        if not self.recognizer:
            self._init_sherpa()

        if not self.recognizer:
            logging.error("No STT engine (mlx or sherpa) initialized.")

    def _convert_to_pcm_16k(self, audio_bytes: bytes) -> tuple[bytes, int]:
        """Convert any audio format to 16kHz mono PCM and return (raw_bytes, sample_rate)."""
        start = time.perf_counter()
        # Fast path: If it's already a WAV, check if it's 16kHz mono
        if audio_bytes.startswith(b"RIFF"):
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as f:
                    if (
                        f.getnchannels() == 1
                        and f.getsampwidth() == 2
                        and f.getframerate() == 16000
                    ):
                        # Already in target format, just extract raw PCM
                        elapsed = (time.perf_counter() - start) * 1000
                        logging.debug(
                            f"Audio conversion: Fast path used ({elapsed:.2f}ms)"
                        )
                        return f.readframes(f.getnframes()), 16000
            except Exception:
                pass

        if settings.stt.disable_conversion:
            # If conversion is disabled, we might still want to try to extract PCM from any WAV
            if audio_bytes.startswith(b"RIFF"):
                try:
                    with wave.open(io.BytesIO(audio_bytes), "rb") as f:
                        rate = f.getframerate()
                        raw_data = f.readframes(f.getnframes())
                        elapsed = (time.perf_counter() - start) * 1000
                        logging.debug(
                            f"Audio conversion: WAV extraction ({elapsed:.2f}ms)"
                        )
                        return raw_data, rate
                except Exception:
                    pass
            # Otherwise, just pass through and assume 16kHz
            elapsed = (time.perf_counter() - start) * 1000
            logging.debug(f"Audio conversion: Passthrough ({elapsed:.2f}ms)")
            return audio_bytes, 16000

        # 1. Try pydub
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio = audio.set_frame_rate(16000).set_channels(1)
            elapsed = (time.perf_counter() - start) * 1000
            logging.debug(f"Audio conversion: pydub used ({elapsed:.2f}ms)")
            return audio.raw_data, 16000
        except Exception as e:
            logging.warning(f"pydub failed to convert audio: {e}. Falling back...")

        # 2. Try ffmpeg command directly if pydub fails
        try:
            cmd = [
                "ffmpeg",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "pipe:1",
            ]
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out, err = process.communicate(input=audio_bytes)
            if process.returncode == 0:
                elapsed = (time.perf_counter() - start) * 1000
                logging.debug(f"Audio conversion: ffmpeg used ({elapsed:.2f}ms)")
                return out, 16000
            else:
                logging.warning(f"ffmpeg conversion failed: {err.decode()}")
        except Exception as e:
            logging.error(f"Failed to run ffmpeg: {e}")

        # 3. Fallback to existing manual WAV parsing if conversion failed
        if audio_bytes.startswith(b"RIFF"):
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as f:
                    rate = f.getframerate()
                    raw_data = f.readframes(f.getnframes())
                    elapsed = (time.perf_counter() - start) * 1000
                    logging.debug(
                        f"Audio conversion: manual WAV parsing ({elapsed:.2f}ms)"
                    )
                    return raw_data, rate
            except Exception as e:
                logging.error(f"Failed to parse WAV manually: {e}")

        # Last resort: assume it's already 16kHz PCM
        elapsed = (time.perf_counter() - start) * 1000
        logging.debug(f"Audio conversion: last resort ({elapsed:.2f}ms)")
        return audio_bytes, 16000

    def _init_mlx(self):
        try:
            model_id = settings.stt.mlx.model_id

            base_dir = Path(settings.stt.models_dir)
            local_path = base_dir / "mlx" / model_id.split("/")[-1]

            model_to_load = str(local_path) if local_path.exists() else model_id

            logging.info(f"Initializing MLX Parakeet with {model_to_load}...")
            self.recognizer = from_pretrained_mlx(model_to_load)
            self.engine_type = "mlx"
            logging.info("MLX Parakeet initialized successfully.")
        except Exception as e:
            logging.exception(f"Failed to init MLX: {e}")
            self.recognizer = None

    def _init_sherpa(self):
        base_dir = Path(settings.stt.models_dir)
        model_dir = base_dir / "sherpa" / settings.stt.sherpa.model_id

        tokens_path = model_dir / "tokens.txt"
        if not tokens_path.exists():
            logging.error(f"Sherpa tokens.txt not found in {model_dir}")
            return

        def find_onnx(name):
            for ext in [".onnx", ".int8.onnx"]:
                p = model_dir / f"{name}{ext}"
                if p.exists():
                    return str(p)
            return None

        encoder = find_onnx("encoder")
        decoder = find_onnx("decoder")
        joiner = find_onnx("joiner")
        nemo_ctc = find_onnx("model")

        provider = settings.stt.sherpa.provider
        num_threads = settings.stt.sherpa.num_threads
        f_dim = 80

        try:
            if encoder and decoder and joiner:
                logging.info(
                    f"Initializing Sherpa Transducer (TDT) from {model_dir} with feature_dim={f_dim}"
                )
                self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=encoder,
                    decoder=decoder,
                    joiner=joiner,
                    tokens=str(tokens_path),
                    num_threads=num_threads,
                    sample_rate=16000,
                    feature_dim=f_dim,
                    decoding_method="greedy_search",
                    provider=provider,
                    debug=settings.server.debug,
                    model_type="nemo_transducer",
                )
            elif nemo_ctc:
                logging.info(
                    f"Initializing Sherpa Nemo CTC from {model_dir} with feature_dim={f_dim}"
                )
                self.recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                    model=nemo_ctc,
                    tokens=str(tokens_path),
                    num_threads=num_threads,
                    sample_rate=16000,
                    feature_dim=f_dim,
                    decoding_method="greedy_search",
                    provider=provider,
                    debug=settings.server.debug,
                )
            else:
                logging.error(
                    f"No valid Sherpa model files found in {model_dir}. "
                    "Expected either (encoder, decoder, joiner) or (model.onnx)."
                )
                return

            self.engine_type = "sherpa_offline"
            logging.info(f"Sherpa-ONNX initialized successfully as {self.engine_type}.")
        except Exception as e:
            logging.exception(f"Failed to initialize Sherpa-ONNX: {e}")
            self.recognizer = None

    def pcm_to_wav(self, pcm_data: bytes, sample_rate=16000) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm_data)
        return buf.getvalue()

    def transcribe(self, audio_bytes: bytes) -> dict:
        if not self.recognizer:
            raise RuntimeError("STT engine not initialized (no mlx or sherpa found)")

        start_total = time.perf_counter()

        # Preprocess: Always convert to 16kHz mono PCM for internal processing
        pcm_raw, sample_rate = self._convert_to_pcm_16k(audio_bytes)
        duration = len(pcm_raw) / (2 * sample_rate)  # 16-bit PCM = 2 bytes per sample
        text = ""

        start_inference = time.perf_counter()

        if self.engine_type == "mlx":
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                # MLX expects a file path. We wrap our PCM in a WAV container.
                wav_data = self.pcm_to_wav(pcm_raw, sample_rate)
                tmp.write(wav_data)
                tmp_path = tmp.name
            try:
                result = self.recognizer.transcribe(tmp_path)
                text = result.text.strip()
            except Exception as e:
                logging.error(f"MLX Transcription failed: {e}", exc_info=True)
                raise RuntimeError(f"MLX Transcription failed: {e}") from e
            finally:
                p = Path(tmp_path)
                if p.exists():
                    p.unlink()
        else:
            # Handle Sherpa
            try:
                samples = (
                    np.frombuffer(pcm_raw, dtype=np.int16).astype(np.float32) / 32768.0
                )

                if self.engine_type == "sherpa_offline":
                    stream = self.recognizer.create_stream()
                    stream.accept_waveform(sample_rate, samples)
                    self.recognizer.decode_stream(stream)
                    text = stream.result.text.strip()
            except Exception as e:
                logging.error(f"Sherpa Transcription failed: {e}", exc_info=True)
                raise RuntimeError(f"Sherpa Transcription failed: {e}") from e

        end_time = time.perf_counter()
        inference_elapsed = (end_time - start_inference) * 1000
        total_elapsed = (end_time - start_total) * 1000

        logging.info(
            f"STT: engine={self.engine_type}, inference={inference_elapsed:.2f}ms, total={total_elapsed:.2f}ms"
        )

        if not text and self.engine_type is None:
            raise RuntimeError(f"Unknown STT engine type: {self.engine_type}")

        return {"text": text, "duration": duration}
