import hashlib
import io
import logging
import subprocess
import tempfile
import time
import wave
from collections import OrderedDict
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
        self.recognizer: sherpa_onnx.OfflineRecognizer | None = None
        self.engine_type: str | None = None
        self.model_dir: Path | None = None
        self.tokens_path: Path | None = None
        self._hotwords_cache: OrderedDict[
            str, tuple[sherpa_onnx.OfflineRecognizer, str]
        ] = OrderedDict()

        logging.info(f"STTEngine CWD: {Path.cwd()}")
        logging.info(f"HAS_MLX: {HAS_MLX}")

        if HAS_MLX:
            self._init_mlx()

        if not self.recognizer:
            self._init_sherpa()

        if not self.recognizer:
            logging.error("No STT engine (mlx or sherpa) initialized.")

    def _parse_hotwords(self, hotwords_csv: str) -> str:
        """Parse CSV hotwords string to hotwords.txt format.

        Input: "Phoebe,OpenAI:2.5,GPT-4"
        Output: "Phoebe\nOpenAI :2.5\nGPT-4"
        """
        lines = []
        for item in hotwords_csv.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                word, score = item.rsplit(":", 1)
                word = word.strip()
                score = score.strip()
                lines.append(f"{word} :{score}")
            else:
                lines.append(item)
        return "\n".join(lines)

    def _create_hotwords_recognizer(self, hotwords_file: str):
        """Create recognizer with hotwords support using modified_beam_search."""
        if self.engine_type != "sherpa_offline":
            raise RuntimeError("Hotwords only supported for Sherpa-ONNX engine")

        if not self.model_dir or not self.tokens_path:
            raise RuntimeError("Model directory or tokens path not initialized")

        model_dir = self.model_dir

        def find_onnx(name):
            for ext in [".onnx", ".int8.onnx"]:
                p = model_dir / f"{name}{ext}"
                if p.exists():
                    return str(p)
            return None

        encoder = find_onnx("encoder")
        decoder = find_onnx("decoder")
        joiner = find_onnx("joiner")

        if not (encoder and decoder and joiner):
            raise RuntimeError(
                "Hotwords requires Transducer model (encoder/decoder/joiner)"
            )

        provider = settings.stt.sherpa.provider
        num_threads = settings.stt.sherpa.num_threads
        f_dim = 80

        model_type = self._detect_model_type(self.tokens_path)
        is_nemo = model_type == "nemo_transducer"
        modeling_unit = "bpe" if is_nemo else "cjkchar"

        bpe_vocab: str | None = None
        if is_nemo:
            bpe_vocab_in_model = model_dir / "bpe.vocab"
            if bpe_vocab_in_model.exists():
                bpe_vocab = str(bpe_vocab_in_model)
            else:
                raise RuntimeError(
                    f"bpe.vocab not found in model directory ({model_dir / 'bpe.vocab'}). "
                    "Hotwords support requires bpe.vocab for Parakeet TDT models. "
                    "Run: parakeet-api download sherpa --generate-bpe-vocab"
                )

        try:
            recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=str(self.tokens_path),
                num_threads=num_threads,
                sample_rate=16000,
                feature_dim=f_dim,
                decoding_method="modified_beam_search",
                provider=provider,
                debug=settings.server.debug,
                model_type=model_type,
                hotwords_file=hotwords_file,
                hotwords_score=settings.stt.sherpa.hotwords.default_score,
                modeling_unit=modeling_unit,
                bpe_vocab=bpe_vocab or "",
            )
            logging.info("Hotwords recognizer created successfully")
            return recognizer
        except Exception as e:
            raise RuntimeError(f"Failed to create hotwords recognizer: {e}") from e

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
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
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

    def _detect_model_type(self, tokens_path: Path) -> str:
        try:
            first_line = tokens_path.read_text(encoding="utf-8").split("\n")[0].strip()
            if first_line.startswith("<unk>"):
                return "nemo_transducer"
        except Exception:
            pass
        logging.debug("Model type not detected as NeMo, using auto-detect")
        return ""

    def _init_sherpa(self):
        base_dir = Path(settings.stt.models_dir)
        model_dir = base_dir / "sherpa" / settings.stt.sherpa.model_id
        self.model_dir = model_dir

        tokens_path = model_dir / "tokens.txt"
        self.tokens_path = tokens_path
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
                model_type = self._detect_model_type(tokens_path)
                logging.info(
                    f"Initializing Sherpa Transducer from {model_dir} with feature_dim={f_dim}, model_type='{model_type}'"
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
                    model_type=model_type,
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

    def transcribe(self, audio_bytes: bytes, hotwords: str | None = None) -> dict:
        if not self.recognizer:
            raise RuntimeError("STT engine not initialized (no mlx or sherpa found)")

        start_total = time.perf_counter()

        pcm_raw, sample_rate = self._convert_to_pcm_16k(audio_bytes)
        duration = len(pcm_raw) / (2 * sample_rate)
        text = ""

        start_inference = time.perf_counter()

        if self.engine_type == "mlx":
            if hotwords:
                logging.warning("Hotwords not supported for MLX engine, ignoring")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
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
            try:
                samples = (
                    np.frombuffer(pcm_raw, dtype=np.int16).astype(np.float32) / 32768.0
                )

                if self.engine_type == "sherpa_offline":
                    if hotwords:
                        hotwords_content = self._parse_hotwords(hotwords)
                        cache_size = settings.stt.sherpa.hotwords.cache_size
                        cache_key = hashlib.md5(hotwords_content.encode()).hexdigest()
                        cached = cache_size > 0 and cache_key in self._hotwords_cache

                        if cached:
                            hw_recognizer, hw_path = self._hotwords_cache[cache_key]
                        else:
                            if cache_size > 0:
                                while len(self._hotwords_cache) >= cache_size:
                                    old_key, (old_rec, old_path) = (
                                        self._hotwords_cache.popitem(last=False)
                                    )
                                    Path(old_path).unlink(missing_ok=True)

                            hw_file = tempfile.NamedTemporaryFile(
                                mode="w", suffix=".txt", delete=False
                            )
                            try:
                                hw_file.write(hotwords_content)
                                hw_path = hw_file.name
                            finally:
                                hw_file.close()

                            try:
                                hw_recognizer = self._create_hotwords_recognizer(
                                    hw_path
                                )
                            except Exception:
                                Path(hw_path).unlink(missing_ok=True)
                                raise

                            if cache_size > 0:
                                self._hotwords_cache[cache_key] = (
                                    hw_recognizer,
                                    hw_path,
                                )

                        try:
                            stream = hw_recognizer.create_stream()
                            stream.accept_waveform(sample_rate, samples)
                            hw_recognizer.decode_stream(stream)
                            text = stream.result.text.strip()
                        finally:
                            if not cached:
                                Path(hw_path).unlink(missing_ok=True)
                    else:
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
            f"STT: engine={self.engine_type}, inference={inference_elapsed:.2f}ms, total={total_elapsed:.2f}ms, hotwords={hotwords is not None}"
        )

        if not text and self.engine_type is None:
            raise RuntimeError(f"Unknown STT engine type: {self.engine_type}")

        return {"text": text, "duration": duration}
