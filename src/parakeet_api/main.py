import argparse
import logging
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from platformdirs import user_data_dir, user_log_dir
from pydantic import BaseModel, Field

from . import config, download_utils
from .stt import STTEngine

app = FastAPI(
    title="Parakeet API",
    description="OpenAI Whisper-compatible API endpoint for Parakeet STT models",
    version="0.1.0",
)

security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
):
    # Only verify if API_KEY is set in settings
    if config.settings.server.api_key:
        if not credentials or credentials.credentials != config.settings.server.api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return credentials


logging.basicConfig(level=logging.INFO)

stt_engine = None


def get_stt_engine() -> STTEngine:
    global stt_engine
    if stt_engine is None:
        stt_engine = STTEngine()
    return stt_engine


class TranscriptionResponse(BaseModel):
    text: str


class TranscriptionWord(BaseModel):
    word: str = Field(..., description="The word text")
    start: float = Field(..., description="Start time of the word in seconds")
    end: float = Field(..., description="End time of the word in seconds")
    # Source: https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml
    # Schema: TranscriptionWord


class TranscriptionSegment(BaseModel):
    id: int = Field(..., description="Segment identifier")
    seek: int = Field(..., description="Seek index within the audio")
    start: float = Field(..., description="Start time of the segment in seconds")
    end: float = Field(..., description="End time of the segment in seconds")
    text: str = Field(..., description="Segment text")
    tokens: list[int] = Field(..., description="Token identifiers")
    temperature: float = Field(..., description="Temperature used for generation")
    avg_logprob: float = Field(..., description="Average log probability")
    compression_ratio: float = Field(..., description="Compression ratio")
    no_speech_prob: float = Field(..., description="Probability of no speech")
    # Source: https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml
    # Schema: TranscriptionSegment


class TranscriptionVerboseResponse(BaseModel):
    task: str = Field(..., description="Task type (transcribe)")
    language: str = Field(..., description="Detected language")
    duration: float = Field(..., description="Audio duration in seconds")
    text: str = Field(..., description="Transcribed text")
    words: list[TranscriptionWord] | None = Field(
        None, description="Word-level timestamps"
    )
    segments: list[TranscriptionSegment] | None = Field(
        None, description="Segment-level details"
    )
    # Source: https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml
    # Schema: TranscriptionVerboseResponse


@app.get("/")
async def root():
    return {
        "message": "Parakeet API is running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def format_timestamp(seconds: float, is_vtt: bool = False) -> str:
    td = float(seconds)
    hours = int(td // 3600)
    minutes = int((td % 3600) // 60)
    secs = td % 60
    millis = int((secs - int(secs)) * 1000)
    sep = "." if is_vtt else ","
    return f"{hours:02}:{minutes:02}:{int(secs):02}{sep}{millis:03}"


def format_srt(text: str, duration: float) -> str:
    return f"1\n{format_timestamp(0.0)} --> {format_timestamp(duration)}\n{text}\n\n"


def format_vtt(text: str, duration: float) -> str:
    return f"WEBVTT\n\n1\n{format_timestamp(0.0, True)} --> {format_timestamp(duration, True)}\n{text}\n\n"


def generate_words_and_segments(
    text: str, duration: float, language: str, temperature: float
) -> tuple[list[TranscriptionWord] | None, list[TranscriptionSegment]]:
    words = text.split()
    num_words = len(words)
    word_duration = duration / num_words if num_words > 0 else 0

    words_data = [
        TranscriptionWord(
            word=word,
            start=round(i * word_duration, 2),
            end=round((i + 1) * word_duration, 2),
        )
        for i, word in enumerate(words)
    ]

    segments_data = [
        TranscriptionSegment(
            id=0,
            seek=0,
            start=0.0,
            end=round(duration, 2),
            text=text,
            tokens=list(range(len(words) + 10)),
            temperature=temperature,
            avg_logprob=-0.5,
            compression_ratio=1.2,
            no_speech_prob=0.01,
        )
    ]

    return words_data, segments_data


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    request: Request,
    file: Annotated[UploadFile, File()],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        list[str] | None, Form(alias="timestamp_granularities[]")
    ] = None,
    _auth: Annotated[
        HTTPAuthorizationCredentials | None, Depends(verify_api_key)
    ] = None,
):
    start_req = time.perf_counter()
    if file is None or file.filename is None:
        raise HTTPException(status_code=400, detail="No file provided")

    if file.content_type and "audio" not in file.content_type:
        logging.warning(f"Unexpected content type: {file.content_type}")

    audio_data = await file.read()
    logging.info(
        f"Received transcription request: filename={file.filename}, size={len(audio_data)} bytes, format={response_format}"
    )

    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    engine = get_stt_engine()

    try:
        result = engine.transcribe(audio_data)
        text = result["text"]
        duration = result["duration"]
    except RuntimeError as e:
        logging.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    lang = language or "en"

    # ... response generation logic ...
    resp_obj = None

    if response_format == "json":
        resp_obj = TranscriptionResponse(text=text)
    elif response_format == "text":
        resp_obj = Response(content=text, media_type="text/plain")
    elif response_format == "srt":
        resp_obj = Response(content=format_srt(text, duration), media_type="text/plain")
    elif response_format == "vtt":
        resp_obj = Response(content=format_vtt(text, duration), media_type="text/plain")
    elif response_format == "verbose_json":
        granularities = timestamp_granularities or ["segment"]
        include_words = "word" in granularities
        include_segments = "segment" in granularities
        words_data, segments_data = generate_words_and_segments(
            text, duration, lang, temperature
        )

        resp_obj = TranscriptionVerboseResponse(
            task="transcribe",
            language=lang,
            duration=duration,
            text=text,
            words=words_data if include_words else None,
            segments=segments_data if include_segments else None,
        )
    elif response_format == "chunked_json":
        resp_obj = JSONResponse(
            content={"text": text},
            media_type="application/x-ndjson",
        )
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported response format: {response_format}"
        )

    elapsed_req = (time.perf_counter() - start_req) * 1000
    logging.info(f"Request handled: total_time={elapsed_req:.2f}ms")
    return resp_obj


@app.post("/v1/audio/transcriptions/raw")
async def transcribe_audio_raw(
    request: Request,
    model: Annotated[str | None, Query()] = None,
    language: Annotated[str | None, Query()] = None,
    prompt: Annotated[str | None, Query()] = None,
    response_format: Annotated[str, Query()] = "json",
    temperature: Annotated[float, Query()] = 0.0,
    timestamp_granularities: Annotated[
        list[str] | None, Query(alias="timestamp_granularities[]")
    ] = None,
    _auth: Annotated[
        HTTPAuthorizationCredentials | None, Depends(verify_api_key)
    ] = None,
):
    start_req = time.perf_counter()
    audio_data = await request.body()

    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="No audio data provided")

    engine = get_stt_engine()

    try:
        result = engine.transcribe(audio_data)
        text = result["text"]
        duration = result["duration"]
    except RuntimeError as e:
        logging.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    lang = language or "en"
    resp_obj = None

    if response_format == "json":
        resp_obj = TranscriptionResponse(text=text)
    elif response_format == "text":
        resp_obj = Response(content=text, media_type="text/plain")
    elif response_format == "srt":
        resp_obj = Response(content=format_srt(text, duration), media_type="text/plain")
    elif response_format == "vtt":
        resp_obj = Response(content=format_vtt(text, duration), media_type="text/plain")
    elif response_format == "verbose_json":
        granularities = timestamp_granularities or ["segment"]
        include_words = "word" in granularities
        include_segments = "segment" in granularities
        words_data, segments_data = generate_words_and_segments(
            text, duration, lang, temperature
        )

        resp_obj = TranscriptionVerboseResponse(
            task="transcribe",
            language=lang,
            duration=duration,
            text=text,
            words=words_data if include_words else None,
            segments=segments_data if include_segments else None,
        )
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported response format: {response_format}"
        )

    elapsed_req = (time.perf_counter() - start_req) * 1000
    logging.info(f"Request handled: total_time={elapsed_req:.2f}ms")
    return resp_obj


def get_executable_command() -> list[str]:
    """Resolve the command to run the parakeet-api executable."""
    # 1. Prefer sys.argv[0] if it's a valid executable
    if sys.argv[0]:
        argv0_path = Path(sys.argv[0]).resolve()
        if argv0_path.is_file() and os.access(argv0_path, os.X_OK):
            return [str(argv0_path)]

    # 2. Check PATH environment
    exe = shutil.which("parakeet-api")
    if exe:
        return [str(Path(exe).resolve())]

    # 3. Fallback to current python -m
    return [sys.executable, "-m", "parakeet_api.main"]


def install_daemon_action(args):
    """Install and enable background service (launchd for macOS, systemd for Linux)."""
    target_os = args.os or sys.platform
    if target_os == "darwin":
        target_os = "macos"
    elif target_os == "linux":
        target_os = "linux"
    else:
        print(f"Unsupported OS for daemon installation: {target_os}")
        sys.exit(1)

    base_dir = Path(user_data_dir("parakeet-api"))
    models_dir = base_dir / "models"
    log_dir = Path(user_log_dir("parakeet-api"))
    env_file = base_dir / ".env"

    # 1. Create directories with secure permissions
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        base_dir.chmod(0o700)  # Owner only
    except OSError:
        pass

    models_dir.mkdir(parents=True, exist_ok=True)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    if not env_file.exists():
        # Owner read/write only
        env_file.touch(mode=0o600)

        # Try to find a template from source or nearby
        example_env = Path(".env.example")
        if not example_env.exists():
            # Check if we are in a source repo (parent of parent)
            pkg_root = Path(__file__).resolve().parent.parent.parent
            example_env = pkg_root / ".env.example"

        if example_env.exists():
            shutil.copy(example_env, env_file)
            print(f"Created configuration file from template: {env_file}")
        else:
            env_file.write_text(
                "# Parakeet API Configuration\n# See https://github.com/likeablob/parakeet-api for options\n",
                encoding="utf-8",
            )
            print(f"Created configuration file: {env_file}")

    exe_cmd = get_executable_command()
    label = "org.parakeet-api"

    if target_os == "macos":
        plist_path = Path.home() / f"Library/LaunchAgents/{label}.plist"
        # Standard paths for macOS
        path_env = f"/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:{Path.home() / '.local/bin'}"

        plist_data = {
            "Label": label,
            "ProgramArguments": exe_cmd + ["serve"],
            "WorkingDirectory": str(base_dir),
            "RunAtLoad": True,
            "KeepAlive": True,
            "EnvironmentVariables": {
                "PATH": path_env,
                "PYTHONUNBUFFERED": "1",
            },
        }
        if log_dir:
            plist_data["StandardOutPath"] = str(log_dir / "stdout.log")
            plist_data["StandardErrorPath"] = str(log_dir / "stderr.log")

        try:
            with open(plist_path, "wb") as f:
                plistlib.dump(plist_data, f)
            print(f"Created plist file: {plist_path}")

            uid = os.getuid()
            # Try to unload first (idempotency)
            subprocess.run(
                ["launchctl", "bootout", f"gui/{uid}", str(plist_path)],
                check=False,
                stderr=subprocess.DEVNULL,
            )
            # Bootstrap the service
            subprocess.run(
                ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)], check=True
            )
            print(f"\nSuccessfully installed and started {label} daemon.")
            if log_dir:
                print("Log files:")
                print(f"  - {log_dir / 'stdout.log'}")
                print(f"  - {log_dir / 'stderr.log'}")
                print(f"To tail logs: tail -f {log_dir / 'stderr.log'}")
            print("\nNext steps:")
            print(f"  1. Configure your settings in {env_file}")
            print(
                f"  2. Restart service after changes: launchctl kickstart -k gui/{uid}/{label}"
            )
            print("  3. Download models: parakeet-api download sherpa")

        except Exception as e:
            print(f"Failed to install macOS daemon: {e}")
            sys.exit(1)

    elif target_os == "linux":
        service_path = Path.home() / f".config/systemd/user/{label}.service"
        service_path.parent.mkdir(parents=True, exist_ok=True)

        exec_start = " ".join(shlex.quote(arg) for arg in exe_cmd + ["serve"])
        # Standard paths for Linux
        path_env = f"{Path.home() / '.local/bin'}:/usr/local/bin:/usr/bin:/bin"

        service_content = f"""[Unit]
Description=Parakeet STT API Daemon
After=network.target

[Service]
Type=simple
ExecStart={exec_start}
WorkingDirectory={base_dir}
Restart=always
Environment=PYTHONUNBUFFERED=1
Environment="PATH={path_env}"

[Install]
WantedBy=default.target
"""
        try:
            service_path.write_text(service_content, encoding="utf-8")
            print(f"Created service file: {service_path}")

            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            subprocess.run(
                ["systemctl", "--user", "enable", "--now", f"{label}.service"],
                check=True,
            )

            print(f"\nSuccessfully installed and started {label} daemon.")
            print(f"To check logs: journalctl --user -u {label} -f")
            print("\nNext steps:")
            print(f"  1. Configure your settings in {env_file}")
            print(
                f"  2. Restart service after changes: systemctl --user restart {label}"
            )
            print("  3. Download models: parakeet-api download sherpa")

        except Exception as e:
            print(f"Failed to install Linux daemon: {e}")
            sys.exit(1)


def uninstall_daemon_action(args):
    """Uninstall background service."""
    target_os = args.os or sys.platform
    if target_os == "darwin":
        target_os = "macos"
    elif target_os == "linux":
        target_os = "linux"
    else:
        print(f"Unsupported OS for daemon uninstallation: {target_os}")
        sys.exit(1)

    label = "org.parakeet-api"

    if target_os == "macos":
        plist_path = Path.home() / f"Library/LaunchAgents/{label}.plist"
        if plist_path.exists():
            print(f"Uninstalling {label} daemon...")
            uid = os.getuid()
            subprocess.run(
                ["launchctl", "bootout", f"gui/{uid}", str(plist_path)], check=False
            )
            plist_path.unlink()
            print(f"Removed plist file: {plist_path}")
        else:
            print(f"No daemon found at {plist_path}")

    elif target_os == "linux":
        service_path = Path.home() / f".config/systemd/user/{label}.service"
        if service_path.exists():
            print(f"Uninstalling {label} daemon...")
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", f"{label}.service"],
                check=False,
            )
            service_path.unlink()
            print(f"Removed service file: {service_path}")
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        else:
            print(f"No daemon found at {service_path}")


def main():
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="parakeet-api",
        description="Parakeet STT API and model management CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # Serve command
    subparsers.add_parser("serve", help="Start the STT API server")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download STT models")
    dl_subparsers = download_parser.add_subparsers(
        dest="engine", help="STT engine to download for"
    )

    # Download Sherpa
    sherpa_parser = dl_subparsers.add_parser(
        "sherpa", help="Download Sherpa-ONNX model"
    )
    sherpa_parser.add_argument(
        "--url",
        type=str,
        default=download_utils.SHERPA_DEFAULT_URL,
        help=f"Model URL (default: {download_utils.SHERPA_DEFAULT_URL})",
    )
    sherpa_parser.add_argument(
        "--out",
        type=str,
        default=config.settings.stt.models_dir,
        help=f"Output base directory (default: {config.settings.stt.models_dir})",
    )

    # Download MLX
    mlx_parser = dl_subparsers.add_parser("mlx", help="Download MLX model")
    mlx_parser.add_argument(
        "--id",
        type=str,
        default=download_utils.MLX_DEFAULT_MODEL,
        help=f"HuggingFace Repo ID (default: {download_utils.MLX_DEFAULT_MODEL})",
    )
    mlx_parser.add_argument(
        "--out",
        type=str,
        default=config.settings.stt.models_dir,
        help=f"Output base directory (default: {config.settings.stt.models_dir})",
    )

    # Install daemon
    install_parser = subparsers.add_parser(
        "install-daemon", help="Install background service"
    )
    install_parser.add_argument(
        "--os", choices=["macos", "linux"], help="Target OS (default: auto)"
    )

    # Uninstall daemon
    uninstall_parser = subparsers.add_parser(
        "uninstall-daemon", help="Uninstall background service"
    )
    uninstall_parser.add_argument(
        "--os", choices=["macos", "linux"], help="Target OS (default: auto)"
    )

    # Parse initial command
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Check the command
    first_arg = sys.argv[1]

    if first_arg == "serve":
        # Remove 'serve' from sys.argv so Pydantic-settings doesn't see it
        # but keep the rest for CliSettingsSource
        sys.argv.pop(1)
        config.settings = config.CLISettings()

        if config.settings.server.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Debug logging enabled.")

        import uvicorn

        uvicorn.run(
            app, host=config.settings.server.host, port=config.settings.server.port
        )
    elif first_arg == "download":
        args = parser.parse_args()
        output_base = Path(args.out).resolve()

        if args.engine == "sherpa":
            download_utils.download_sherpa(args.url, output_base)
        elif args.engine == "mlx":
            download_utils.download_mlx(args.id, output_base)
        else:
            download_parser.print_help()
    elif first_arg == "install-daemon":
        args = parser.parse_args()
        install_daemon_action(args)
    elif first_arg == "uninstall-daemon":
        args = parser.parse_args()
        uninstall_daemon_action(args)
    else:
        # If it's not a known command, maybe it's just --help or legacy behavior?
        # But per request, default is help.
        # However, we should handle --help
        if first_arg in ("-h", "--help"):
            parser.print_help()
        else:
            print(f"Unknown command: {first_arg}")
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
