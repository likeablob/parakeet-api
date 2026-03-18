# Parakeet API

OpenAI Whisper-compatible API endpoint for Parakeet STT models. (MLX for Apple Silicon, Sherpa-ONNX for others)

## Installation & Setup

The easiest way to install and run parakeet-api is using [uv](https://github.com/astral-sh/uv).

### 1. Install the CLI

**For Linux, Windows, or Intel Mac (Sherpa-ONNX / CPU):**

```bash
uv tool install parakeet-api
```

**For Apple Silicon (MLX):**

```bash
uv tool install "parakeet-api[mlx]"
```

### 2. Install System Dependencies

ffmpeg must be installed on your system for non-WAV audio support.

- **macOS:** brew install ffmpeg
- **Ubuntu/Debian:** sudo apt-get install ffmpeg

### 3. Download Models

Models are saved to your platform's standard data directory (e.g., ~/.local/share/parakeet-api/models).

#### Default Models

Download the default English/European model for your engine:

**Sherpa-ONNX:**

```bash
parakeet-api download sherpa
```

**MLX:**

```bash
parakeet-api download mlx
```

#### Custom Models

You can use different Parakeet models by specifying a URL or Repo ID.

**MLX:**

- [Parakeet - a mlx-community Collection](https://huggingface.co/collections/mlx-community/parakeet)

1. Download using the script with --id:
   ```bash
   parakeet-api download mlx --id mlx-community/parakeet-tdt_ctc-0.6b-ja
   ```
2. Update STT__MLX__MODEL_ID in your .env (or set as environment variable):
   ```env
   STT__MLX__MODEL_ID=mlx-community/parakeet-tdt_ctc-0.6b-ja
   ```

**Sherpa-ONNX:**

- [Sherpa-ONNX Pretrained Models](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html)
- [Sherpa-ONNX GitHub Releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)

1. Download using the script with --url:
   ```bash
   parakeet-api download sherpa --url https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8.tar.bz2
   ```
2. Update STT__SHERPA__MODEL_ID in your .env (or set as environment variable):
   ```env
   STT__SHERPA__MODEL_ID=sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8
   ```

### 4. Run the Server

```bash
parakeet-api serve
```

The API will be available at http://localhost:8816.

### 5. (Optional) Run as a Background Service

You can install parakeet-api as a background service (launchd on macOS, systemd on Linux).

```bash
parakeet-api install-daemon
```

This will create a service file and set up a configuration file at ~/.local/share/parakeet-api/.env.
To uninstall: `parakeet-api uninstall-daemon`

## Running with Docker (Sherpa-ONNX)

For Linux or CPU environments, you can use Docker and Docker Compose.

1. **Setup environment:**

   ```bash
   cp .env.example .env
   # Edit .env to set your SERVER__API_KEY and other settings
   ```

2. **Download the model using Docker:**

   ```bash
   # Download models into the ./models directory using the container
   docker compose run --rm api download sherpa --out /app/models
   ```

3. **Run with Docker Compose:**
   ```bash
   docker compose up --build
   ```

The ./models/ directory is bind-mounted into the container.

## Usage

### API Endpoints

#### POST /v1/audio/transcriptions

Transcribe audio to text using the OpenAI Whisper-compatible API format.

**Example with curl:**

```bash
curl -X POST "http://localhost:8816/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.wav" \
  -F "response_format=json"
```

#### POST /v1/audio/transcriptions/raw

Same as above but accepts raw audio bytes in the request body.

```bash
curl -X POST "http://localhost:8816/v1/audio/transcriptions/raw" \
  -H "Content-Type: audio/wav" \
  --data-binary @/path/to/audio.wav
```

### Supported Parameters

| Parameter                 | Type   | Default     | Description                             |
| ------------------------- | ------ | ----------- | --------------------------------------- |
| file                      | file   | -           | The audio file to transcribe.           |
| response_format           | string | json        | json, text, verbose_json, srt, vtt.     |
| timestamp_granularities[] | array  | ["segment"] | word, segment (used with verbose_json). |

> [!NOTE]
> **Limitations of Response Formats:**
> The current implementation provides simplified timestamp information. Consequently:
>
> - **srt / vtt**: Return a single segment covering the entire audio duration (0.0 to end).
> - **verbose_json**: Timestamps for words and segments are placeholders/estimations.

> [!NOTE]
> **Ignored Parameters:** The following parameters are accepted for compatibility with the OpenAI API but are currently **ignored**:
> model, language, prompt, temperature.

### Examples

Check the examples/ directory for client implementations:

- examples/client_requests.py: Basic transcription using requests.
- examples/client_openai_sdk.py: Using the official OpenAI Python SDK.

For full API compatibility details, refer to the [OpenAI Audio API Reference](https://platform.openai.com/docs/api-reference/audio) and their [OpenAPI specification](https://github.com/openai/openai-openapi).

## Development

### Setup from Source

1. **Clone the repository:**
   ```bash
   git clone https://github.com/likeablob/parakeet-api.git
   cd parakeet-api
   ```
2. **Install dependencies:**
   ```bash
   # Includes dev tools (ruff, ty, pytest) and optional mlx support
   uv sync --all-extras --dev
   ```
3. **Run:**
   ```bash
   uv run parakeet-api serve
   ```

### Code Quality & Tests

```bash
# Linting & Formatting
uv run ruff check .
uv run ruff format .

# Type Checking
uv run ty check src/ tests/

# Run Tests
uv run pytest tests/mock
uv run pytest tests/inference # Requires models
```

## License

MIT
