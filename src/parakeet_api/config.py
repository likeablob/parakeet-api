import os

from platformdirs import user_data_dir
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

SHERPA_DEFAULT_MODEL = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
MLX_DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
DEFAULT_DATA_DIR = user_data_dir("parakeet-api")
DEFAULT_MODELS_DIR = os.path.join(DEFAULT_DATA_DIR, "models")


class SherpaSettings(BaseModel):
    provider: str = Field(
        default="cpu",
        description="ONNX Runtime provider (Env: STT__SHERPA__PROVIDER)",
    )
    num_threads: int = Field(
        default=2,
        description="Threads for STT (Env: STT__SHERPA__NUM_THREADS)",
    )
    model_id: str = Field(
        default=SHERPA_DEFAULT_MODEL,
        description="Model directory name (Env: STT__SHERPA__MODEL_ID)",
    )


class MLXSettings(BaseModel):
    model_id: str = Field(
        default=MLX_DEFAULT_MODEL,
        description="MLX Repo ID (Env: STT__MLX__MODEL_ID)",
    )


class STTSettings(BaseModel):
    models_dir: str = Field(
        default=DEFAULT_MODELS_DIR,
        description="Base directory for models (Env: STT__MODELS_DIR)",
    )
    disable_conversion: bool = Field(
        default=False,
        description="Disable FFmpeg/pydub conversion logic (Env: STT__DISABLE_CONVERSION)",
    )
    sherpa: SherpaSettings = SherpaSettings()
    mlx: MLXSettings = MLXSettings()


class ServerSettings(BaseModel):
    host: str = Field(
        default="0.0.0.0",
        description="Bind host (Env: SERVER__HOST)",
    )
    port: int = Field(
        default=8816,
        description="Bind port (Env: SERVER__PORT)",
    )
    debug: bool = Field(
        default=False,
        description="Debug mode (Env: SERVER__DEBUG)",
    )
    api_key: str | None = Field(
        default=None,
        description="Optional API key for Bearer authentication (Env: SERVER__API_KEY)",
    )


class Settings(BaseSettings):
    """Base settings that only loads from environment and .env."""

    stt: STTSettings = STTSettings()
    server: ServerSettings = ServerSettings()

    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__", extra="ignore"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, dotenv_settings)


class CLISettings(Settings):
    """Special subclass for server execution to enable full Pydantic CLI parsing."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            CliSettingsSource(settings_cls, cli_parse_args=True),
            env_settings,
            dotenv_settings,
        )


settings = Settings()
