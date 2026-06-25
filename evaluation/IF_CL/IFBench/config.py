"""Configuration for IFBench using pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchmarkSettings(BaseSettings):
    """Settings for running IFBench benchmarks."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Configuration
    api_base: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for the OpenAI-compatible API",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authentication",
    )
    model: str = Field(
        default="",
        description="Model name to use for generation",
    )

    # Generation Parameters
    temperature: float = Field(
        default=1.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens to generate",
    )
    seed: int | None = Field(
        default=42,
        description="Random seed for reproducibility (None for random)",
    )

    top_p: float = Field(default=0.95, description="Top-p sampling")
    top_k: int = Field(default=20, description="Top-k sampling")
    min_p: float = Field(default=0.0, description="Min-p sampling")
    presence_penalty: float = Field(default=1.5, description="Presence penalty")
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")

    # Benchmark Parameters
    input_file: str = Field(
        default="../../datasets/data/ifbench/IFBench_test.jsonl",
        description="Path to IFBench test file",
    )
    output_file: str = Field(
        default="data/responses.jsonl",
        description="Output file for responses",
    )
    workers: int = Field(
        default=8,
        description="Number of parallel workers",
    )


def get_settings() -> BenchmarkSettings:
    """Load settings from environment and .env file."""
    return BenchmarkSettings()
