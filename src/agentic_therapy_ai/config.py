from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vlm_model_id: str = Field(default="Qwen/Qwen2-VL-72B-Instruct")
    llm_model_id: str = Field(default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    output_dir: Path = Field(default=Path("outputs"))
    tensor_parallel_size: int = Field(default=4, ge=1)
    max_new_tokens_vlm: int = Field(default=900, ge=128)
    max_new_tokens_llm: int = Field(default=700, ge=128)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    dtype: str = Field(default="bfloat16")


settings = RuntimeSettings()
