from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_seed: int = Field(default=42, alias="GROQ_SEED")
    eval_offline_mode: bool = False

    data_dir: Path = REPO_ROOT / "data"
    knowledge_base_dir: Path = REPO_ROOT / "knowledge-base"

    # Deterministic generation for production tasks
    temperature: float = 0.0


settings = Settings()
