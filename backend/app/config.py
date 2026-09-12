from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "HOMADOS AI"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: str = "*"
    public_api_url: str = ""
    public_frontend_url: str = ""

    enable_ecapa: bool = True
    enable_aasist: bool = True
    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    ecapa_savedir: str = "models/ecapa_tdnn"
    aasist_savedir: str = "models/aasist"
    torch_device: str = "cpu"

    risk_spoof_weight: float = 0.55
    risk_identity_weight: float = 0.30
    risk_spectral_weight: float = 0.15
    alert_soft_threshold: float = 45
    alert_hard_threshold: float = 75

    @property
    def origin_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]

    @property
    def ecapa_dir(self) -> Path:
        return (ROOT / self.ecapa_savedir).resolve()

    @property
    def aasist_dir(self) -> Path:
        return (ROOT / self.aasist_savedir).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
