"""
app/config.py — Configuración centralizada del microservicio
"""

from pathlib import Path
# pyrefly: ignore [missing-import]
from functools import lru_cache
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings
# pyrefly: ignore [missing-import]
from pydantic import ConfigDict


class Settings(BaseSettings):
    # ── App ──────────────────────────────
    app_name:    str = "ML Predicción Estudiantil — Bolivia"
    app_version: str = "2.0.0"
    debug:       bool = False

    # ── Modelos ──────────────────────────
    models_path:      Path = Path("saved_models")
    modelo_principal: str  = "xgboost"   # xgboost | random_forest

    # ── Gemini ───────────────────────────
    gemini_api_key: str = ""
    gemini_model:   str = "gemini-1.5-flash"

    # extra="ignore" → cualquier variable del .env no declarada aquí
    # simplemente se descarta en lugar de lanzar ValidationError
    model_config = ConfigDict(env_file=".env", extra="ignore")
    
    # ── Youtube api key ────────────────────────────────────────────
    youtube_api_key: str = ""


@lru_cache()
def get_settings() -> Settings:
    return Settings()