"""
main.py — Entrypoint del microservicio ML

Arranca con:
    uvicorn main:app --reload --port 8000

Swagger UI disponible en:
    http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
# DEBUG TEMPORAL — sacar después
settings = get_settings()
print(f"DEBUG GEMINI KEY: '{settings.gemini_api_key[:8]}...' modelo: {settings.gemini_model}")
from app.services.ml_service import modelo_manager
from app.routers import prediccion

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# LIFESPAN — startup / shutdown
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──
    settings = get_settings()
    logger.info(f"🚀 Iniciando {settings.app_name} v{settings.app_version}")

    cargado = modelo_manager.cargar(settings.models_path)
    if cargado:
        logger.info("✅ Modelos ML listos")
    else:
        logger.warning("⚠️  Modelos no disponibles — ejecuta models/train.py")

    yield

    # ── SHUTDOWN ──
    logger.info("🛑 Apagando microservicio ML...")


# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title       = settings.app_name,
    version     = settings.app_version,
    description = """
## 🎓 Microservicio de Predicción de Rendimiento Estudiantil

Sistema de IA híbrido para instituciones educativas bolivianas.

### Capacidades
- **Predicción** de riesgo de reprobar T3 basada en T1 y T2
- **Estimación** de nota final del trimestre
- **Simulación** de escenarios de intervención pedagógica
- **Análisis narrativo** con Gemini AI

### Modelos disponibles
- Random Forest (93.97% accuracy)
- XGBoost (94.56% accuracy)

### Escala boliviana
- `ED` En Desarrollo: 0–50 (reprobado)
- `DA` Desarrollo Aceptable: 51–68
- `DO` Desarrollo Óptimo: 69–84
- `DP` Desarrollo Pleno: 85–100
    """,
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS — permite llamadas desde Express
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:3001"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Routers
app.include_router(
    prediccion.router,
    prefix = "/api/v1",
    tags   = ["Predicción"],
)


@app.get("/", tags=["Root"])
async def root():
    return {
        "servicio": settings.app_name,
        "version":  settings.app_version,
        "docs":     "/docs",
        "health":   "/api/v1/health",
    }