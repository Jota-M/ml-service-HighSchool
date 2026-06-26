"""
main.py — Entrypoint del microservicio ML v8.1

Arranca con:
    uvicorn main:app --reload --port 8000

Swagger UI disponible en:
    http://localhost:8000/docs

Cambios respecto a v7:
  - Descripción actualizada: 31 features, modelos con historial
  - DEBUG de Gemini key movido a un logger en lugar de print
    para no contaminar stdout en producción
  - CORS: sin cambios
"""

import logging
from contextlib import asynccontextmanager

# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
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

    # Debug de configuración — solo en INFO, no en print
    key_preview = settings.gemini_api_key[:8] if settings.gemini_api_key else "NO CONFIGURADA"
    logger.info(f"Gemini key: '{key_preview}...' | modelo: {settings.gemini_model}")

    cargado = modelo_manager.cargar(settings.models_path)
    if cargado:
        logger.info(
            f"✅ Modelos ML listos — "
            f"versión: {modelo_manager.version} | "
            f"features: {modelo_manager.n_features}"
        )
    else:
        logger.warning("⚠️  Modelos no disponibles — ejecuta models/train_v8.py")

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
## 🎓 Microservicio de Predicción de Rendimiento Estudiantil v8.1

Sistema de IA híbrido para instituciones educativas bolivianas.

### Capacidades
- **Predicción** de riesgo de reprobar basada en datos del trimestre actual
- **Historial** intertrimestral — el modelo recuerda los trimestres anteriores
- **Observaciones** pedagógicas — conducta, socioemocional, logros
- **Correlación** entre materias — detecta dificultad generalizada
- **Estimación** de nota final del trimestre
- **Simulación** de escenarios de intervención pedagógica
- **Análisis narrativo** con Gemini AI — planes de recuperación con contexto histórico

### Modelos disponibles
- Random Forest
- XGBoost (recomendado)

### Features del modelo
- 14 features legacy (asistencia, notas, tendencia)
- 7 features de historial intertrimestral
- 5 features de observaciones pedagógicas
- 2 features de correlación entre materias
- 2 features de nivel educativo y carga horaria
- 1 feature de régimen de ponderación ministerial

**Total: 29 features**

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
        "servicio":  settings.app_name,
        "version":   settings.app_version,
        "modelo":    modelo_manager.version,
        "features":  modelo_manager.n_features,
        "docs":      "/docs",
        "health":    "/api/v1/health",
    }