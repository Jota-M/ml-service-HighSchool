"""
app/routers/prediccion.py — Endpoints del microservicio ML (Tiempo Real)

POST /api/v1/predecir          → predicción semanal en tiempo real
POST /api/v1/simular           → simulación de escenarios
POST /api/v1/reentrenar        → reentrenar modelo
GET  /api/v1/health            → estado del servicio
GET  /api/v1/modelo/info       → metadata del modelo
"""

import logging, subprocess, sys, json
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, status

from app.config import get_settings
from app.schemas.prediccion import (
    DatosTiempoRealRequest, SimulacionRequest,
    PrediccionResponse, SimulacionResponse,
    ReentrenarResponse, HealthResponse,
)
from app.services.ml_service import modelo_manager, predecir_tiempo_real
from app.services.gemini_service import analizar_prediccion, analizar_escenarios, verificar_disponibilidad
from app.services.simulate_service import simular_escenarios

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predecir",
    response_model=PrediccionResponse,
    summary="Predicción en tiempo real — snapshot semanal del estudiante",
    description="""
    Predice el rendimiento final del trimestre con los datos disponibles hasta la semana actual.
    
    Se llama automáticamente cada vez que se registra:
    - Una nueva práctica o examen
    - Asistencia de la semana
    - Una conducta negativa o positiva
    
    La confianza de la predicción aumenta conforme avanza el trimestre.
    """,
)
async def predecir(
    datos: DatosTiempoRealRequest,
    usar_xgboost:   bool = Query(default=True,  description="True=XGBoost (recomendado), False=Random Forest"),
    incluir_gemini: bool = Query(default=True,  description="Incluir análisis narrativo de Gemini"),
):
    if not modelo_manager.esta_listo:
        raise HTTPException(status_code=503, detail="Modelos ML no cargados. Ejecuta train.py primero.")

    try:
        resultado_ml = predecir_tiempo_real(datos, usar_xgboost)

        analisis, gemini_ok = None, False
        if incluir_gemini:
            analisis  = await analizar_prediccion(datos, resultado_ml)
            gemini_ok = analisis is not None

        return PrediccionResponse(
            estudiante_id     = datos.estudiante_id,
            materia           = datos.materia,
            codigo_materia    = datos.codigo_materia,
            trimestre         = datos.trimestre,
            semana_actual     = datos.semana,
            modelo            = resultado_ml,
            analisis          = analisis,
            modelo_usado      = "xgboost" if usar_xgboost else "random_forest",
            gemini_disponible = gemini_ok,
        )
    except Exception as e:
        logger.error(f"Error predicción estudiante {datos.estudiante_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/simular",
    response_model=SimulacionResponse,
    summary="Simulación de escenarios de intervención",
    description="""
    Simula hasta 5 escenarios hipotéticos y muestra cómo cambiaría el riesgo.
    
    Ejemplos de escenarios útiles:
    - ¿Qué pasa si mejora su asistencia al 90%?
    - ¿Qué pasa si saca 70 en la próxima práctica?
    - ¿Qué pasa si elimina las conductas negativas?
    """,
)
async def simular(
    request: SimulacionRequest,
    usar_xgboost:   bool = Query(default=True),
    incluir_gemini: bool = Query(default=True),
):
    if not modelo_manager.esta_listo:
        raise HTTPException(status_code=503, detail="Modelos ML no cargados.")

    try:
        situacion_actual, resultados = await simular_escenarios(
            request.datos_base, request.escenarios, usar_xgboost
        )

        recomendacion = None
        if incluir_gemini:
            recomendacion = await analizar_escenarios(
                request.datos_base, situacion_actual, resultados
            )

        return SimulacionResponse(
            estudiante_id        = request.datos_base.estudiante_id,
            materia              = request.datos_base.materia,
            semana_actual        = request.datos_base.semana,
            situacion_actual     = situacion_actual,
            escenarios           = resultados,
            recomendacion_gemini = recomendacion,
        )
    except Exception as e:
        logger.error(f"Error simulación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reentrenar", response_model=ReentrenarResponse, summary="Reentrenar el modelo")
async def reentrenar():
    settings = get_settings()
    inicio   = datetime.now()
    try:
        res = subprocess.run(
            [sys.executable, "models/train.py"],
            capture_output=True, text=True, timeout=300
        )
        if res.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Error en train.py: {res.stderr[:500]}")

        modelo_manager.cargar(settings.models_path)
        duracion = (datetime.now() - inicio).total_seconds()

        metricas = {}
        mp = settings.models_path / "metadata.json"
        if mp.exists():
            with open(mp) as f: metricas = json.load(f).get("xgb_riesgo", {})

        return ReentrenarResponse(
            exitoso=True, registros_usados=0,
            duracion_segundos=round(duracion, 2),
            metricas=metricas,
            mensaje=f"Modelo reentrenado en {duracion:.1f}s"
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Entrenamiento excedió tiempo límite.")


@router.get("/health", response_model=HealthResponse, summary="Estado del servicio")
async def health():
    gemini_ok = await verificar_disponibilidad()
    st = "ok"
    if not modelo_manager.esta_listo: st = "degradado — modelos no cargados"
    elif not gemini_ok: st = "parcial — ML ok, Gemini no disponible"
    return HealthResponse(
        status=st, modelos_cargados=modelo_manager.esta_listo,
        gemini_disponible=gemini_ok, version_modelo=modelo_manager.version,
        tipo_modelo="tiempo_real_semanal"
    )


@router.get("/modelo/info", summary="Metadata del modelo actual")
async def info_modelo():
    settings = get_settings()
    mp = settings.models_path / "metadata.json"
    if not mp.exists():
        raise HTTPException(status_code=404, detail="No hay metadata. Ejecuta train.py primero.")
    with open(mp) as f: return json.load(f)