"""
app/routers/prediccion.py — Endpoints v8.5"""

import logging
import subprocess
import sys
import json
from datetime import datetime

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException, Query
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

from app.config import get_settings
from app.schemas.prediccion import (
    DatosTiempoRealRequest,
    SimulacionRequest,
    PrediccionResponse,
    SimulacionResponse,
    ReentrenarResponse,
    HealthResponse,
    ConfigPeriodo,
    RecursosExternosRequest,
    RecursosExternosResponse,
    SimulacionOptimoRequest,
    SimulacionOptimoResponse,
    AccionRequeridaResponse,
    SimulacionOptimoV2Request,
    SimulacionOptimoV2Response,
    EscenarioDetalladoResponse,
    EvaluacionPendienteResponse,
)
from app.services.ml_service import modelo_manager, predecir_tiempo_real
from app.services.gemini_service import (
    analizar_prediccion,
    analizar_escenarios,
    generar_plan_recuperacion,
    analizar_clase,
    verificar_disponibilidad,
    generar_recursos_externos,
)
from app.services.simulate_service import (
    simular_escenarios,
    calcular_optimo,
    calcular_optimo_v2,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────
# SCHEMAS LOCALES
# ─────────────────────────────────────────

class ClaseRequest(BaseModel):
    asignacion_docente_id: int           = Field(..., description="ID de la asignación docente")
    materia:               str           = Field(..., description="Nombre de la materia")
    semana_actual:         int           = Field(..., ge=1)
    config_periodo:        ConfigPeriodo = Field(default_factory=ConfigPeriodo)
    estudiantes:           list[DatosTiempoRealRequest] = Field(..., min_length=1)


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@router.post(
    "/predecir",
    response_model=PrediccionResponse,
    summary="Predicción semanal en tiempo real",
)
async def predecir(
    datos:          DatosTiempoRealRequest,
    usar_xgboost:   bool = Query(default=True),
    incluir_gemini: bool = Query(default=True),
    incluir_plan:   bool = Query(default=False),
):
    if not modelo_manager.esta_listo:
        raise HTTPException(
            status_code=503,
            detail="Modelos ML no cargados. Ejecutá train.py primero."
        )

    try:
        resultado_ml = predecir_tiempo_real(datos, usar_xgboost)

        analisis, gemini_ok = None, False
        if incluir_gemini:
            analisis  = await analizar_prediccion(datos, resultado_ml)
            gemini_ok = analisis is not None

        if incluir_plan and gemini_ok and resultado_ml.nivel_riesgo.value != "bajo":
            plan = await generar_plan_recuperacion(datos, resultado_ml)
            if analisis and plan:
                analisis.recomendaciones.append(
                    f"📋 Plan de recuperación generado: {plan.get('objetivo', '')}"
                )

        return PrediccionResponse(
            estudiante_id     = datos.estudiante_id,
            materia           = datos.materia,
            codigo_materia    = datos.codigo_materia,
            trimestre         = datos.trimestre,
            semana_actual     = datos.semana,
            total_semanas     = datos.config_periodo.total_semanas,
            modelo            = resultado_ml,
            analisis          = analisis,
            modelo_usado      = "xgboost" if usar_xgboost else "random_forest",
            gemini_disponible = gemini_ok,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicción estudiante {datos.estudiante_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/simular",
    response_model=SimulacionResponse,
    summary="Simulación de escenarios de intervención",
)
async def simular(
    request:        SimulacionRequest,
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
            total_semanas        = request.datos_base.config_periodo.total_semanas,
            situacion_actual     = situacion_actual,
            escenarios           = resultados,
            recomendacion_gemini = recomendacion,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error simulación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/simular/optimo",
    response_model=SimulacionOptimoResponse,
    summary="Simulación automática óptima — mínimo esfuerzo para una nota objetivo",
    description="""
    Calcula automáticamente qué acciones mínimas necesita el estudiante
    para alcanzar la nota objetivo (default: 51 = aprobar).

    Ordena las palancas disponibles por peso/retorno (SAB 45%, HAC 40%,
    asistencia impacto indirecto) y aplica de mayor a menor hasta cubrir
    el gap, o hasta agotar posibilidades.

    Modos de uso desde el cliente:
    - objetivo_nota=51  → "¿qué necesito para aprobar?"
    - objetivo_nota=70  → "¿qué necesito para una nota X?"
    - restricciones.bloquear_asistencia=true → "no puedo mejorar asistencia"
    """,
)
async def simular_optimo(
    request:      SimulacionOptimoRequest,
    usar_xgboost: bool = Query(default=True),
):
    if not modelo_manager.esta_listo:
        raise HTTPException(status_code=503, detail="Modelos ML no cargados.")

    try:
        resultado = await calcular_optimo(
            datos          = request.datos_base,
            objetivo_nota  = request.objetivo_nota,
            restricciones  = request.restricciones.model_dump(),
            usar_xgboost   = usar_xgboost,
        )

        return SimulacionOptimoResponse(
            objetivo_nota       = resultado.objetivo_nota,
            nota_actual         = resultado.nota_actual,
            nota_proyectada     = resultado.nota_proyectada,
            alcanzable          = resultado.alcanzable,
            acciones            = [
                AccionRequeridaResponse(
                    componente      = a.componente,
                    label           = a.label,
                    valor_actual    = a.valor_actual,
                    valor_necesario = a.valor_necesario,
                    delta           = a.delta,
                    impacto_nota    = a.impacto_nota,
                    dificultad      = a.dificultad,
                )
                for a in resultado.acciones
            ],
            nota_maxima_posible = resultado.nota_maxima_posible,
            mensaje             = resultado.mensaje,
            modo                = resultado.modo,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error simulación óptima: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/simular/optimo/v2",
    response_model=SimulacionOptimoV2Response,
    summary="Simulación óptima v2 — escenarios con desglose evaluación por evaluación",
    description="""
    Versión mejorada de /simular/optimo.
 
    Novedades v8.6:
    - practicas_restantes / examenes_restantes opcionales: el docente puede
      especificar cuántas evaluaciones quedan. Si no se mandan, el backend
      las estima por ritmo.
    - Techo realista garantiza siempre al menos 1 escenario alcanzable,
      estirando desde el rendimiento actual del estudiante hasta lo necesario
      para llegar a objetivo_nota.
    - 4 escenarios: camino mínimo, solo prácticas, priorizar examen, agresivo.
    """,
)
async def simular_optimo_v2(
    request: SimulacionOptimoV2Request,
    usar_xgboost: bool = Query(default=True),
):
    if not modelo_manager.esta_listo:
        raise HTTPException(status_code=503, detail="Modelos ML no cargados.")
 
    try:
        resultado = await calcular_optimo_v2(
            datos=request.datos_base,
            objetivo_nota=request.objetivo_nota,
            restricciones=request.restricciones.model_dump(),
            usar_xgboost=usar_xgboost,
            practicas_restantes_input=request.practicas_restantes,
            examenes_restantes_input=request.examenes_restantes,
        )
 
        return SimulacionOptimoV2Response(
            objetivo_nota=resultado.objetivo_nota,
            nota_actual=resultado.nota_actual,
            nota_maxima_posible=resultado.nota_maxima_posible,
            semanas_restantes=resultado.semanas_restantes,
            practicas_restantes_est=resultado.practicas_restantes_est,
            examenes_restantes_est=resultado.examenes_restantes_est,
            techo_practicas=resultado.techo_practicas,
            techo_examenes=resultado.techo_examenes,
            escenarios=[
                EscenarioDetalladoResponse(
                    id=e.id, titulo=e.titulo, descripcion=e.descripcion,
                    evaluaciones=[
                        EvaluacionPendienteResponse(
                            numero=ev.numero, tipo=ev.tipo,
                            nota_objetivo=ev.nota_objetivo,
                            es_alcanzable=ev.es_alcanzable,
                        )
                        for ev in e.evaluaciones
                    ],
                    nota_proyectada=e.nota_proyectada,
                    alcanzable=e.alcanzable,
                    porcentaje_exito=e.porcentaje_exito,
                    mensaje=e.mensaje,
                )
                for e in resultado.escenarios
            ],
            ya_alcanza=resultado.ya_alcanza,
            imposible=resultado.imposible,
            mensaje_general=resultado.mensaje_general,
        )
 
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error simulación óptima v2: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/predecir/plan",
    summary="Plan de recuperación semana a semana",
    description="""
Recibe DatosTiempoRealRequest directamente (sin envoltura).
Devuelve semanas_restantes en el response.

v8.1: si el estudiante tiene racha_trims_riesgo >= 3, el plan
incluye involucrar_direccion=true con mensaje para dirección.
    """,
)
async def plan_recuperacion(datos: DatosTiempoRealRequest):
    if not modelo_manager.esta_listo:
        raise HTTPException(status_code=503, detail="Modelos ML no cargados.")

    try:
        resultado_ml = predecir_tiempo_real(datos)
        semanas_rest = datos.config_periodo.total_semanas - datos.semana

        if semanas_rest < 2:
            return {
                "estudiante_id":     datos.estudiante_id,
                "materia":           datos.materia,
                "semana_actual":     datos.semana,
                "semanas_restantes": semanas_rest,
                "total_semanas":     datos.config_periodo.total_semanas,
                "nivel_riesgo":      resultado_ml.nivel_riesgo.value,
                "nota_estimada":     resultado_ml.nota_estimada_final,
                "plan":              None,
                "gemini_disponible": False,
                "mensaje":           (
                    f"Solo {semanas_rest} semana(s) restante(s) — "
                    "plan de recuperación no aplicable."
                ),
            }

        if resultado_ml.nivel_riesgo.value == "bajo":
            return {
                "estudiante_id":     datos.estudiante_id,
                "materia":           datos.materia,
                "semana_actual":     datos.semana,
                "semanas_restantes": semanas_rest,
                "total_semanas":     datos.config_periodo.total_semanas,
                "nivel_riesgo":      "bajo",
                "nota_estimada":     resultado_ml.nota_estimada_final,
                "plan":              None,
                "gemini_disponible": False,
                "mensaje":           "El estudiante no requiere plan de recuperación.",
            }

        plan = await generar_plan_recuperacion(datos, resultado_ml)

        return {
            "estudiante_id":     datos.estudiante_id,
            "materia":           datos.materia,
            "semana_actual":     datos.semana,
            "semanas_restantes": semanas_rest,
            "total_semanas":     datos.config_periodo.total_semanas,
            "nivel_riesgo":      resultado_ml.nivel_riesgo.value,
            "nota_estimada":     resultado_ml.nota_estimada_final,
            "plan":              plan,
            "gemini_disponible": plan is not None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generando plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predecir/clase",
    summary="Análisis de clase completa",
    description="""
Recibe la lista de estudiantes con sus datos individuales,
corre las predicciones ML internamente, calcula el resumen
y llama a Gemini UNA sola vez con el agregado.

v8.1: el resumen ahora incluye estadísticas de patrón crónico
(cuántos estudiantes llevan 2+ trimestres en riesgo) para que
Gemini pueda dar recomendaciones de intervención institucional
cuando corresponda.
    """,
)
async def analizar_clase_endpoint(
    request:        ClaseRequest,
    incluir_gemini: bool = Query(default=True),
    usar_xgboost:   bool = Query(default=True),
):
    if not modelo_manager.esta_listo:
        raise HTTPException(status_code=503, detail="Modelos ML no cargados.")

    try:
        resultados_individuales = []
        for datos_est in request.estudiantes:
            res = predecir_tiempo_real(datos_est, usar_xgboost)
            resultados_individuales.append({
                "estudiante_id":         datos_est.estudiante_id,
                "nivel_riesgo":          res.nivel_riesgo.value,
                "probabilidad_reprobar": res.probabilidad_reprobar,
                "nota_estimada_final":   res.nota_estimada_final,
                "clasificacion":         res.clasificacion_estimada.value,
                "asistencia_pct":        datos_est.asistencia_acumulada_pct,
                "factores_riesgo":       res.factores_riesgo,
                "racha_trims_riesgo":    getattr(datos_est, "racha_trims_riesgo", 0) or 0,
            })

        total      = len(resultados_individuales)
        critico    = sum(1 for r in resultados_individuales if r["nivel_riesgo"] == "critico")
        alto       = sum(1 for r in resultados_individuales if r["nivel_riesgo"] == "alto")
        medio      = sum(1 for r in resultados_individuales if r["nivel_riesgo"] == "medio")
        bajo       = sum(1 for r in resultados_individuales if r["nivel_riesgo"] == "bajo")
        prom_nota  = round(
            sum(r["nota_estimada_final"] for r in resultados_individuales) / total, 1
        )
        prom_asist = round(
            sum(r["asistencia_pct"] for r in resultados_individuales) / total, 1
        )
        pct_riesgo = round((critico + alto + medio) / total * 100, 1)

        cronicos_2 = sum(
            1 for r in resultados_individuales if r["racha_trims_riesgo"] >= 2
        )
        cronicos_3 = sum(
            1 for r in resultados_individuales if r["racha_trims_riesgo"] >= 3
        )

        resumen = {
            "total_estudiantes":   total,
            "critico":             critico,
            "alto":                alto,
            "medio":               medio,
            "bajo":                bajo,
            "promedio_clase":      prom_nota,
            "asistencia_promedio": prom_asist,
            "pct_riesgo":          pct_riesgo,
            "cronicos_2_trims":    cronicos_2,
            "cronicos_3_trims":    cronicos_3,
        }

        analisis = None
        if incluir_gemini:
            analisis = await analizar_clase(
                materia       = request.materia,
                trimestre     = request.estudiantes[0].trimestre,
                semana        = request.semana_actual,
                total_semanas = request.config_periodo.total_semanas,
                resumen_clase = resumen,
            )

        return {
            "total_estudiantes":   total,
            "en_riesgo_critico":   critico,
            "en_riesgo_alto":      alto,
            "en_riesgo_medio":     medio,
            "sin_riesgo":          bajo,
            "promedio_clase":      prom_nota,
            "asistencia_promedio": prom_asist,
            "pct_riesgo":          pct_riesgo,
            "cronicos_2_trims":    cronicos_2,
            "cronicos_3_trims":    cronicos_3,
            "estudiantes":         resultados_individuales,
            "materia":             request.materia,
            "semana":              request.semana_actual,
            "analisis":            analisis,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error análisis clase: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/materiales/recursos-externos",
    response_model=RecursosExternosResponse,
    summary="Sugerir recursos externos via Gemini para un tema sin materiales internos",
    description="""
Llamado por Node.js en la Rama B de dispararAsignacionMaterial,
cuando no hay materiales internos (fecha_publicacion IS NULL) para el tema.

Centraliza la llamada a Gemini en el servicio ML en lugar de
llamarlo directamente desde Node.js, eliminando la necesidad de
mantener GEMINI_API_KEY en el entorno Node.
    """,
)
async def recursos_externos(request: RecursosExternosRequest):
    try:
        recursos_raw = await generar_recursos_externos(
            tema_titulo      = request.tema_titulo,
            tema_descripcion = request.tema_descripcion,
            palabras_clave   = request.palabras_clave,
            nivel_dificultad = request.nivel_dificultad,
            objetivos_unidad = request.objetivos_unidad,
            nivel_educativo  = request.nivel_educativo,
        )

        recursos = [
            {
                "titulo":         r["titulo"],
                "url":            r["url"],
                "origen_externo": r.get("origen_externo", "web"),
            }
            for r in recursos_raw
        ]

        return RecursosExternosResponse(
            recursos          = recursos,
            gemini_disponible = len(recursos) > 0,
            total             = len(recursos),
        )

    except Exception as e:
        logger.error(f"[recursos_externos] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/reentrenar",
    response_model=ReentrenarResponse,
    summary="Reentrenar el modelo",
)
async def reentrenar():
    """
    v8.1: llama a train_v8.py en lugar de train.py.
    Si train_v8.py no existe, intenta train.py como fallback
    para no romper entornos que todavía no migraron.
    """
    settings = get_settings()
    inicio   = datetime.now()

    root         = settings.models_path.parent
    script_v8    = root / "models" / "train_v8.py"
    script_v7    = root / "models" / "train.py"
    script_final = script_v8 if script_v8.exists() else script_v7

    logger.info(f"Reentrenando con: {script_final.name}")

    try:
        res = subprocess.run(
            [sys.executable, str(script_final)],
            capture_output=True, text=True, timeout=600,
        )
        if res.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Error en {script_final.name}:\n{res.stderr[:1000]}"
            )

        modelo_manager.cargar(settings.models_path)
        duracion = (datetime.now() - inicio).total_seconds()

        metricas = {}
        mp = settings.models_path / "metadata.json"
        if mp.exists():
            with open(mp) as f:
                meta     = json.load(f)
                metricas = meta.get("xgb_riesgo", {})

        return ReentrenarResponse(
            exitoso           = True,
            registros_usados  = 0,
            duracion_segundos = round(duracion, 2),
            metricas          = metricas,
            mensaje           = (
                f"Modelos v8.5 reentrenados con {script_final.name} en {duracion:.1f}s"
            ),
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Entrenamiento excedió el tiempo límite (600s)."
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio",
)
async def health():
    gemini_ok = await verificar_disponibilidad()

    status = "ok"
    if not modelo_manager.esta_listo:
        status = "degradado — modelos no cargados"
    elif not gemini_ok:
        status = "parcial — ML ok, Gemini no disponible"

    return HealthResponse(
        status            = status,
        modelos_cargados  = modelo_manager.esta_listo,
        gemini_disponible = gemini_ok,
        version_modelo    = modelo_manager.version,
        tipo_modelo       = modelo_manager.tipo_modelo,
        n_features        = modelo_manager.n_features,
    )


@router.get("/modelo/info", summary="Metadata del modelo actual")
async def info_modelo():
    settings = get_settings()
    mp       = settings.models_path / "metadata.json"
    if not mp.exists():
        raise HTTPException(
            status_code=404,
            detail="No hay metadata. Ejecutá train_v8.py primero."
        )
    with open(mp) as f:
        return json.load(f)