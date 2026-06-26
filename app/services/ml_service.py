"""
app/services/ml_service.py — v8.1

Cambios respecto a v7:
  - ModeloManager ahora reporta version="v8.1" y n_features=31
  - predecir_tiempo_real() pasa todos los campos nuevos del schema v8.1
    a calcular_nota_final() — historial, observaciones, correlación,
    nivel educativo, régimen de ponderación
  - Ajuste de consistencia post-predicción sin cambios lógicos,
    solo actualizado el comentario de versión
  - tipo_modelo en HealthResponse actualizado a "tiempo_real_semanal_v8"
"""

# pyrefly: ignore [missing-import]
import joblib
import json
import logging
# pyrefly: ignore [missing-import]
import numpy as np
from pathlib import Path

from app.config import get_settings
from app.schemas.prediccion import (
    DatosTiempoRealRequest,
    ResultadoModelo,
    NivelRiesgo,
    Clasificacion,
    ConfianzaPrediccion,
    NivelConfianza,
)
from app.services.feature_service import (
    construir_features,
    calcular_confianza,
    calcular_factores,
    calcular_nota_final,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# MANAGER DE MODELOS
# ─────────────────────────────────────────

class ModeloManager:
    def __init__(self):
        self._rf_riesgo  = None
        self._rf_nota    = None
        self._xgb_riesgo = None
        self._xgb_nota   = None
        self._metadata   = None
        self._cargado    = False

    def cargar(self, models_path: Path) -> bool:
        try:
            self._rf_riesgo  = joblib.load(models_path / "rf_riesgo.pkl")
            self._rf_nota    = joblib.load(models_path / "rf_nota.pkl")
            self._xgb_riesgo = joblib.load(models_path / "xgb_riesgo.pkl")
            self._xgb_nota   = joblib.load(models_path / "xgb_nota.pkl")

            mp = models_path / "metadata.json"
            if mp.exists():
                with open(mp) as f:
                    self._metadata = json.load(f)

            self._cargado = True
            n_features = self._metadata.get("n_features", 31) if self._metadata else 31
            version    = self._metadata.get("version", "v8.1") if self._metadata else "v8.1"
            logger.info(f"✅ Modelos ML {version} cargados ({n_features} features)")
            return True

        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            self._cargado = False
            return False

    @property
    def esta_listo(self) -> bool:
        return self._cargado

    @property
    def version(self) -> str:
        if self._metadata:
            return self._metadata.get("version", "v8.1")
        return "v8.1"

    @property
    def n_features(self) -> int:
        if self._metadata:
            return self._metadata.get("n_features", 29)  # era 31
        return 29  # era 31

    @property
    def tipo_modelo(self) -> str:
        if self._metadata:
            return self._metadata.get("tipo", "tiempo_real_semanal_con_historial")
        return "tiempo_real_semanal_con_historial"

    def predecir_riesgo(self, features: np.ndarray, usar_xgboost: bool = True) -> float:
        if not self._cargado:
            raise RuntimeError("Modelos no cargados. Ejecutá train_v8.py primero.")
        modelo = self._xgb_riesgo if usar_xgboost else self._rf_riesgo
        return round(float(modelo.predict_proba(features)[0][1]), 4)

    def predecir_nota_raw(self, features: np.ndarray, usar_xgboost: bool = True) -> float:
        if not self._cargado:
            raise RuntimeError("Modelos no cargados. Ejecutá train_v8.py primero.")
        modelo = self._xgb_nota if usar_xgboost else self._rf_nota
        nota   = float(modelo.predict(features)[0])
        return round(max(1.0, min(100.0, nota)), 1)


modelo_manager = ModeloManager()


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def _nivel_riesgo(prob: float) -> NivelRiesgo:
    if prob >= 0.75:   return NivelRiesgo.CRITICO
    elif prob >= 0.50: return NivelRiesgo.ALTO
    elif prob >= 0.25: return NivelRiesgo.MEDIO
    else:              return NivelRiesgo.BAJO


def _clasificacion(nota: float) -> Clasificacion:
    if nota < 51:   return Clasificacion.ED
    elif nota < 69: return Clasificacion.DA
    elif nota < 85: return Clasificacion.DO
    else:           return Clasificacion.DP


# ─────────────────────────────────────────
# PREDICCIÓN PRINCIPAL
# ─────────────────────────────────────────

def predecir_tiempo_real(
    datos: DatosTiempoRealRequest,
    usar_xgboost: bool = True,
) -> ResultadoModelo:
    features = construir_features(datos)

    prob     = modelo_manager.predecir_riesgo(features, usar_xgboost)
    nota_raw = modelo_manager.predecir_nota_raw(features, usar_xgboost)

    tiene_evaluaciones = len(datos.todas_las_notas) > 0

    nota_trim_ant = getattr(datos, "nota_trim_ant", -1) or -1
    pct_periodo = datos.semana / datos.config_periodo.total_semanas

    if tiene_evaluaciones:
        nota_final = calcular_nota_final(
            notas_practicas=list(datos.notas_practicas),
            notas_examenes=list(datos.notas_examenes),
            ponderaciones=datos.config_periodo.ponderaciones,
            notas_sab=list(datos.notas_sab),
            notas_hac=list(datos.notas_hac),
            nota_complementaria_pct=datos.nota_complementaria_pct,
            peso_complementario=datos.peso_complementario,
        )
    else:
        nota_final = nota_raw

    n_eval = len(datos.todas_las_notas)

    # ── Si NO hay evaluaciones reales ─────────────────────────────
    if not tiene_evaluaciones:

        if nota_trim_ant > 0:

            # Peso base según avance del período
            peso_hist_tiempo = max(0.1, 0.85 - pct_periodo * 1.6)

            # Ajuste según cantidad de evaluaciones
            if n_eval >= 6:
                ajuste_eval = 0.5
            elif n_eval >= 4:
                ajuste_eval = 0.7
            elif n_eval >= 2:
                ajuste_eval = 0.85
            else:
                ajuste_eval = 1.0

            peso_hist = round(peso_hist_tiempo * ajuste_eval, 2)
            peso_model = round(1.0 - peso_hist, 2)

            nota_final = round(
                nota_trim_ant * peso_hist +
                nota_raw * peso_model,
                1
            )

            logger.info(
                f"[ML] Anclaje historial → "
                f"hist={peso_hist:.2f} "
                f"model={peso_model:.2f} "
                f"nota={nota_final}"
            )

    else:
        # Sin historial → usar modelo puro
        nota_final = nota_raw

    n_eval = len(datos.todas_las_notas)

    confianza_dict = calcular_confianza(
        datos.semana,
        datos.config_periodo.total_semanas,
        n_eval,
    )
    factores_r, factores_p = calcular_factores(datos)

    confianza = ConfianzaPrediccion(
        nivel              = NivelConfianza(confianza_dict["nivel"]),
        porcentaje_periodo = confianza_dict["porcentaje_periodo"],
        mensaje            = confianza_dict["mensaje"],
    )

    # ── Ajuste de consistencia post-predicción (v8.2, igual que v7) ──────────
    # Si nota_final < 51 pero nivel_riesgo quedó "bajo", es contradictorio.
    # Ocurre cuando hay pocas evaluaciones y el modelo es conservador.
    # Se corrige escalando prob y nivel según el porcentaje del período.
    nivel       = _nivel_riesgo(prob)
    pct_periodo = datos.semana / datos.config_periodo.total_semanas

    if nota_final < 51:
        if pct_periodo >= 0.85:
            prob  = max(prob, 0.80)
            nivel = NivelRiesgo.CRITICO
        elif pct_periodo >= 0.70:
            prob  = max(prob, 0.55)
            nivel = NivelRiesgo.ALTO
        elif nivel == NivelRiesgo.BAJO:
            prob  = max(prob, 0.25)
            nivel = NivelRiesgo.MEDIO

    # ── Ajuste adicional v8.2: patrón crónico ────────────────────────────────
    racha_trims = getattr(datos, "racha_trims_riesgo", 0) or 0
    if racha_trims >= 3 and nivel not in (NivelRiesgo.CRITICO,):
        prob  = max(prob, 0.52)
        nivel = NivelRiesgo.ALTO
        logger.info(
            f"[ML v8.2] Ajuste patrón crónico: estudiante {datos.estudiante_id} "
            f"lleva {racha_trims} trimestres en riesgo → nivel forzado a ALTO mínimo"
        )

    # ── Ajuste v8.3: asistencia baja para institución privada ────────────────
    # En institución privada la asistencia es 85-100%.
    # El umbral legal de 75% nunca se alcanza, así que se sube a 88%
    # para detectar casos donde la asistencia baja respecto al promedio.
    # Si asistencia < 88% y nivel es BAJO → forzar MEDIO.
    if datos.asistencia_acumulada_pct < 88 and nivel == NivelRiesgo.BAJO:
        prob  = max(prob, 0.26)
        nivel = NivelRiesgo.MEDIO
        logger.info(
            f"[ML v8.3] Ajuste asistencia: {datos.asistencia_acumulada_pct:.1f}% "
            f"→ nivel mínimo MEDIO (baja para institución privada)"
        )

    return ResultadoModelo(
        probabilidad_reprobar  = prob,
        nivel_riesgo           = nivel,
        nota_estimada_final    = nota_final,
        clasificacion_estimada = _clasificacion(nota_final),
        factores_riesgo        = factores_r,
        factores_positivos     = factores_p,
        confianza              = confianza,
    )