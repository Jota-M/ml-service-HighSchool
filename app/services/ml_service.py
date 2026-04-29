"""
app/services/ml_service.py — Motor de predicción tiempo real
"""

import joblib, json, logging
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
from app.services.feature_service import construir_features, calcular_confianza, calcular_factores

logger = logging.getLogger(__name__)


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
            logger.info("✅ Modelos ML cargados correctamente")
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
        if self._metadata and "entrenado_en" in self._metadata:
            return self._metadata["entrenado_en"]
        return "tiempo_real_v4"

    def predecir(self, features: np.ndarray, usar_xgboost: bool = True) -> dict:
        if not self._cargado:
            raise RuntimeError("Modelos no cargados. Ejecuta train.py primero.")

        mr = self._xgb_riesgo if usar_xgboost else self._rf_riesgo
        mn = self._xgb_nota   if usar_xgboost else self._rf_nota

        prob = float(mr.predict_proba(features)[0][1])
        nota = float(mn.predict(features)[0])
        nota = round(max(1.0, min(100.0, nota)), 1)

        return {
            "prob_reprobar":  round(prob, 4),   # ← única clave, sin typos
            "nota_estimada":  nota,
            "modelo":         "xgboost" if usar_xgboost else "random_forest",
        }


modelo_manager = ModeloManager()


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


def predecir_tiempo_real(
    datos: DatosTiempoRealRequest,
    usar_xgboost: bool = True,
) -> ResultadoModelo:

    features = construir_features(datos)
    res      = modelo_manager.predecir(features, usar_xgboost)

    prob = res["prob_reprobar"]   # ← corregido: eliminado el if/else con el typo
    nota = res["nota_estimada"]

    n_evaluaciones = len(datos.notas_practicas) + len(datos.notas_examenes)
    confianza_dict = calcular_confianza(datos.semana, n_evaluaciones)
    factores_r, factores_p = calcular_factores(datos)

    confianza = ConfianzaPrediccion(
        nivel                = NivelConfianza(confianza_dict["nivel"]),
        porcentaje_trimestre = confianza_dict["porcentaje_trimestre"],
        mensaje              = confianza_dict["mensaje"],
    )

    return ResultadoModelo(
        probabilidad_reprobar  = prob,
        nivel_riesgo           = _nivel_riesgo(prob),
        nota_estimada_final    = nota,             # ← nombre correcto del schema
        clasificacion_estimada = _clasificacion(nota),
        factores_riesgo        = factores_r,
        factores_positivos     = factores_p,
        confianza              = confianza,
    )