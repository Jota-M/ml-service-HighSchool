"""
app/services/simulate_service.py — Simulación de escenarios tiempo real
"""

from app.schemas.prediccion import (
    DatosTiempoRealRequest, EscenarioSimulacion,
    ResultadoModelo, ResultadoEscenario, NivelRiesgo
)
from app.services.feature_service import aplicar_escenario
from app.services.ml_service import predecir_tiempo_real


def _conclusion(cambio_prob: float, nivel: NivelRiesgo, cambio_nota: float) -> str:
    if cambio_prob <= -0.20:   impacto = "Impacto muy alto"
    elif cambio_prob <= -0.10: impacto = "Impacto significativo"
    elif cambio_prob <= -0.05: impacto = "Impacto moderado"
    elif cambio_prob < 0:      impacto = "Impacto leve"
    elif cambio_prob == 0:     impacto = "Sin cambio"
    else:                      impacto = "Empeora la situación"

    estados = {
        NivelRiesgo.BAJO: "riesgo bajo", NivelRiesgo.MEDIO: "riesgo medio",
        NivelRiesgo.ALTO: "riesgo alto", NivelRiesgo.CRITICO: "riesgo crítico",
    }
    partes = [impacto, f"→ {estados.get(nivel,'')}"]
    if cambio_nota != 0:
        partes.append(f"nota {cambio_nota:+.1f} pts")
    return " | ".join(partes)


async def simular_escenarios(
    datos_base: DatosTiempoRealRequest,
    escenarios: list[EscenarioSimulacion],
    usar_xgboost: bool = True,
) -> tuple[ResultadoModelo, list[ResultadoEscenario]]:

    situacion_actual = predecir_tiempo_real(datos_base, usar_xgboost)
    resultados = []

    for esc in escenarios:
        datos_mod = aplicar_escenario(datos_base, esc)
        res_esc   = predecir_tiempo_real(datos_mod, usar_xgboost)

        cambio_prob = round(res_esc.probabilidad_reprobar - situacion_actual.probabilidad_reprobar, 4)
        cambio_nota = round(res_esc.nota_estimada_final   - situacion_actual.nota_estimada_final, 2)

        resultados.append(ResultadoEscenario(
            descripcion           = esc.descripcion,
            probabilidad_reprobar = res_esc.probabilidad_reprobar,
            nivel_riesgo          = res_esc.nivel_riesgo,
            nota_estimada_final   = res_esc.nota_estimada_final,
            cambio_probabilidad   = cambio_prob,
            cambio_nota           = cambio_nota,
            conclusion            = _conclusion(cambio_prob, res_esc.nivel_riesgo, cambio_nota),
        ))

    return situacion_actual, resultados