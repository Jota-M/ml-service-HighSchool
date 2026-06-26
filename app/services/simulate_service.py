"""
app/services/simulate_service.py — v8.10

Cambios respecto a v8.9:
  - prom_sab_act y prom_hac_act usan sab_trim_ant / hac_trim_ant cuando no hay
    notas actuales, en lugar de caer directo a nota_trim_ant o fallback genérico.
    Jerarquía SAB: notas_sab actuales > sab_trim_ant > mejor_hist*0.88 > 40
    Jerarquía HAC: notas_hac actuales > hac_trim_ant > 40
    Así "Si sigue igual" refleja el rendimiento real por dimensión del trimestre anterior.
"""

from dataclasses import dataclass
from typing import Optional

from app.schemas.prediccion import (
    DatosTiempoRealRequest,
    EscenarioSimulacion,
    ResultadoModelo,
    ResultadoEscenario,
    NivelRiesgo,
    Clasificacion,
)

from app.services.feature_service import aplicar_escenario, calcular_nota_final
from app.services.ml_service import predecir_tiempo_real

# pyrefly: ignore [missing-import]
import numpy as np


# ─────────────────────────────────────────
# SIMULACIÓN ORIGINAL — sin cambios
# ─────────────────────────────────────────

def _conclusion(cambio_prob, nivel, cambio_nota):
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
    partes = [impacto, f"→ {estados.get(nivel, '')}"]
    if cambio_nota != 0:
        partes.append(f"nota {cambio_nota:+.1f} pts")
    return " | ".join(partes)


def _clasificacion(nota):
    if nota < 51:   return Clasificacion.ED
    elif nota < 69: return Clasificacion.DA
    elif nota < 85: return Clasificacion.DO
    else:           return Clasificacion.DP


def _con_nota_real(resultado_ml, datos):
    nota_real = calcular_nota_final(
        notas_practicas=list(datos.notas_practicas),
        notas_examenes=list(datos.notas_examenes),
        ponderaciones=datos.config_periodo.ponderaciones,
        notas_sab=list(datos.notas_sab),
        notas_hac=list(datos.notas_hac),
        nota_complementaria_pct=datos.nota_complementaria_pct,
        peso_complementario=datos.peso_complementario,
    )
    return resultado_ml.model_copy(update={
        "nota_estimada_final":    nota_real,
        "clasificacion_estimada": _clasificacion(nota_real),
    })


async def simular_escenarios(
    datos_base: DatosTiempoRealRequest,
    escenarios: list[EscenarioSimulacion],
    usar_xgboost: bool = True,
) -> tuple[ResultadoModelo, list[ResultadoEscenario]]:
    situacion_actual = _con_nota_real(
        predecir_tiempo_real(datos_base, usar_xgboost), datos_base,
    )
    resultados = []
    for esc in escenarios:
        datos_mod = aplicar_escenario(datos_base, esc)
        res_esc   = _con_nota_real(predecir_tiempo_real(datos_mod, usar_xgboost), datos_mod)
        cambio_prob = round(res_esc.probabilidad_reprobar - situacion_actual.probabilidad_reprobar, 4)
        cambio_nota = round(res_esc.nota_estimada_final - situacion_actual.nota_estimada_final, 2)
        resultados.append(ResultadoEscenario(
            descripcion=esc.descripcion,
            probabilidad_reprobar=res_esc.probabilidad_reprobar,
            nivel_riesgo=res_esc.nivel_riesgo,
            nota_estimada_final=res_esc.nota_estimada_final,
            cambio_probabilidad=cambio_prob,
            cambio_nota=cambio_nota,
            conclusion=_conclusion(cambio_prob, res_esc.nivel_riesgo, cambio_nota),
        ))
    return situacion_actual, resultados


# ─────────────────────────────────────────
# OPTIMIZACIÓN ORIGINAL — sin cambios
# ─────────────────────────────────────────

@dataclass
class AccionRequerida:
    componente: str; label: str; valor_actual: float
    valor_necesario: float; delta: float; impacto_nota: float; dificultad: str


@dataclass
class OptimizacionResponse:
    objetivo_nota: float; nota_actual: float; nota_proyectada: float
    alcanzable: bool; acciones: list; nota_maxima_posible: float
    mensaje: str; modo: str


def _nota_actual_formula(datos):
    return calcular_nota_final(
        notas_practicas=list(datos.notas_practicas),
        notas_examenes=list(datos.notas_examenes),
        ponderaciones=datos.config_periodo.ponderaciones,
        notas_sab=list(datos.notas_sab),
        notas_hac=list(datos.notas_hac),
        nota_complementaria_pct=datos.nota_complementaria_pct,
        peso_complementario=datos.peso_complementario,
    )


def _nota_maxima_formula(datos):
    ponds = {k: v / 100 for k, v in datos.config_periodo.ponderaciones.items()}
    return round(
        100.0 * ponds.get("SAB", 0.45) + 100.0 * ponds.get("HAC", 0.40)
        + datos.nota_complementaria_pct * datos.peso_complementario, 1,
    )


def _prom(notas):
    return round(sum(notas) / len(notas), 1) if notas else None


def _label_palanca(componente, sin_notas, actual, necesario):
    if componente == "asistencia":
        return f"Mejorar asistencia de {actual:.0f}% a {necesario:.0f}%"
    nombre_sing = "examen"   if componente == "examenes"  else "práctica"
    nombre_pl   = "exámenes" if componente == "examenes"  else "prácticas"
    return (f"Sacar al menos {necesario:.0f} en el próximo {nombre_sing}" if sin_notas
            else f"Subir promedio de {nombre_pl} de {actual:.0f} a {necesario:.0f}")


def _dificultad(delta, componente):
    if componente == "asistencia":
        return "baja" if delta <= 5 else "media" if delta <= 12 else "alta"
    return "baja" if delta <= 10 else "media" if delta <= 20 else "alta"


async def calcular_optimo(
    datos, objetivo_nota=51.0, restricciones=None, usar_xgboost=True,
) -> OptimizacionResponse:
    restricciones = restricciones or {}
    nota_actual = _nota_actual_formula(datos)
    nota_maxima = _nota_maxima_formula(datos)

    if nota_actual >= objetivo_nota:
        return OptimizacionResponse(
            objetivo_nota=objetivo_nota, nota_actual=nota_actual,
            nota_proyectada=nota_actual, alcanzable=True, acciones=[],
            nota_maxima_posible=nota_maxima,
            mensaje=f"El estudiante ya supera la nota objetivo ({nota_actual:.1f} ≥ {objetivo_nota}).",
            modo="minimo_esfuerzo",
        )
    if nota_maxima < objetivo_nota:
        return OptimizacionResponse(
            objetivo_nota=objetivo_nota, nota_actual=nota_actual,
            nota_proyectada=nota_maxima, alcanzable=False, acciones=[],
            nota_maxima_posible=nota_maxima,
            mensaje=f"Aunque suba todo a 100, la nota máxima alcanzable es {nota_maxima:.1f}.",
            modo="minimo_esfuerzo",
        )

    ponds    = datos.config_periodo.ponderaciones
    peso_sab = ponds.get("SAB", 45) / 100
    peso_hac = ponds.get("HAC", 40) / 100
    prom_sab = _prom(datos.notas_saber)
    prom_hac = _prom(datos.notas_hacer)
    palancas = []

    if not restricciones.get("bloquear_examenes", False):
        palancas.append({"componente": "examenes", "sin_notas": prom_sab is None,
                         "actual": prom_sab or 0.0, "maximo": 100.0, "peso": peso_sab})
    if not restricciones.get("bloquear_practicas", False):
        palancas.append({"componente": "practicas", "sin_notas": prom_hac is None,
                         "actual": prom_hac or 0.0, "maximo": 100.0, "peso": peso_hac})
    if not restricciones.get("bloquear_asistencia", False):
        palancas.append({"componente": "asistencia", "sin_notas": False,
                         "actual": datos.asistencia_acumulada_pct, "maximo": 100.0, "peso": 0.08})

    palancas.sort(key=lambda p: p["peso"], reverse=True)
    acciones = []
    nota_acumulada = nota_actual
    gap_restante   = round(objetivo_nota - nota_actual, 2)

    for p in palancas:
        if gap_restante <= 0.05: break
        margen = p["maximo"] - p["actual"]
        if margen <= 0.5: continue
        aporte_max = margen * p["peso"]
        if aporte_max <= 0: continue
        aporte_nec  = min(aporte_max, gap_restante)
        delta_nec   = round(aporte_nec / p["peso"], 1)
        valor_nec   = round(min(p["actual"] + delta_nec, p["maximo"]), 1)
        delta_real  = round(valor_nec - p["actual"], 1)
        impacto     = round(delta_real * p["peso"], 2)
        if delta_real < 0.5: continue
        acciones.append(AccionRequerida(
            componente=p["componente"],
            label=_label_palanca(p["componente"], p["sin_notas"], p["actual"], valor_nec),
            valor_actual=round(p["actual"], 1), valor_necesario=valor_nec,
            delta=delta_real, impacto_nota=impacto,
            dificultad=_dificultad(delta_real, p["componente"]),
        ))
        nota_acumulada += impacto
        gap_restante   -= impacto

    nota_proyectada = round(nota_acumulada, 1)
    alcanzable      = nota_proyectada >= objetivo_nota - 0.5
    if alcanzable and acciones:
        mensaje = (f"Con {len(acciones)} acción(es) puede alcanzar {nota_proyectada:.1f}: "
                   + "; ".join(a.label for a in acciones) + ".")
    elif not alcanzable:
        mensaje = f"Aplicando todas las palancas disponibles se proyecta {nota_proyectada:.1f}."
    else:
        mensaje = "Sin acciones necesarias — ya supera el objetivo."

    return OptimizacionResponse(
        objetivo_nota=objetivo_nota, nota_actual=nota_actual,
        nota_proyectada=nota_proyectada, alcanzable=alcanzable,
        acciones=acciones, nota_maxima_posible=nota_maxima,
        mensaje=mensaje, modo="minimo_esfuerzo",
    )


# ─────────────────────────────────────────
# OPTIMIZACIÓN V2 — v8.10
# ─────────────────────────────────────────

def _estimar_pendientes(notas, semana_actual, total_semanas, max_total):
    semanas_rest = max(0, total_semanas - semana_actual)
    if semanas_rest == 0: return 0
    n = len(notas)
    est = max(1, round(semanas_rest / 3)) if n == 0 else round(n / max(1, semana_actual) * semanas_rest)
    return max(0, min(est, max_total - n))


def _techo_garantizado(notas, mejor_hist, nota_necesaria):
    base = float(np.mean(notas)) if notas else (mejor_hist * 0.9 if mejor_hist > 0 else 55.0)
    if mejor_hist > 0:
        base = max(base, mejor_hist * 0.88)
    return round(max(55.0, min(95.0, max(base, nota_necesaria + 5.0))), 1)


def _nota_con_futuras(notas_sab, notas_hac, fut_sab, fut_hac, ponds, nota_comp, peso_comp):
    return calcular_nota_final(
        notas_practicas=[], notas_examenes=[], ponderaciones=ponds,
        notas_sab=notas_sab + fut_sab, notas_hac=notas_hac + fut_hac,
        nota_complementaria_pct=nota_comp, peso_complementario=peso_comp,
    )


def _calcular_progresion_hac(
    n_hac: int,
    prom_hac_actual: float,
    paso: float,
) -> list[float]:
    """
    Genera una progresión gradual de n_hac notas empezando desde prom_hac_actual
    con incremento de `paso` por evaluación.
    Ej: prom=45, paso=2, n=3 → [45, 47, 49]
    Clampea a [0, 100].
    """
    if n_hac == 0:
        return []
    return [round(max(0.0, min(100.0, prom_hac_actual + paso * i)), 1) for i in range(n_hac)]


def _sab_necesario_con_hac_fijo(
    n_sab: int,
    notas_sab: list,
    notas_hac: list,
    fut_hac: list,
    objetivo: float,
    ponds: dict,
    peso_sab: float,
    nota_comp: float,
    peso_comp: float,
) -> float:
    """
    Dado que HAC futuras = fut_hac, ¿qué promedio necesitan las SAB futuras
    para alcanzar objetivo?
    Retorna 999 si n_sab=0 o denominador=0.
    """
    if n_sab == 0:
        return 999.0
    nota_sin = calcular_nota_final(
        notas_practicas=[], notas_examenes=[], ponderaciones=ponds,
        notas_sab=notas_sab, notas_hac=notas_hac + fut_hac,
        nota_complementaria_pct=nota_comp, peso_complementario=peso_comp,
    )
    gap = objetivo - nota_sin
    if gap <= 0:
        return sum(notas_sab) / len(notas_sab) if notas_sab else 0.0
    n_sab_total = len(notas_sab) + n_sab
    prom_sab    = sum(notas_sab) / len(notas_sab) if notas_sab else 0.0
    denom       = (n_sab / n_sab_total) * peso_sab
    return (prom_sab + gap / denom) if denom > 0 else 999.0


def _construir_escenario_con_progresion(
    eid: str,
    titulo: str,
    descripcion: str,
    n_sab: int,
    n_hac: int,
    notas_sab: list,
    notas_hac: list,
    objetivo: float,
    prom_hac_actual: float,
    paso_hac: float,
    ponds: dict,
    peso_sab: float,
    nota_comp: float,
    peso_comp: float,
    mejor_hist: float,
    bloquear_sab: bool,
    bloquear_hac: bool,
):
    """
    Construye un escenario donde:
    - HAC sube gradualmente `paso_hac` puntos por evaluación
    - SAB se ajusta para cubrir el gap restante
    """
    # 1. Generar progresión HAC
    fut_hac = _calcular_progresion_hac(n_hac, prom_hac_actual, paso_hac) if not bloquear_hac else []

    # 2. Calcular SAB necesario considerando esa mejora en HAC
    sab_nec = _sab_necesario_con_hac_fijo(
        n_sab, notas_sab, notas_hac, fut_hac,
        objetivo, ponds, peso_sab, nota_comp, peso_comp,
    ) if not bloquear_sab else 999.0

    # 3. Techo garantizado para SAB
    techo_s = _techo_garantizado(notas_sab, mejor_hist, sab_nec)
    sab_uso = round(min(sab_nec, 95.0), 1)

    # 4. Si SAB necesario > 95, ajustar HAC para compensar
    if sab_nec > 95.0 and n_hac > 0 and not bloquear_hac:
        nota_con_sab95 = calcular_nota_final(
            notas_practicas=[], notas_examenes=[], ponderaciones=ponds,
            notas_sab=notas_sab + [95.0] * n_sab if n_sab > 0 else notas_sab,
            notas_hac=notas_hac,
            nota_complementaria_pct=nota_comp, peso_complementario=peso_comp,
        )
        gap_hac = objetivo - nota_con_sab95
        if gap_hac > 0:
            n_hac_total = len(notas_hac) + n_hac
            denom_hac   = (n_hac / n_hac_total) * (ponds.get("HAC", 40) / 100)
            if denom_hac > 0:
                hac_nec = prom_hac_actual + gap_hac / denom_hac
                paso_ajustado = (hac_nec - prom_hac_actual) / max(1, n_hac - 1) if n_hac > 1 else 0
                fut_hac = _calcular_progresion_hac(n_hac, prom_hac_actual, max(paso_ajustado, paso_hac))
        sab_uso = 95.0

    fut_sab = [sab_uso] * n_sab if not bloquear_sab and n_sab > 0 else []

    techo_h = _techo_garantizado(notas_hac, mejor_hist, max(fut_hac) if fut_hac else prom_hac_actual)

    evals = []
    for i, nota in enumerate(fut_sab):
        nc = min(nota, 100.0)
        evals.append(EvaluacionPendiente(
            numero=len(notas_sab) + i + 1, tipo="examen",
            nota_objetivo=round(nc, 1), es_alcanzable=nc <= techo_s,
        ))
    for i, nota in enumerate(fut_hac):
        nc = min(nota, 100.0)
        evals.append(EvaluacionPendiente(
            numero=len(notas_hac) + i + 1, tipo="practica",
            nota_objetivo=round(nc, 1), es_alcanzable=nc <= techo_h,
        ))

    nota_proy  = _nota_con_futuras(notas_sab, notas_hac, fut_sab, fut_hac, ponds, nota_comp, peso_comp)
    alcanzable = all(e.es_alcanzable for e in evals) and nota_proy >= objetivo - 0.5
    holgura    = nota_proy - objetivo
    pct        = min(100.0, max(0.0, round(50 + holgura * 3, 1)))

    n_inal = sum(1 for e in evals if not e.es_alcanzable)
    if not evals:
        msg = "Sin evaluaciones pendientes."
    elif n_inal > 0:
        msg = f"{n_inal} evaluación(es) por encima del historial del estudiante."
    elif alcanzable:
        msg = f"Proyección: {nota_proy:.1f} ({'+' if holgura >= 0 else ''}{holgura:.1f} sobre el objetivo)."
    else:
        msg = f"Con estas notas se proyecta {nota_proy:.1f}, aún por debajo del objetivo."

    return EscenarioDetallado(
        id=eid, titulo=titulo, descripcion=descripcion, evaluaciones=evals,
        nota_proyectada=round(nota_proy, 1), alcanzable=alcanzable,
        porcentaje_exito=pct, mensaje=msg,
    )


@dataclass
class EvaluacionPendiente:
    numero: int; tipo: str; nota_objetivo: float; es_alcanzable: bool


@dataclass
class EscenarioDetallado:
    id: str; titulo: str; descripcion: str; evaluaciones: list
    nota_proyectada: float; alcanzable: bool; porcentaje_exito: float; mensaje: str


@dataclass
class SimulacionOptimoV2Result:
    objetivo_nota: float; nota_actual: float; nota_maxima_posible: float
    semanas_restantes: int; practicas_restantes_est: int; examenes_restantes_est: int
    techo_practicas: float; techo_examenes: float; escenarios: list
    ya_alcanza: bool; imposible: bool; mensaje_general: str


async def calcular_optimo_v2(
    datos,
    objetivo_nota: float = 51.0,
    restricciones: Optional[dict] = None,
    usar_xgboost: bool = True,
    practicas_restantes_input: Optional[int] = None,
    examenes_restantes_input: Optional[int] = None,
) -> SimulacionOptimoV2Result:

    restricciones = restricciones or {}
    ponds         = datos.config_periodo.ponderaciones
    peso_sab      = ponds.get("SAB", 45) / 100
    peso_hac      = ponds.get("HAC", 40) / 100
    total_sem     = datos.config_periodo.total_semanas
    semana        = datos.semana
    sem_rest      = max(0, total_sem - semana)
    notas_sab     = list(datos.notas_saber)
    notas_hac     = list(datos.notas_hacer)
    mejor_hist    = getattr(datos, "mejor_nota_historica", -1) or -1
    nota_trim_ant = getattr(datos, "nota_trim_ant", -1) or -1
    nota_comp     = datos.nota_complementaria_pct
    peso_comp     = datos.peso_complementario

    # ── v8.10: leer sab_trim_ant y hac_trim_ant del payload ─────────────────
    sab_trim_ant = getattr(datos, "sab_trim_ant", -1) or -1
    hac_trim_ant = getattr(datos, "hac_trim_ant", -1) or -1

    nota_actual = calcular_nota_final(
        notas_practicas=[], notas_examenes=[], ponderaciones=ponds,
        notas_sab=notas_sab, notas_hac=notas_hac,
        nota_complementaria_pct=nota_comp, peso_complementario=peso_comp,
    )
    nota_max = round(100.0 * peso_sab + 100.0 * peso_hac + nota_comp * peso_comp, 1)

    if nota_actual >= objetivo_nota:
        return SimulacionOptimoV2Result(
            objetivo_nota=objetivo_nota, nota_actual=nota_actual, nota_maxima_posible=nota_max,
            semanas_restantes=sem_rest, practicas_restantes_est=0, examenes_restantes_est=0,
            techo_practicas=0, techo_examenes=0, escenarios=[], ya_alcanza=True, imposible=False,
            mensaje_general=f"El estudiante ya tiene {nota_actual:.1f} y supera el objetivo de {objetivo_nota:.0f}.",
        )

    if nota_max < objetivo_nota:
        return SimulacionOptimoV2Result(
            objetivo_nota=objetivo_nota, nota_actual=nota_actual, nota_maxima_posible=nota_max,
            semanas_restantes=sem_rest, practicas_restantes_est=0, examenes_restantes_est=0,
            techo_practicas=0, techo_examenes=0, escenarios=[], ya_alcanza=False, imposible=True,
            mensaje_general=f"Aunque saque 100 en todo lo que queda, la nota máxima posible es {nota_max:.1f}.",
        )

    bloquear_hac = restricciones.get("bloquear_practicas", False)
    bloquear_sab = restricciones.get("bloquear_examenes", False)

    n_hac = (0 if bloquear_hac else
             practicas_restantes_input if practicas_restantes_input is not None else
             _estimar_pendientes(notas_hac, semana, total_sem, 8))
    n_sab = (0 if bloquear_sab else
             examenes_restantes_input if examenes_restantes_input is not None else
             _estimar_pendientes(notas_sab, semana, total_sem, 2))

    # ── v8.10: promedios de referencia con jerarquía por dimensión ───────────
    #
    # HAC: notas actuales > hac_trim_ant > 40
    if notas_hac:
        prom_hac_act = round(sum(notas_hac) / len(notas_hac), 1)
    elif hac_trim_ant > 0:
        prom_hac_act = round(hac_trim_ant, 1)       # ← HAC real del trim anterior
    else:
        prom_hac_act = 40.0

    # SAB: notas actuales > sab_trim_ant > mejor_hist*0.88 > 40
    if notas_sab:
        prom_sab_act = round(sum(notas_sab) / len(notas_sab), 1)
    elif sab_trim_ant > 0:
        prom_sab_act = round(sab_trim_ant, 1)       # ← SAB real del trim anterior
    elif mejor_hist > 0:
        prom_sab_act = round(mejor_hist * 0.88, 1)
    else:
        prom_sab_act = 40.0

    escenarios = []

    # ── ESCENARIO 1: Si sigue igual ──────────────────────────────────────────
    # SAB y HAC usan sus referencias históricas reales por dimensión
    techo_igual = max(prom_sab_act, prom_hac_act) + 5
    evals_igual = []
    for i in range(n_sab):
        nc = min(prom_sab_act, 100.0)
        evals_igual.append(EvaluacionPendiente(
            numero=len(notas_sab) + i + 1, tipo="examen",
            nota_objetivo=round(nc, 1), es_alcanzable=nc <= techo_igual,
        ))
    for i in range(n_hac):
        nc = min(prom_hac_act, 100.0)
        evals_igual.append(EvaluacionPendiente(
            numero=len(notas_hac) + i + 1, tipo="practica",
            nota_objetivo=round(nc, 1), es_alcanzable=nc <= techo_igual,
        ))

    fut_sab_igual = [prom_sab_act] * n_sab if not bloquear_sab else []
    fut_hac_igual = [prom_hac_act] * n_hac if not bloquear_hac else []
    nota_igual = _nota_con_futuras(notas_sab, notas_hac, fut_sab_igual, fut_hac_igual, ponds, nota_comp, peso_comp)
    holgura_igual = nota_igual - objetivo_nota
    alcanzable_igual = nota_igual >= objetivo_nota - 0.5
    msg_igual = (f"Proyección: {nota_igual:.1f} ({'+' if holgura_igual >= 0 else ''}{holgura_igual:.1f} sobre el objetivo)."
                 if alcanzable_igual else
                 f"Con estas notas se proyecta {nota_igual:.1f}, aún por debajo del objetivo.")

    escenarios.append(EscenarioDetallado(
        id="proyeccion", titulo="Si sigue igual",
        descripcion="Proyección manteniendo el rendimiento actual. Muestra qué tan lejos está del objetivo sin cambios.",
        evaluaciones=evals_igual, nota_proyectada=round(nota_igual, 1),
        alcanzable=alcanzable_igual,
        porcentaje_exito=min(100.0, max(0.0, round(50 + holgura_igual * 3, 1))),
        mensaje=msg_igual,
    ))

    # ── ESCENARIO 2: Para aprobar (51) — HAC sube +2 pts/evaluación ──────────
    OBJ_APR = 51.0
    escenarios.append(_construir_escenario_con_progresion(
        eid="aprobar",
        titulo="Para aprobar (51)",
        descripcion="Las notas necesarias en cada evaluación para llegar exactamente a 51, con las prácticas mejorando gradualmente.",
        n_sab=n_sab, n_hac=n_hac,
        notas_sab=notas_sab, notas_hac=notas_hac,
        objetivo=OBJ_APR,
        prom_hac_actual=prom_hac_act,
        paso_hac=2.0,
        ponds=ponds, peso_sab=peso_sab,
        nota_comp=nota_comp, peso_comp=peso_comp,
        mejor_hist=mejor_hist,
        bloquear_sab=bloquear_sab, bloquear_hac=bloquear_hac,
    ))

    # ── ESCENARIO 3: Para aprobar con margen (60) — HAC sube +3 pts/evaluación
    OBJ_MAR = min(60.0, nota_max - 2) if nota_max < 62 else 60.0
    escenarios.append(_construir_escenario_con_progresion(
        eid="margen",
        titulo=f"Para aprobar con margen ({OBJ_MAR:.0f})",
        descripcion=f"Llegar a {OBJ_MAR:.0f} puntos da margen de error si alguna evaluación sale peor de lo esperado.",
        n_sab=n_sab, n_hac=n_hac,
        notas_sab=notas_sab, notas_hac=notas_hac,
        objetivo=OBJ_MAR,
        prom_hac_actual=prom_hac_act,
        paso_hac=3.0,
        ponds=ponds, peso_sab=peso_sab,
        nota_comp=nota_comp, peso_comp=peso_comp,
        mejor_hist=mejor_hist,
        bloquear_sab=bloquear_sab, bloquear_hac=bloquear_hac,
    ))

    escenarios.sort(key=lambda e: (not e.alcanzable, -e.nota_proyectada))

    techo_s_ref = _techo_garantizado(notas_sab, mejor_hist, 73.0)
    techo_h_ref = _techo_garantizado(notas_hac, mejor_hist, prom_hac_act + 10)

    n_alc = sum(1 for e in escenarios if e.alcanzable)
    if n_alc == 0:
        msg_gen = f"Las notas objetivo requieren mejorar respecto al rendimiento actual. Es posible con esfuerzo."
    elif sem_rest <= 2:
        msg_gen = f"Quedan {sem_rest} semanas — hay que actuar esta semana."
    else:
        msg_gen = f"Con {sem_rest} semanas restantes hay {n_alc} camino(s) alcanzable(s) según el historial del estudiante."

    return SimulacionOptimoV2Result(
        objetivo_nota=objetivo_nota, nota_actual=nota_actual, nota_maxima_posible=nota_max,
        semanas_restantes=sem_rest,
        practicas_restantes_est=n_hac, examenes_restantes_est=n_sab,
        techo_practicas=techo_h_ref, techo_examenes=techo_s_ref,
        escenarios=escenarios, ya_alcanza=False, imposible=False,
        mensaje_general=msg_gen,
    )