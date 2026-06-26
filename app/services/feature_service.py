"""
app/services/feature_service.py — v8.2

Cambios respecto a v8.1:
  - FEATURE_ORDER actualizado a 32 features (31 anteriores + estilo_docente)
  - construir_features() lee estilo_docente del request con getattr default=1
    (1=normal, que es el valor neutro cuando Node.js no lo manda todavía)
  - calcular_nota_final() sin cambios
  - calcular_factores() sin cambios
  - aplicar_escenario() sin cambios
"""

import numpy as np
from app.schemas.prediccion import DatosTiempoRealRequest, EscenarioSimulacion

# ─────────────────────────────────────────
# FEATURE ORDER v8.2 — 32 features
# CRÍTICO: debe coincidir exactamente con FEATURES en train_v8.py
# ─────────────────────────────────────────

FEATURE_ORDER = [
    # ── 11 legacy (eliminadas racha, max_racha, asist_critica) ───────────────
    # racha_inasistencias  → eliminada: en institución privada casi siempre 0
    # max_racha_inasistencias → eliminada: misma razón
    # asist_critica        → eliminada: nunca se activa con asistencia 85-100%
    "semana", "pct",
    "asist_acum",
    "n_prac", "n_exam",
    "prom_parcial", "prom_prac", "prom_exam",
    "ultima_nota", "tend_reciente", "min_nota",
    # ── Historial intertrimestral (7) ─────────────────────────────────────────
    "nota_trim_ant",
    "asist_trim_ant",
    "reprobo_trim_ant",
    "racha_trims_riesgo",
    "mejor_nota_historica",
    "tend_intertrimestral",
    # v8.3: diferencia normalizada (-1 a 1) entre nota de esta materia
    # y promedio de otras materias el trimestre anterior.
    # Ya no es un flag binario redundante con reprobo_trim_ant.
    "reprobo_misma_mat_ant",
    # ── Observaciones pedagógicas (5) ─────────────────────────────────────────
    "n_obs_conducta",
    "n_obs_socioem",
    "n_obs_urgentes",
    "n_logros",
    "ratio_obs_negativas",
    # ── Correlación entre materias (2) ────────────────────────────────────────
    "n_materias_riesgo_sim",
    "reprobo_mat_correlac",
    # ── Nivel educativo y carga horaria (2) ───────────────────────────────────
    "nivel_educativo",
    "horas_grado",
    # ── Régimen de ponderación (1) ────────────────────────────────────────────
    "regimen_pond",
    # ── Factor docente (1) ────────────────────────────────────────────────────
    "estilo_docente",
]

N_FEATURES = len(FEATURE_ORDER)  # 29


def construir_features(datos: DatosTiempoRealRequest) -> np.ndarray:
    """
    Construye el vector de 29 features para el modelo v8.3.

    Cambios respecto a v8.2:
      - Eliminadas racha_inasistencias, max_racha_inasistencias, asist_critica:
        en institución privada con asistencia 85-100% son prácticamente
        constantes y no aportan información útil al modelo.
      - reprobo_misma_mat_ant: ahora es un valor continuo enviado por Node.js
        (diferencia normalizada entre nota de esta materia y promedio de otras).
        Si Node.js todavía manda el valor entero 0/1, funciona igual — el modelo
        lo interpreta como -1 (peor que promedio) o 0 (igual que promedio).
    """
    total_sem = datos.config_periodo.total_semanas
    pct       = round(datos.semana / total_sem, 3)

    notas_sab = datos.notas_saber
    notas_hac = datos.notas_hacer
    todas     = notas_sab + notas_hac

    prom_parcial  = round(float(np.mean(todas)),     1) if todas     else -1
    prom_prac     = round(float(np.mean(notas_hac)), 1) if notas_hac else -1
    prom_exam     = round(float(np.mean(notas_sab)), 1) if notas_sab else -1
    ultima_nota   = todas[-1]                            if todas     else -1
    min_nota      = round(min(todas), 1)                 if todas     else -1
    tend_reciente = round(todas[-1] - todas[-2], 1)      if len(todas) >= 2 else -1

    # ── Observaciones ─────────────────────────────────────────────────────────
    n_conducta = getattr(datos, "n_obs_conducta", 0) or 0
    n_socioem  = getattr(datos, "n_obs_socioem",  0) or 0
    n_urgentes = getattr(datos, "n_obs_urgentes", 0) or 0
    n_logros   = getattr(datos, "n_logros",       0) or 0
    total_obs  = n_conducta + n_socioem + n_urgentes + n_logros
    ratio_neg  = round((n_conducta + n_socioem + n_urgentes) / total_obs, 3) if total_obs > 0 else 0.0

    v = {
        # ── 11 legacy ─────────────────────────────────────────────────────────
        "semana":        datos.semana,
        "pct":           pct,
        "asist_acum":    datos.asistencia_acumulada_pct,
        # racha, max_racha, asist_critica eliminadas en v8.3
        "n_prac":        len(notas_hac),
        "n_exam":        len(notas_sab),
        "prom_parcial":  prom_parcial,
        "prom_prac":     prom_prac,
        "prom_exam":     prom_exam,
        "ultima_nota":   ultima_nota,
        "tend_reciente": tend_reciente,
        "min_nota":      min_nota,
        # ── Historial ─────────────────────────────────────────────────────────
        "nota_trim_ant":         getattr(datos, "nota_trim_ant",         -1) or -1,
        "asist_trim_ant":        getattr(datos, "asist_trim_ant",        -1) or -1,
        "reprobo_trim_ant":      getattr(datos, "reprobo_trim_ant",       0) or 0,
        "racha_trims_riesgo":    getattr(datos, "racha_trims_riesgo",     0) or 0,
        "mejor_nota_historica":  getattr(datos, "mejor_nota_historica",  -1) or -1,
        "tend_intertrimestral":  getattr(datos, "tend_intertrimestral",   0) or 0,
        # v8.3: continuo -1 a 1, no flag binario
        "reprobo_misma_mat_ant": getattr(datos, "reprobo_misma_mat_ant",  0.0),
        # ── Observaciones ─────────────────────────────────────────────────────
        "n_obs_conducta":        n_conducta,
        "n_obs_socioem":         n_socioem,
        "n_obs_urgentes":        n_urgentes,
        "n_logros":              n_logros,
        "ratio_obs_negativas":   ratio_neg,
        # ── Correlación ───────────────────────────────────────────────────────
        "n_materias_riesgo_sim": getattr(datos, "n_materias_riesgo_sim", 0) or 0,
        "reprobo_mat_correlac":  getattr(datos, "reprobo_mat_correlac",  0) or 0,
        # ── Nivel educativo ───────────────────────────────────────────────────
        "nivel_educativo":       getattr(datos, "nivel_educativo",        1) or 1,
        "horas_grado":           getattr(datos, "horas_grado",          168) or 168,
        # ── Régimen ───────────────────────────────────────────────────────────
        "regimen_pond":          getattr(datos, "regimen_pond",           1) or 1,
        # ── Docente ───────────────────────────────────────────────────────────
        "estilo_docente":        getattr(datos, "estilo_docente",         1) or 1,
    }

    return np.array([[v[f] for f in FEATURE_ORDER]])


def calcular_nota_final(
    notas_practicas:         list[float],
    notas_examenes:          list[float],
    ponderaciones:           dict[str, float],
    notas_sab:               list[float] = [],
    notas_hac:               list[float] = [],
    nota_complementaria_pct: float = 0.0,
    peso_complementario:     float = 0.15,
) -> float:
    """
    Sin cambios respecto a v8.1 — sigue usando ponderaciones reales de la BD.
    Node.js manda nota_complementaria_pct (SER+AUT ya colapsados).
    """
    ponds = {k: v / 100 for k, v in ponderaciones.items()}

    _sab = notas_sab if notas_sab else notas_examenes
    _hac = notas_hac if notas_hac else notas_practicas

    ns = round(float(np.mean(_sab)), 1) if _sab else 0.0
    nh = round(float(np.mean(_hac)), 1) if _hac else 0.0

    peso_sab = ponds.get("SAB", 0.45)
    peso_hac = ponds.get("HAC", 0.40)

    nf = ns * peso_sab + nh * peso_hac + nota_complementaria_pct * peso_complementario
    return round(max(1.0, min(100.0, nf)), 1)


def calcular_confianza(semana: int, total_semanas: int, n_evaluaciones: int) -> dict:
    """Sin cambios."""
    pct = semana / total_semanas

    if pct < 0.25 and n_evaluaciones == 0:
        return {
            "nivel": "muy_baja",
            "porcentaje_periodo": round(pct * 100, 1),
            "mensaje": "Inicio del período sin evaluaciones — predicción basada en asistencia",
        }
    elif pct < 0.25:
        return {
            "nivel": "baja",
            "porcentaje_periodo": round(pct * 100, 1),
            "mensaje": f"Semana {semana}/{total_semanas} — la predicción mejorará con más datos",
        }
    elif pct < 0.50:
        return {
            "nivel": "media",
            "porcentaje_periodo": round(pct * 100, 1),
            "mensaje": f"Período en curso ({semana}/{total_semanas} sem, {n_evaluaciones} evaluaciones)",
        }
    elif pct < 0.75:
        return {
            "nivel": "alta",
            "porcentaje_periodo": round(pct * 100, 1),
            "mensaje": f"Más de la mitad del período ({semana}/{total_semanas} sem) — predicción confiable",
        }
    else:
        return {
            "nivel": "muy_alta",
            "porcentaje_periodo": round(pct * 100, 1),
            "mensaje": f"Período casi completo ({semana}/{total_semanas} sem) — predicción muy precisa",
        }


def calcular_factores(datos: DatosTiempoRealRequest) -> tuple[list[str], list[str]]:
    """
    Detecta factores de riesgo y positivos.
    v8.2: incluye factor docente cuando está disponible.
    """
    riesgo, positivos = [], []

    notas_sab = datos.notas_saber
    notas_hac = datos.notas_hacer
    todas     = notas_sab + notas_hac
    prom      = float(np.mean(todas)) if todas else None
    total     = datos.config_periodo.total_semanas

    # ── Promedio parcial ──────────────────────────────────────────────────────
    if prom is not None:
        if prom < 45:
            riesgo.append(f"Promedio actual {prom:.1f} — significativamente bajo (mínimo 51)")
        elif prom < 51:
            riesgo.append(f"Promedio parcial en riesgo: {prom:.1f} (mínimo 51 para aprobar)")
        elif prom < 60:
            riesgo.append(f"Promedio parcial ajustado: {prom:.1f} — margen reducido")
        elif prom >= 75:
            positivos.append(f"Promedio parcial sólido: {prom:.1f}")

    # ── Saber vs Hacer ────────────────────────────────────────────────────────
    if notas_sab and notas_hac:
        prom_sab = float(np.mean(notas_sab))
        prom_hac = float(np.mean(notas_hac))
        if prom_sab < 51 and prom_hac >= 60:
            riesgo.append(f"Bajo rendimiento en Saber: {prom_sab:.1f} (Hacer está en {prom_hac:.1f})")
        elif prom_hac < 51 and prom_sab >= 60:
            riesgo.append(f"Bajo rendimiento en Hacer: {prom_hac:.1f} (Saber está en {prom_sab:.1f})")

    # ── Asistencia ────────────────────────────────────────────────────────────
    asist = datos.asistencia_acumulada_pct
    if asist < 75:
        riesgo.append(f"Asistencia crítica: {asist:.1f}% (por debajo del mínimo legal 75%)")
    elif asist < 82:
        riesgo.append(f"Asistencia baja: {asist:.1f}% — riesgo de perder regularidad")
    elif asist >= 92:
        positivos.append(f"Excelente asistencia: {asist:.1f}%")
    elif asist >= 85:
        positivos.append(f"Buena asistencia: {asist:.1f}%")

    # ── Racha de inasistencias ────────────────────────────────────────────────
    if datos.racha_inasistencias >= 5:
        riesgo.append(f"Racha de {datos.racha_inasistencias} días consecutivos sin asistir")

    # ── Tendencia reciente ────────────────────────────────────────────────────
    if len(todas) >= 2:
        tend = todas[-1] - todas[-2]
        if tend < -8:
            riesgo.append(f"Caída fuerte en últimas evaluaciones: {tend:+.1f} pts")
        elif tend < -4:
            riesgo.append(f"Tendencia negativa reciente: {tend:+.1f} pts")
        elif tend > 5:
            positivos.append(f"Mejora en últimas evaluaciones: {tend:+.1f} pts")

    # ── Historial intertrimestral ─────────────────────────────────────────────
    racha_trims = getattr(datos, "racha_trims_riesgo", 0) or 0
    reprobo_ant = getattr(datos, "reprobo_trim_ant",   0) or 0
    reprobo_mat = getattr(datos, "reprobo_misma_mat_ant", 0) or 0
    tend_inter  = getattr(datos, "tend_intertrimestral",  0) or 0

    if racha_trims >= 3:
        riesgo.append(
            f"Patrón crónico: {racha_trims} trimestres consecutivos en riesgo "
            f"— requiere intervención institucional"
        )
    elif racha_trims == 2:
        riesgo.append("Segundo trimestre consecutivo en riesgo en esta materia")
    elif racha_trims == 1 and reprobo_ant:
        riesgo.append("Reprobó esta materia el trimestre anterior")

    if tend_inter == 1 and reprobo_ant:
        positivos.append("Mejorando respecto al trimestre anterior a pesar del historial")
    elif tend_inter == -1:
        riesgo.append("Tendencia descendente entre trimestres")

    # ── Observaciones pedagógicas ─────────────────────────────────────────────
    n_urgentes = getattr(datos, "n_obs_urgentes", 0) or 0
    n_conducta = getattr(datos, "n_obs_conducta", 0) or 0
    n_logros_v = getattr(datos, "n_logros",       0) or 0

    if n_urgentes >= 1:
        riesgo.append(f"{n_urgentes} observación(es) urgente(s) registrada(s) este trimestre")
    if n_conducta >= 3:
        riesgo.append(f"Alto número de observaciones de conducta: {n_conducta}")
    if n_logros_v >= 2:
        positivos.append(f"{n_logros_v} logros destacados registrados por el docente")

    # ── Correlación entre materias ────────────────────────────────────────────
    n_mat_riesgo = getattr(datos, "n_materias_riesgo_sim", 0) or 0
    if n_mat_riesgo >= 3:
        riesgo.append(
            f"En riesgo simultáneo en {n_mat_riesgo} materias "
            f"— patrón de dificultad generalizada"
        )
    elif n_mat_riesgo >= 2:
        riesgo.append(f"En riesgo en {n_mat_riesgo} materias al mismo tiempo")

    # ── Factor docente — NUEVO v8.2 ───────────────────────────────────────────
    estilo_doc = getattr(datos, "estilo_docente", 1) or 1
    if estilo_doc == 2:  # estricto
        positivos.append("Docente con criterio estricto — aprobación tiene mayor valor")
    elif estilo_doc == 3:  # inconsistente
        riesgo.append("Docente con criterio inconsistente — variabilidad alta en calificaciones")

    # ── Poco tiempo para recuperarse ──────────────────────────────────────────
    pct_periodo = datos.semana / total
    if pct_periodo > 0.7 and prom is not None and prom < 60:
        riesgo.append(
            f"Poco tiempo para recuperarse: semana {datos.semana}/{total} "
            f"con promedio {prom:.1f}"
        )

    if not riesgo:
        riesgo.append("Sin factores de riesgo significativos detectados")
    if not positivos:
        positivos.append("Continuar monitoreando el progreso")

    return riesgo, positivos


def aplicar_escenario(
    datos: DatosTiempoRealRequest,
    escenario: "EscenarioSimulacion",
) -> DatosTiempoRealRequest:
    """
    Sin cambios respecto a v8.1 — los campos nuevos (incluyendo estilo_docente)
    se heredan del datos original, que es correcto: una simulación de
    "qué pasa si mejora la asistencia" no cambia el docente ni el historial.
    """
    d = datos.model_dump()

    if escenario.asistencia_proyectada is not None:
        d["asistencia_acumulada_pct"] = escenario.asistencia_proyectada

    if escenario.nota_proxima_practica is not None:
        if d.get("notas_hac"):
            d["notas_hac"] = list(d["notas_hac"]) + [escenario.nota_proxima_practica]
        else:
            d["notas_practicas"] = list(d["notas_practicas"]) + [escenario.nota_proxima_practica]

    if escenario.nota_proximo_examen is not None:
        if d.get("notas_sab"):
            d["notas_sab"] = list(d["notas_sab"]) + [escenario.nota_proximo_examen]
        else:
            d["notas_examenes"] = list(d["notas_examenes"]) + [escenario.nota_proximo_examen]

    if escenario.semanas_adicionales is not None:
        nueva_semana = min(
            d["semana"] + escenario.semanas_adicionales,
            d["config_periodo"]["total_semanas"],
        )
        d["semana"] = nueva_semana

    return datos.__class__(**d)