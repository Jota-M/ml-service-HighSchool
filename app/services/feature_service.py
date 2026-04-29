"""
app/services/feature_service.py — v2 Tiempo Real

Construye el vector de features desde datos acumulados semana a semana.
El modelo recibe el snapshot actual del estudiante y predice la nota final.

Diferencia clave vs v1:
- Antes: necesitabas T1 y T2 completos para predecir T3
- Ahora: predices desde la semana 1 del trimestre actual,
         y la predicción mejora conforme se registran más datos
"""

import numpy as np
from typing import Optional

FEATURE_ORDER = [
    "fiscal", "privado", "rural", "nse_bajo", "nse_alto", "trabaja",
    "dif_muy_alta", "dif_alta", "dif_baja",
    "semana", "pct",
    "asist_acum", "racha", "max_racha", "asist_critica",
    "n_prac", "n_exam", "prom_parcial", "prom_prac", "prom_exam",
    "ultima_nota", "tend_reciente", "min_nota",
    "cn_acum", "cn_sem", "en_riesgo_parcial", "tend_neg",
]

SEMANAS_POR_TRIMESTRE = 13


def construir_features(datos) -> np.ndarray:
    pct = round(datos.semana / SEMANAS_POR_TRIMESTRE, 3)
    todas = list(datos.notas_practicas) + list(datos.notas_examenes)
    prom_parcial  = round(float(np.mean(todas)), 1) if todas else -1
    prom_prac     = round(float(np.mean(datos.notas_practicas)), 1) if datos.notas_practicas else -1
    prom_exam     = round(float(np.mean(datos.notas_examenes)), 1)  if datos.notas_examenes  else -1
    ultima_nota   = todas[-1] if todas else -1
    min_nota      = round(min(todas), 1) if todas else -1
    tend_reciente = round(todas[-1] - todas[-2], 1) if len(todas) >= 2 else -1

    en_riesgo_parcial = 1 if (prom_parcial != -1 and prom_parcial < 51) else 0
    asist_critica     = 1 if datos.asistencia_acumulada_pct < 65 else 0
    tend_neg          = 1 if (tend_reciente != -1 and tend_reciente < -5) else 0

    v = {
        "fiscal":   1 if datos.tipo_colegio == "fiscal"  else 0,
        "privado":  1 if datos.tipo_colegio == "privado" else 0,
        "rural":    1 if datos.zona         == "rural"   else 0,
        "nse_bajo": 1 if datos.nivel_socioeconomico == "bajo"  else 0,
        "nse_alto": 1 if datos.nivel_socioeconomico == "alto"  else 0,
        "trabaja":  1 if datos.trabaja else 0,
        "dif_muy_alta": 1 if datos.dificultad_materia == "muy_alta" else 0,
        "dif_alta":     1 if datos.dificultad_materia == "alta"     else 0,
        "dif_baja":     1 if datos.dificultad_materia == "baja"     else 0,
        "semana": datos.semana, "pct": pct,
        "asist_acum": datos.asistencia_acumulada_pct,
        "racha": datos.racha_inasistencias,
        "max_racha": datos.max_racha_inasistencias,
        "asist_critica": asist_critica,
        "n_prac": len(datos.notas_practicas),
        "n_exam": len(datos.notas_examenes),
        "prom_parcial": prom_parcial,
        "prom_prac": prom_prac,
        "prom_exam": prom_exam,
        "ultima_nota": ultima_nota,
        "tend_reciente": tend_reciente,
        "min_nota": min_nota,
        "cn_acum": datos.conductas_negativas_acumuladas,
        "cn_sem": datos.conductas_negativas_semana,
        "en_riesgo_parcial": en_riesgo_parcial,
        "tend_neg": tend_neg,
    }
    return np.array([[v[f] for f in FEATURE_ORDER]])


def calcular_confianza(semana: int, n_evaluaciones: int) -> dict:
    pct = semana / SEMANAS_POR_TRIMESTRE
    if pct < 0.25 and n_evaluaciones == 0:
        return {"nivel": "muy_baja", "porcentaje_trimestre": round(pct*100,1),
                "mensaje": "Inicio del trimestre sin evaluaciones — predicción basada en contexto"}
    elif pct < 0.25:
        return {"nivel": "baja", "porcentaje_trimestre": round(pct*100,1),
                "mensaje": f"Semana {semana}/13 — la predicción mejorará con más registros"}
    elif pct < 0.50:
        return {"nivel": "media", "porcentaje_trimestre": round(pct*100,1),
                "mensaje": f"Trimestre en curso ({semana}/13 semanas, {n_evaluaciones} evaluaciones)"}
    elif pct < 0.75:
        return {"nivel": "alta", "porcentaje_trimestre": round(pct*100,1),
                "mensaje": f"Más de la mitad ({semana}/13 semanas) — predicción confiable"}
    else:
        return {"nivel": "muy_alta", "porcentaje_trimestre": round(pct*100,1),
                "mensaje": f"Trimestre casi completo ({semana}/13 semanas) — predicción muy precisa"}


def calcular_factores(datos) -> tuple[list[str], list[str]]:
    riesgo, positivos = [], []
    todas = list(datos.notas_practicas) + list(datos.notas_examenes)
    prom = float(np.mean(todas)) if todas else None

    if prom is not None and prom < 51:
        riesgo.append(f"Promedio actual {prom:.1f} — por debajo del mínimo (51)")
    elif prom is not None and prom < 62:
        riesgo.append(f"Promedio parcial bajo: {prom:.1f}")

    if datos.asistencia_acumulada_pct < 65:
        riesgo.append(f"Asistencia crítica: {datos.asistencia_acumulada_pct:.1f}%")
    elif datos.asistencia_acumulada_pct < 75:
        riesgo.append(f"Asistencia baja: {datos.asistencia_acumulada_pct:.1f}%")

    if datos.racha_inasistencias >= 5:
        riesgo.append(f"Racha de {datos.racha_inasistencias} días consecutivos sin asistir")

    if datos.conductas_negativas_acumuladas >= 3:
        riesgo.append(f"{datos.conductas_negativas_acumuladas} conductas negativas acumuladas")

    if len(todas) >= 2:
        tend = todas[-1] - todas[-2]
        if tend < -8:
            riesgo.append(f"Caída fuerte en últimas evaluaciones: {tend:+.1f} pts")
        elif tend < -4:
            riesgo.append(f"Tendencia negativa reciente: {tend:+.1f} pts")
        elif tend > 5:
            positivos.append(f"Mejora en últimas evaluaciones: {tend:+.1f} pts")

    if datos.trabaja:
        riesgo.append("Estudiante trabaja — tiempo de estudio reducido")

    if prom is not None and prom >= 75:
        positivos.append(f"Promedio parcial sólido: {prom:.1f}")
    if datos.asistencia_acumulada_pct >= 88:
        positivos.append(f"Excelente asistencia: {datos.asistencia_acumulada_pct:.1f}%")
    elif datos.asistencia_acumulada_pct >= 80:
        positivos.append(f"Buena asistencia: {datos.asistencia_acumulada_pct:.1f}%")
    if datos.conductas_negativas_acumuladas == 0:
        positivos.append("Sin conductas negativas registradas")

    if not riesgo: riesgo.append("Sin factores de riesgo significativos detectados")
    if not positivos: positivos.append("Continuar monitoreando el progreso")
    return riesgo, positivos


def aplicar_escenario(datos, escenario) -> object:
    d = datos.model_dump()
    if escenario.asistencia_proyectada is not None:
        d["asistencia_acumulada_pct"] = escenario.asistencia_proyectada
    if escenario.nota_proxima_practica is not None:
        d["notas_practicas"] = list(d["notas_practicas"]) + [escenario.nota_proxima_practica]
    if escenario.reducir_conductas_negativas:
        d["conductas_negativas_semana"] = 0
        d["conductas_negativas_acumuladas"] = max(0, d["conductas_negativas_acumuladas"] - 1)
    if escenario.trabaja is not None:
        d["trabaja"] = escenario.trabaja
    return datos.__class__(**d)