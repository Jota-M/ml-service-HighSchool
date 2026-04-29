"""
app/services/gemini_service.py
"""

import logging
import json
import re
import httpx
from typing import Optional

from app.config import get_settings
from app.schemas.prediccion import (
    DatosTiempoRealRequest,
    ResultadoModelo,
    AnalisisGemini,
    ResultadoEscenario,
)

logger = logging.getLogger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def _prompt_prediccion(datos: DatosTiempoRealRequest, resultado: ResultadoModelo) -> str:
    clasificaciones = {
        "ED": "En Desarrollo (reprobado, nota < 51)",
        "DA": "Desarrollo Aceptable (51-68)",
        "DO": "Desarrollo Óptimo (69-84)",
        "DP": "Desarrollo Pleno (85-100)",
    }
    todas       = list(datos.notas_practicas) + list(datos.notas_examenes)
    prom_actual = round(sum(todas) / len(todas), 1) if todas else None
    prom_texto  = f"{prom_actual}" if prom_actual is not None else "Sin evaluaciones aún"
    factores_r  = ", ".join(resultado.factores_riesgo)    if resultado.factores_riesgo    else "Ninguno"
    factores_p  = ", ".join(resultado.factores_positivos) if resultado.factores_positivos else "Ninguno"

    return f"""Eres un sistema de apoyo pedagógico para docentes en Bolivia.
Analiza el rendimiento de este estudiante y proporciona orientación práctica.

DATOS DEL ESTUDIANTE:
- Materia: {datos.materia} (dificultad: {datos.dificultad_materia})
- Tipo de colegio: {datos.tipo_colegio} | Zona: {datos.zona}
- Trabaja: {"Sí" if datos.trabaja else "No"}
- Semana actual: {datos.semana}/13 del trimestre {datos.trimestre}

RENDIMIENTO ACTUAL:
- Promedio parcial: {prom_texto}
- Asistencia acumulada: {datos.asistencia_acumulada_pct:.1f}%
- Racha de inasistencias: {datos.racha_inasistencias} días
- Conductas negativas acumuladas: {datos.conductas_negativas_acumuladas}
- Prácticas registradas: {len(datos.notas_practicas)} | Exámenes: {len(datos.notas_examenes)}

PREDICCIÓN DEL MODELO ML:
- Probabilidad de reprobar: {resultado.probabilidad_reprobar*100:.1f}%
- Nivel de riesgo: {resultado.nivel_riesgo.value.upper()}
- Nota estimada final: {resultado.nota_estimada_final:.1f} ({clasificaciones.get(resultado.clasificacion_estimada.value, "")})
- Confianza: {resultado.confianza.nivel.value} ({resultado.confianza.mensaje})

FACTORES DE RIESGO DETECTADOS: {factores_r}
FACTORES POSITIVOS: {factores_p}

Responde ÚNICAMENTE con un JSON válido, sin texto adicional, con esta estructura exacta:
{{
  "explicacion": "Explicación clara y empática en 2-3 oraciones para el docente, mencionando los factores más importantes",
  "recomendaciones": [
    "Acción concreta 1 que puede tomar el docente",
    "Acción concreta 2",
    "Acción concreta 3"
  ],
  "alerta_urgente": true o false,
  "mensaje_alerta": "Mensaje de alerta solo si alerta_urgente es true, si no pon null"
}}"""


def _prompt_simulacion(
    datos: DatosTiempoRealRequest,
    situacion_actual: ResultadoModelo,
    escenarios: list[ResultadoEscenario]
) -> str:
    esc_texto = "\n".join([
        f"- {e.descripcion}: riesgo {e.probabilidad_reprobar*100:.1f}% "
        f"(cambio: {e.cambio_probabilidad:+.1f}%), nota estimada {e.nota_estimada_final:.1f}"
        for e in escenarios
    ])
    return f"""Eres un sistema de apoyo pedagógico para docentes en Bolivia.
Analiza estos escenarios de intervención para un estudiante en {datos.materia}.

SITUACIÓN ACTUAL:
- Riesgo de reprobar: {situacion_actual.probabilidad_reprobar*100:.1f}%
- Nota estimada: {situacion_actual.nota_estimada_final:.1f}
- Nivel: {situacion_actual.nivel_riesgo.value.upper()}

ESCENARIOS SIMULADOS:
{esc_texto}

Responde con una recomendación concisa (máximo 3 oraciones) indicando:
1. Qué escenario tiene mayor impacto positivo
2. Qué acción específica debería priorizar el docente
3. Si la situación requiere también involucrar a los padres

Responde SOLO con el texto de la recomendación, sin JSON ni formato especial."""


async def _llamar_gemini(prompt: str) -> Optional[str]:
    settings = get_settings()

    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY no configurada — saltando análisis Gemini")
        return None

    url = GEMINI_URL.format(model=settings.gemini_model, key=settings.gemini_api_key)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":     0.3,
            "maxOutputTokens": 1024,  # FIX: era 600, muy ajustado para JSON completo
            "topP":            0.8,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    except httpx.TimeoutException:
        logger.warning("Gemini API timeout — continuando sin análisis")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"Gemini API error {e.response.status_code} — continuando sin análisis")
        return None
    except Exception as e:
        logger.warning(f"Gemini error inesperado: {e} — continuando sin análisis")
        return None


def _limpiar_json(texto: str) -> str:
    """
    Extrae JSON limpio de una respuesta que puede contener:
    - Bloques de markdown (```json ... ``` o ``` ... ```)
    - Texto introductorio antes del JSON
    - Saltos de línea y espacios extras
    """
    texto = texto.strip()

    # Intenta extraer el bloque entre ```json ... ``` o ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Si no hay backticks, busca el primer objeto JSON válido en el texto
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if match:
        return match.group(0).strip()

    return texto


async def analizar_prediccion(
    datos: DatosTiempoRealRequest,
    resultado: ResultadoModelo
) -> Optional[AnalisisGemini]:
    prompt = _prompt_prediccion(datos, resultado)
    texto  = await _llamar_gemini(prompt)

    if not texto:
        return None

    try:
        texto_limpio = _limpiar_json(texto)
        data = json.loads(texto_limpio)

        return AnalisisGemini(
            explicacion     = data.get("explicacion", ""),
            recomendaciones = data.get("recomendaciones", []),
            alerta_urgente  = data.get("alerta_urgente", False),
            mensaje_alerta  = data.get("mensaje_alerta"),
        )

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Error parseando respuesta de Gemini: {e}")
        logger.debug(f"Texto recibido de Gemini: {texto[:300]}")
        return AnalisisGemini(
            explicacion     = "No se pudo procesar el análisis automático. Revise los factores de riesgo manualmente.",
            recomendaciones = ["Consulte con el equipo pedagógico para definir acciones de apoyo."],
            alerta_urgente  = resultado.nivel_riesgo.value in ["alto", "critico"],
            mensaje_alerta  = None,
        )


async def analizar_escenarios(
    datos: DatosTiempoRealRequest,
    situacion_actual: ResultadoModelo,
    escenarios: list[ResultadoEscenario]
) -> Optional[str]:
    prompt = _prompt_simulacion(datos, situacion_actual, escenarios)
    return await _llamar_gemini(prompt)


async def verificar_disponibilidad() -> bool:
    settings = get_settings()
    if not settings.gemini_api_key:
        return False
    texto = await _llamar_gemini("Responde solo: ok")
    return texto is not None