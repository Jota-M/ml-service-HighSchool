"""
app/services/gemini_service.py — v8.3

Cambios respecto a v8.2:
  - generar_recursos_externos: reemplazada completamente por la versión v8.3
    que ahora usa un flujo de dos pasos:
      1. Gemini genera 2-3 search_queries pedagógicamente precisos para el tema.
      2. YouTube Data API v3 busca con esos queries y retorna IDs de videos REALES.
    Esto elimina el problema de URLs inventadas que tenía la v8.2 (donde Gemini
    generaba URLs de YouTube que no existían).
    URLs resultantes: https://www.youtube.com/watch?v=<ID_REAL> — siempre válidas.
  - Nueva función interna: _buscar_video_youtube (cliente YouTube Data API v3).
  - Configuración: agregar YOUTUBE_API_KEY al .env.
    Obtener en: https://console.cloud.google.com → YouTube Data API v3
    Costo: 100 unidades por búsqueda · 10,000 gratis/día → 100 búsquedas/día.
  - Todo lo demás sin cambios respecto a v8.2.
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
    RecursoRecomendado,
    ResultadoEscenario,
)

logger = logging.getLogger(__name__)

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/{model}:generateContent?key={key}"
)

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

# ─────────────────────────────────────────
# HELPERS DE FORMATO
# ─────────────────────────────────────────

def _formatear_materiales(datos: DatosTiempoRealRequest) -> str:
    if not datos.materiales_disponibles:
        return ""
    lineas = ["", "MATERIALES DISPONIBLES EN EL REPOSITORIO:"]
    for m in datos.materiales_disponibles:
        tema_str = f" [Tema: {m.tema_titulo}]" if m.tema_titulo else ""
        dest_str = " ⭐ DESTACADO" if m.es_destacado else ""
        desc_str = f" — {m.descripcion}" if m.descripcion else ""
        lineas.append(f"  - [ID:{m.id}] [{m.tipo_codigo}]{dest_str} \"{m.titulo}\"{tema_str}{desc_str}")
    lineas.append(
        "\nAl recomendar recursos, SIEMPRE referencia los materiales del repositorio "
        "usando su ID exacto. Solo sugiere externos si no hay opciones relevantes."
    )
    return "\n".join(lineas)


def _estructura_recursos_json(tiene_materiales: bool) -> str:
    if tiene_materiales:
        return """  "recursos_sugeridos": [
    {
      "material_id": 123,
      "titulo": "Título del material del repositorio",
      "tipo": "PDF",
      "tema_titulo": "Nombre del tema o null",
      "url": null,
      "razon": "Por qué ayuda a este estudiante"
    }
  ]"""
    else:
        return """  "recursos_sugeridos": [
    {
      "material_id": null,
      "titulo": "Nombre descriptivo del recurso",
      "tipo": "VIDEO",
      "tema_titulo": "Tema que refuerza",
      "url": null,
      "search_query": "palabras clave para buscar el recurso",
      "razon": "Por qué ayuda a este estudiante"
    }
  ]"""


def _formatear_historial(datos: DatosTiempoRealRequest) -> str:
    nota_ant         = getattr(datos, "nota_trim_ant",         -1)
    asist_ant        = getattr(datos, "asist_trim_ant",        -1)
    racha            = getattr(datos, "racha_trims_riesgo",     0)
    mejor            = getattr(datos, "mejor_nota_historica",  -1)
    tend             = getattr(datos, "tend_intertrimestral",   0)
    reprobo_m        = getattr(datos, "reprobo_misma_mat_ant",  0)
    reprobo_trim_ant = getattr(datos, "reprobo_trim_ant",       0)

    if nota_ant < 0:
        return "HISTORIAL: Primer trimestre — sin datos históricos previos."

    tend_txt = {-1: "↓ descendente", 0: "→ estable", 1: "↑ ascendente"}.get(tend, "desconocida")
    lineas = [
        "HISTORIAL INTERTRIMESTRAL:",
        f"  - Nota trimestre anterior: {nota_ant:.1f} ({'reprobó' if nota_ant < 51 else 'aprobó'})",
        f"  - Asistencia trimestre anterior: {asist_ant:.1f}%",
        f"  - Trimestres consecutivos en riesgo: {racha}",
        f"  - Mejor nota histórica en esta materia: {mejor:.1f}" if mejor >= 0 else "  - Sin nota histórica previa",
        f"  - Tendencia entre trimestres: {tend_txt}",
        f"  - Reprobó esta materia el trimestre anterior: {'Sí' if reprobo_trim_ant else 'No'}",
    ]

    if racha >= 3:
        lineas.append(f"  ⚠ PATRÓN CRÓNICO: {racha} trimestres seguidos en riesgo")
    elif racha == 2:
        lineas.append("  ⚠ Segundo trimestre consecutivo en riesgo")

    return "\n".join(lineas)


def _formatear_observaciones(datos: DatosTiempoRealRequest) -> str:
    n_conducta = getattr(datos, "n_obs_conducta", 0) or 0
    n_socioem  = getattr(datos, "n_obs_socioem",  0) or 0
    n_urgentes = getattr(datos, "n_obs_urgentes", 0) or 0
    n_logros   = getattr(datos, "n_logros",       0) or 0
    total      = n_conducta + n_socioem + n_urgentes + n_logros

    if total == 0:
        return "OBSERVACIONES PEDAGÓGICAS: Sin observaciones registradas este trimestre."

    lineas = ["OBSERVACIONES PEDAGÓGICAS ESTE TRIMESTRE:"]
    if n_urgentes > 0:
        lineas.append(f"  ⚠ URGENTES: {n_urgentes} — requieren atención inmediata")
    if n_conducta > 0:
        lineas.append(f"  - Conducta: {n_conducta}")
    if n_socioem > 0:
        lineas.append(f"  - Socioemocionales: {n_socioem}")
    if n_logros > 0:
        lineas.append(f"  ✓ Logros destacados: {n_logros}")

    n_mat_riesgo = getattr(datos, "n_materias_riesgo_sim", 0) or 0
    if n_mat_riesgo >= 2:
        lineas.append(f"  ⚠ En riesgo simultáneo en {n_mat_riesgo} materias")

    return "\n".join(lineas)


# ─────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────

def _prompt_prediccion(
    datos: DatosTiempoRealRequest,
    resultado: ResultadoModelo,
) -> str:
    clasificaciones = {
        "ED": "En Desarrollo (reprobado, nota < 51)",
        "DA": "Desarrollo Aceptable (51-68)",
        "DO": "Desarrollo Óptimo (69-84)",
        "DP": "Desarrollo Pleno (85-100)",
    }

    notas_sab   = datos.notas_saber
    notas_hac   = datos.notas_hacer
    todas       = notas_sab + notas_hac
    prom_actual = round(sum(todas) / len(todas), 1) if todas else None
    prom_texto  = f"{prom_actual}" if prom_actual is not None else "Sin evaluaciones aún"
    factores_r  = ", ".join(resultado.factores_riesgo)    or "Ninguno"
    factores_p  = ", ".join(resultado.factores_positivos) or "Ninguno"

    ponds        = datos.config_periodo.ponderaciones
    ponds_texto  = " | ".join(f"{k}={v:.0f}%" for k, v in ponds.items())
    total_sem    = datos.config_periodo.total_semanas
    tipo_periodo = "trimestre" if total_sem >= 12 else "bimestre" if total_sem <= 10 else "período"

    historial_texto     = _formatear_historial(datos)
    observaciones_texto = _formatear_observaciones(datos)
    materiales_texto    = _formatear_materiales(datos)
    tiene_materiales    = len(datos.materiales_disponibles) > 0
    recursos_schema     = _estructura_recursos_json(tiene_materiales)

    nivel_edu = "Secundaria" if getattr(datos, "nivel_educativo", 1) == 1 else "Primaria"
    horas_g   = getattr(datos, "horas_grado", 168)

    return f"""Eres un sistema de apoyo pedagógico para docentes en Bolivia.
Analiza el rendimiento de este estudiante y proporciona orientación práctica.

DATOS DEL ESTUDIANTE:
- Materia: {datos.materia} ({nivel_edu} — {horas_g} horas/año)
- {tipo_periodo.capitalize()} {datos.trimestre} — Semana {datos.semana}/{total_sem}
- Ponderación del período: {ponds_texto}

RENDIMIENTO ACTUAL:
- Promedio parcial: {prom_texto}
- Notas Saber (SAB): {len(notas_sab)} registradas — {notas_sab}
- Notas Hacer (HAC): {len(notas_hac)} registradas — {notas_hac}
- Asistencia acumulada: {datos.asistencia_acumulada_pct:.1f}%
- Racha de inasistencias: {datos.racha_inasistencias} días consecutivos

PREDICCIÓN DEL MODELO ML:
- Probabilidad de reprobar: {resultado.probabilidad_reprobar * 100:.1f}%
- Nivel de riesgo: {resultado.nivel_riesgo.value.upper()}
- Nota estimada final: {resultado.nota_estimada_final:.1f} ({clasificaciones.get(resultado.clasificacion_estimada.value, "")})
- Confianza: {resultado.confianza.nivel.value} ({resultado.confianza.mensaje})

FACTORES DE RIESGO: {factores_r}
FACTORES POSITIVOS: {factores_p}

{historial_texto}

{observaciones_texto}
{materiales_texto}

Responde ÚNICAMENTE con un JSON válido, sin texto adicional, con esta estructura exacta:
{{
  "explicacion": "Explicación clara y empática en 2-3 oraciones para el docente. Si hay patrón crónico o urgentes, mencionarlo explícitamente.",
  "recomendaciones": [
    "Acción concreta 1 que puede tomar el docente esta semana",
    "Acción concreta 2",
    "Acción concreta 3"
  ],
{recursos_schema},
  "alerta_urgente": true o false,
  "mensaje_alerta": "Mensaje de alerta si alerta_urgente es true, sino null"
}}"""


def _prompt_simulacion(
    datos: DatosTiempoRealRequest,
    situacion_actual: ResultadoModelo,
    escenarios: list[ResultadoEscenario],
) -> str:
    esc_texto = "\n".join([
        f"- {e.descripcion}: riesgo {e.probabilidad_reprobar * 100:.1f}% "
        f"(cambio: {e.cambio_probabilidad:+.1f}pp), nota estimada {e.nota_estimada_final:.1f}"
        for e in escenarios
    ])
    total_sem    = datos.config_periodo.total_semanas
    tipo_periodo = "trimestre" if total_sem >= 12 else "bimestre"
    semanas_rest = total_sem - datos.semana
    racha_trims  = getattr(datos, "racha_trims_riesgo", 0) or 0
    contexto_hist = f"(lleva {racha_trims} trimestre(s) en riesgo)" if racha_trims > 0 else ""

    return f"""Eres un sistema de apoyo pedagógico para docentes en Bolivia.
Analiza estos escenarios de intervención para un estudiante en {datos.materia}.

SITUACIÓN ACTUAL (semana {datos.semana}/{total_sem} del {tipo_periodo}) {contexto_hist}:
- Riesgo de reprobar: {situacion_actual.probabilidad_reprobar * 100:.1f}%
- Nota estimada: {situacion_actual.nota_estimada_final:.1f}
- Nivel: {situacion_actual.nivel_riesgo.value.upper()}
- Semanas restantes: {semanas_rest}

ESCENARIOS SIMULADOS:
{esc_texto}

Responde con una recomendación concisa (máximo 3 oraciones) indicando:
1. Qué escenario tiene mayor impacto positivo y por qué
2. Qué acción específica debería priorizar el docente esta semana
3. Si con {semanas_rest} semanas restantes la situación requiere involucrar a los padres

Responde SOLO con el texto de la recomendación, sin JSON ni formato especial."""


def _prompt_plan_recuperacion(
    datos: DatosTiempoRealRequest,
    resultado: ResultadoModelo,
) -> str:
    total_sem    = datos.config_periodo.total_semanas
    semanas_rest = total_sem - datos.semana
    tipo_periodo = "trimestre" if total_sem >= 12 else "bimestre"
    ponds        = datos.config_periodo.ponderaciones
    ponds_texto  = " | ".join(f"{k}={v:.0f}%" for k, v in ponds.items())

    notas_sab = datos.notas_saber
    notas_hac = datos.notas_hacer
    todas     = notas_sab + notas_hac
    prom      = round(sum(todas) / len(todas), 1) if todas else None

    historial_texto     = _formatear_historial(datos)
    observaciones_texto = _formatear_observaciones(datos)
    materiales_texto    = _formatear_materiales(datos)

    racha_trims  = getattr(datos, "racha_trims_riesgo", 0) or 0
    nivel_edu    = "Secundaria" if getattr(datos, "nivel_educativo", 1) == 1 else "Primaria"

    instruccion_extra = ""
    if racha_trims >= 3:
        instruccion_extra = (
            f"\nIMPORTANTE: Este estudiante lleva {racha_trims} trimestres consecutivos en riesgo. "
            "El plan debe incluir acciones de intervención institucional (dirección, orientación, padres) "
            "además de las acciones pedagógicas habituales."
        )
    elif racha_trims == 2:
        instruccion_extra = (
            "\nNOTA: Es el segundo trimestre consecutivo en riesgo. "
            "El plan debe incluir notificación formal a los padres."
        )

    return f"""Eres un asesor pedagógico experto en el sistema educativo boliviano.
Diseña un plan de recuperación concreto para este estudiante.

CONTEXTO:
- Materia: {datos.materia} ({nivel_edu})
- {tipo_periodo.capitalize()} {datos.trimestre} — Semana actual: {datos.semana}/{total_sem}
- Semanas restantes: {semanas_rest}
- Ponderación: {ponds_texto}

SITUACIÓN ACADÉMICA:
- Promedio actual: {prom if prom is not None else "Sin evaluaciones"}
- Notas Saber (SAB): {len(notas_sab)} registradas — {notas_sab}
- Notas Hacer (HAC): {len(notas_hac)} registradas — {notas_hac}
- Asistencia: {datos.asistencia_acumulada_pct:.1f}%
- Nota estimada final: {resultado.nota_estimada_final:.1f}
- Riesgo: {resultado.nivel_riesgo.value.upper()} ({resultado.probabilidad_reprobar * 100:.1f}%)
- Necesita llegar a 51 para aprobar

{historial_texto}

{observaciones_texto}
{materiales_texto}
{instruccion_extra}

Responde ÚNICAMENTE con un JSON válido con esta estructura:
{{
  "objetivo": "Qué nota necesita alcanzar y por qué es alcanzable (1 oración)",
  "plan_semanal": [
    {{
      "semana": número de semana,
      "accion_docente": "Qué hace el docente esa semana",
      "accion_estudiante": "Qué debe hacer el estudiante",
      "meta": "Meta medible para esa semana",
      "material_id_sugerido": ID_del_repositorio_o_null
    }}
  ],
  "nota_proyectada": número entre 1 y 100,
  "involucrar_padres": true o false,
  "mensaje_padres": "Qué decirle a los padres si involucrar_padres es true, sino null",
  "involucrar_direccion": true o false,
  "mensaje_direccion": "Qué reportar a dirección si aplica, sino null"
}}

Genera el plan solo para las próximas {min(semanas_rest, 4)} semanas más críticas."""


def _prompt_analisis_clase(
    materia: str,
    trimestre: int,
    semana: int,
    total_semanas: int,
    resumen_clase: dict,
) -> str:
    tipo_periodo = "trimestre" if total_semanas >= 12 else "bimestre"
    return f"""Eres un asesor pedagógico experto en el sistema educativo boliviano.
Analiza el rendimiento general de esta clase.

CLASE: {materia} — {tipo_periodo.capitalize()} {trimestre}, Semana {semana}/{total_semanas}

RESUMEN DE LA CLASE ({resumen_clase.get('total_estudiantes', 0)} estudiantes):
- En riesgo CRÍTICO: {resumen_clase.get('critico', 0)} estudiantes
- En riesgo ALTO: {resumen_clase.get('alto', 0)} estudiantes
- En riesgo MEDIO: {resumen_clase.get('medio', 0)} estudiantes
- Sin riesgo: {resumen_clase.get('bajo', 0)} estudiantes
- Promedio de la clase: {resumen_clase.get('promedio_clase', 'N/A')}
- Asistencia promedio: {resumen_clase.get('asistencia_promedio', 'N/A')}%
- Tasa de reprobación proyectada: {resumen_clase.get('pct_riesgo', 0):.1f}%

Responde ÚNICAMENTE con un JSON válido:
{{
  "diagnostico": "Diagnóstico general de la clase en 2 oraciones",
  "patron_principal": "El patrón más preocupante detectado",
  "acciones_grupo": [
    "Acción para toda la clase 1",
    "Acción para toda la clase 2"
  ],
  "acciones_individuales": "Qué hacer con los estudiantes en riesgo crítico/alto",
  "alerta_institucional": true o false,
  "mensaje_institucional": "Si alerta_institucional es true, qué reportar a dirección"
}}"""


# ─────────────────────────────────────────
# CLIENTE GEMINI
# ─────────────────────────────────────────

async def _llamar_gemini(prompt: str, max_tokens: int = 1500) -> Optional[str]:
    settings = get_settings()
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY no configurada — saltando análisis Gemini")
        return None

    url = GEMINI_URL.format(model=settings.gemini_model, key=settings.gemini_api_key)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":     0.3,
            "maxOutputTokens": max_tokens,
            "topP":            0.8,
        },
    }

    if "2.5" in settings.gemini_model and "flash" in settings.gemini_model.lower():
        payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 512}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data      = response.json()
            candidate = data["candidates"][0]
            if "content" not in candidate:
                logger.warning(f"Gemini sin content — finishReason: {candidate.get('finishReason')}")
                return None
            parts      = candidate["content"].get("parts", [])
            text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought", False)]
            return text_parts[-1] if text_parts else None
    except httpx.TimeoutException:
        logger.warning("Gemini API timeout")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"Gemini API error {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"Gemini error inesperado: {e}")
        return None


def _limpiar_json(texto: str) -> str:
    texto = texto.strip()

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto, re.DOTALL)
    if match:
        texto = match.group(1).strip()
    else:
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if match:
            texto = match.group(0).strip()

    texto = re.sub(r",\s*([}\]])", r"\1", texto)

    def limpiar_string(m):
        contenido = m.group(1)
        contenido = contenido.replace('\n', '\\n').replace('\r', '')
        return f'"{contenido}"'

    texto = re.sub(r'"((?:[^"\\]|\\.)*)"', limpiar_string, texto, flags=re.DOTALL)

    return texto


def _parsear_recursos_sugeridos(raw_recursos: list, materiales_disponibles: list) -> list:
    ids_validos = {m.id for m in materiales_disponibles}
    resultado   = []
    for r in raw_recursos:
        if not isinstance(r, dict):
            continue
        mid = r.get("material_id")
        if not isinstance(mid, int):
            mid = None
        if mid is not None and mid not in ids_validos:
            logger.warning(f"Gemini referenció material_id={mid} no válido")
            mid = None
        url          = r.get("url") or None
        search_query = r.get("search_query") or None
        if url and any(x in url.lower() for x in ["ejemplo", "example", "sample", "placeholder"]):
            url = None
        try:
            resultado.append(RecursoRecomendado(
                material_id  = mid,
                titulo       = str(r.get("titulo", "Material sugerido")),
                tipo         = str(r.get("tipo", "Otro")),
                tema_titulo  = r.get("tema_titulo") or None,
                url          = url,
                search_query = search_query,
                razon        = str(r.get("razon", "Recomendado para reforzar los temas con dificultad")),
            ))
        except Exception as e:
            logger.warning(f"Error parseando recurso: {e}")
    return resultado


# ─────────────────────────────────────────
# FUNCIONES PÚBLICAS
# ─────────────────────────────────────────

async def analizar_prediccion(
    datos: DatosTiempoRealRequest,
    resultado: ResultadoModelo,
) -> Optional[AnalisisGemini]:

    racha = getattr(datos, "racha_trims_riesgo", 0) or 0
    logger.info("=" * 60)
    logger.info(f"[GEMINI v8.3] Estudiante {datos.estudiante_id} | {datos.materia}")
    logger.info(f"[GEMINI] Semana {datos.semana}/{datos.config_periodo.total_semanas}")
    logger.info(f"[GEMINI] Riesgo: {resultado.nivel_riesgo.value} | Nota est: {resultado.nota_estimada_final}")
    logger.info(f"[GEMINI] Racha trimestres en riesgo: {racha}")
    logger.info(f"[GEMINI] Obs urgentes: {getattr(datos, 'n_obs_urgentes', 0)}")
    logger.info("=" * 60)

    prompt  = _prompt_prediccion(datos, resultado)
    max_tok = 1500 if datos.materiales_disponibles else 1024
    texto   = await _llamar_gemini(prompt, max_tokens=max_tok)

    if not texto:
        return None

    try:
        data     = json.loads(_limpiar_json(texto))
        recursos = _parsear_recursos_sugeridos(
            data.get("recursos_sugeridos", []),
            datos.materiales_disponibles,
        )
        return AnalisisGemini(
            explicacion        = data.get("explicacion", ""),
            recomendaciones    = data.get("recomendaciones", []),
            recursos_sugeridos = recursos,
            alerta_urgente     = data.get("alerta_urgente", False),
            mensaje_alerta     = data.get("mensaje_alerta"),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Error parseando Gemini (intento 1): {e}")

        try:
            explicacion = re.search(
                r'"explicacion"\s*:\s*"(.*?)"(?=\s*,\s*")',
                texto, re.DOTALL
            )
            recomendaciones_raw = re.findall(
                r'"([^"]{20,})"',
                texto[texto.find('"recomendaciones"'):texto.find('"recursos_sugeridos"')]
                if '"recomendaciones"' in texto else ''
            )
            alerta = '"alerta_urgente": true' in texto.lower()

            if explicacion:
                return AnalisisGemini(
                    explicacion        = explicacion.group(1).replace('\\n', ' '),
                    recomendaciones    = recomendaciones_raw[:3] if recomendaciones_raw else [
                        "Revisá las notas del estudiante con el equipo pedagógico."
                    ],
                    recursos_sugeridos = [],
                    alerta_urgente     = alerta,
                    mensaje_alerta     = None,
                )
        except Exception as e2:
            logger.warning(f"Error en segundo intento: {e2}")

        return AnalisisGemini(
            explicacion        = "El análisis automático no pudo procesarse correctamente.",
            recomendaciones    = ["Consultá con el equipo pedagógico para definir acciones de apoyo."],
            recursos_sugeridos = [],
            alerta_urgente     = resultado.nivel_riesgo.value in ["alto", "critico"],
            mensaje_alerta     = None,
        )


async def analizar_escenarios(
    datos: DatosTiempoRealRequest,
    situacion_actual: ResultadoModelo,
    escenarios: list[ResultadoEscenario],
) -> Optional[str]:
    prompt = _prompt_simulacion(datos, situacion_actual, escenarios)
    return await _llamar_gemini(prompt, max_tokens=512)


async def generar_plan_recuperacion(
    datos: DatosTiempoRealRequest,
    resultado: ResultadoModelo,
) -> Optional[dict]:
    if resultado.nivel_riesgo.value == "bajo":
        return None
    prompt = _prompt_plan_recuperacion(datos, resultado)
    texto  = await _llamar_gemini(prompt, max_tokens=1800)
    if not texto:
        return None
    try:
        return json.loads(_limpiar_json(texto))
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Error parseando plan: {e}")
        return None


async def analizar_clase(
    materia: str,
    trimestre: int,
    semana: int,
    total_semanas: int,
    resumen_clase: dict,
) -> Optional[dict]:
    prompt = _prompt_analisis_clase(materia, trimestre, semana, total_semanas, resumen_clase)
    texto  = await _llamar_gemini(prompt, max_tokens=1024)
    if not texto:
        return None
    try:
        return json.loads(_limpiar_json(texto))
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Error parseando análisis clase: {e}")
        return None


async def verificar_disponibilidad() -> bool:
    settings = get_settings()
    if not settings.gemini_api_key:
        return False
    texto = await _llamar_gemini("Responde solo: ok", max_tokens=200)
    return texto is not None and "ok" in texto.lower()


# ─────────────────────────────────────────
# YOUTUBE DATA API v3 — helper interno
# ─────────────────────────────────────────

async def _buscar_video_youtube(query: str, api_key: str) -> Optional[dict]:
    """
    YouTube Data API v3 — devuelve el primer video real para el query.
    Retorna { titulo, url, video_id } o None.

    Costo: 100 unidades por llamada · 10,000 gratis/día → 100 búsquedas/día.
    """
    params = {
        "part":              "snippet",
        "q":                 query,
        "type":              "video",
        "maxResults":        1,
        "relevanceLanguage": "es",
        "safeSearch":        "strict",    # contenido seguro para estudiantes
        "videoEmbeddable":   "true",
        "key":               api_key,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(YOUTUBE_SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        items = data.get("items", [])
        if not items:
            logger.warning(f"[YouTube API] Sin resultados: {query!r}")
            return None

        item     = items[0]
        video_id = item["id"].get("videoId")
        titulo   = item["snippet"].get("title", query)
        if not video_id:
            return None

        return {
            "titulo":   titulo,
            "url":      f"https://www.youtube.com/watch?v={video_id}",
            "video_id": video_id,
        }
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 403:
            logger.error("[YouTube API] 403 — quota excedida o API key inválida")
        else:
            logger.warning(f"[YouTube API] HTTP {status} para: {query!r}")
        return None
    except Exception as e:
        logger.warning(f"[YouTube API] Error: {e}")
        return None


# ─────────────────────────────────────────
# generar_recursos_externos  ← v8.3
#
# Reemplaza la versión v8.2 que pedía URLs directamente a Gemini
# (con el problema de que Gemini inventaba IDs de videos inexistentes).
#
# Nuevo flujo de dos pasos:
#   1. Gemini genera search_queries pedagógicamente precisos.
#   2. YouTube Data API v3 busca con esos queries y retorna IDs REALES.
#
# Requiere: YOUTUBE_API_KEY en .env
#           YouTube Data API v3 activa en https://console.cloud.google.com
# ─────────────────────────────────────────

async def generar_recursos_externos(
    tema_titulo:      str,
    tema_descripcion: Optional[str]       = None,
    palabras_clave:   Optional[list[str]] = None,
    nivel_dificultad: Optional[str]       = None,
    objetivos_unidad: Optional[str]       = None,
    nivel_educativo:  Optional[str]       = None,
    _llamar_gemini_fn=None,
) -> list[dict]:
    """
    Devuelve 2-3 videos REALES de YouTube para reforzar el tema.

    Paso 1 — Gemini: genera search_queries pedagógicamente precisos.
    Paso 2 — YouTube API: busca con esos queries y retorna IDs reales.
    Paso 3 — URLs verificadas: youtube.com/watch?v=ID_REAL (siempre existe).

    El parámetro _llamar_gemini_fn permite inyectar un mock en tests unitarios.
    Si YOUTUBE_API_KEY no está configurada → retorna [] con warning.
    Si un query no tiene resultados → ese recurso se omite.
    """
    _fn = _llamar_gemini_fn or _llamar_gemini

    settings    = get_settings()
    youtube_key = getattr(settings, "youtube_api_key", None) or ""

    if not youtube_key:
        logger.warning(
            "[generar_recursos_externos] YOUTUBE_API_KEY no configurada. "
            "Agregá YOUTUBE_API_KEY=... al .env y activá YouTube Data API v3 "
            "en https://console.cloud.google.com"
        )
        return []

    # ── Paso 1: Gemini genera queries específicos ─────────────────────────────
    partes = [f'Tema: "{tema_titulo}"']
    if tema_descripcion:
        partes.append(f"Descripción: {tema_descripcion}")
    if palabras_clave:
        partes.append(f"Palabras clave: {', '.join(palabras_clave)}")
    if nivel_dificultad:
        partes.append(f"Nivel de dificultad: {nivel_dificultad}")
    if objetivos_unidad:
        partes.append(f"Objetivos: {objetivos_unidad}")

    nivel = nivel_educativo or "secundaria"

    prompt = f"""Eres un asistente educativo. Un estudiante de nivel "{nivel}" necesita \
videos de YouTube para reforzar este tema:

{chr(10).join(partes)}

Genera 2 o 3 queries de búsqueda para YouTube que encuentren videos educativos \
específicos y útiles. Los videos deben ser en español y apropiados para el nivel.

Devuelve SOLO un JSON (sin markdown):
[
  {{
    "titulo_sugerido": "Descripción de lo que debería enseñar el video",
    "search_query": "query específico en español para buscar en YouTube"
  }}
]

Reglas para search_query:
- Específico al tema, no genérico.
- Incluir palabras clave del tema + nivel educativo + "educativo" o "tutorial".
- En español.
- Solo JSON, sin texto adicional."""

    texto = await _fn(prompt, max_tokens=400)
    if not texto:
        logger.warning(f"[generar_recursos_externos] Gemini no respondió | tema: {tema_titulo}")
        return []

    # ── Parsear queries ───────────────────────────────────────────────────────
    try:
        clean = texto.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\[.*\]", clean, re.DOTALL)
        clean = match.group(0).strip() if match else clean
        queries_raw = json.loads(clean)

        if not isinstance(queries_raw, list):
            raise ValueError("No es array")

        queries = [
            {
                "titulo_sugerido": q.get("titulo_sugerido", tema_titulo),
                "search_query":    q["search_query"].strip(),
            }
            for q in queries_raw
            if isinstance(q, dict) and q.get("search_query", "").strip()
        ][:3]

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"[generar_recursos_externos] Error parseando Gemini: {e}")
        # Fallback: query directo con el título del tema
        queries = [{
            "titulo_sugerido": tema_titulo,
            "search_query":    f"{tema_titulo} educativo {nivel} tutorial",
        }]

    if not queries:
        return []

    logger.info(
        f"[generar_recursos_externos] {len(queries)} query(s) de Gemini | tema: {tema_titulo}"
    )

    # ── Paso 2: Buscar videos reales en YouTube ───────────────────────────────
    resultado = []
    for q in queries:
        logger.info(f"[YouTube API] Buscando: {q['search_query']!r}")
        video = await _buscar_video_youtube(q["search_query"], youtube_key)

        if video:
            resultado.append({
                "titulo":         video["titulo"],   # título real del video
                "url":            video["url"],       # URL verificada con ID real
                "origen_externo": "youtube",
            })
            logger.info(f"[YouTube API] ✓ {video['titulo']} → {video['url']}")
        else:
            logger.warning(f"[YouTube API] Sin resultado para: {q['search_query']!r}")

    logger.info(
        f"[generar_recursos_externos] {len(resultado)} video(s) real(es) | tema: {tema_titulo}"
    )
    return resultado