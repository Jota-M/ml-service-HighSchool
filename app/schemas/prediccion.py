"""
schemas/prediccion.py — Modelos Pydantic request/response
Sistema de predicción en tiempo real por semana
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class NivelRiesgo(str, Enum):
    BAJO    = "bajo"
    MEDIO   = "medio"
    ALTO    = "alto"
    CRITICO = "critico"

class Clasificacion(str, Enum):
    ED = "ED"
    DA = "DA"
    DO = "DO"
    DP = "DP"

class NivelConfianza(str, Enum):
    MUY_BAJA = "muy_baja"
    BAJA     = "baja"
    MEDIA    = "media"
    ALTA     = "alta"
    MUY_ALTA = "muy_alta"


# ─────────────────────────────────────────
# REQUEST — DATOS EN TIEMPO REAL
# ─────────────────────────────────────────

class DatosTiempoRealRequest(BaseModel):
    """
    Snapshot semanal del estudiante en una materia.
    Se envía cada vez que se registra una práctica, asistencia o conducta.
    """
    # Identificación
    estudiante_id:  int = Field(..., description="ID del estudiante")
    materia:        str = Field(..., description="Nombre de la materia")
    codigo_materia: str = Field(..., description="Código (MAT, FIS, etc.)")
    trimestre:      int = Field(..., ge=1, le=3, description="Trimestre actual (1, 2 o 3)")

    # Contexto del estudiante (no cambia en el trimestre)
    tipo_colegio:          str  = Field(..., description="fiscal / convenio / privado")
    zona:                  str  = Field(..., description="urbana / rural")
    nivel_socioeconomico:  str  = Field(..., description="bajo / medio / alto")
    trabaja:               bool = Field(..., description="Si el estudiante trabaja")
    dificultad_materia:    str  = Field(..., description="baja / media / alta / muy_alta")

    # Tiempo actual en el trimestre
    semana: int = Field(..., ge=1, le=13, description="Semana actual del trimestre (1-13)")

    # Asistencia acumulada hasta esta semana
    asistencia_acumulada_pct:   float = Field(..., ge=0, le=100, description="% asistencia acumulada")
    racha_inasistencias:        int   = Field(default=0, ge=0, description="Días consecutivos sin asistir")
    max_racha_inasistencias:    int   = Field(default=0, ge=0, description="Máxima racha de inasistencias")

    # Evaluaciones registradas hasta esta semana
    notas_practicas: list[float] = Field(default=[], description="Notas de prácticas/tareas registradas")
    notas_examenes:  list[float] = Field(default=[], description="Notas de exámenes registrados")

    # Conducta acumulada
    conductas_negativas_acumuladas: int = Field(default=0, ge=0, description="Total conductas negativas acumuladas")
    conductas_negativas_semana:     int = Field(default=0, ge=0, description="Conductas negativas esta semana")

    @field_validator("notas_practicas", "notas_examenes")
    @classmethod
    def validar_notas(cls, v):
        return [round(max(1, min(100, n)), 1) for n in v]

    class Config:
        json_schema_extra = {
            "example": {
                "estudiante_id": 42,
                "materia": "Matematica",
                "codigo_materia": "MAT",
                "trimestre": 2,
                "tipo_colegio": "fiscal",
                "zona": "urbana",
                "nivel_socioeconomico": "medio",
                "trabaja": False,
                "dificultad_materia": "muy_alta",
                "semana": 6,
                "asistencia_acumulada_pct": 73.3,
                "racha_inasistencias": 0,
                "max_racha_inasistencias": 2,
                "notas_practicas": [72.0, 65.0, 58.0],
                "notas_examenes": [54.0],
                "conductas_negativas_acumuladas": 2,
                "conductas_negativas_semana": 0,
            }
        }


class EscenarioSimulacion(BaseModel):
    """Modificaciones hipotéticas para simular un escenario."""
    descripcion:               str            = Field(..., description="Descripción del escenario")
    asistencia_proyectada:     Optional[float] = Field(None, ge=0, le=100, description="Nueva asistencia proyectada")
    nota_proxima_practica:     Optional[float] = Field(None, ge=1, le=100, description="Nota de próxima práctica simulada")
    reducir_conductas_negativas: bool          = Field(default=False, description="Simular mejora en conducta")
    trabaja:                   Optional[bool]  = Field(None, description="Cambiar si trabaja")

    class Config:
        json_schema_extra = {
            "example": {
                "descripcion": "¿Qué pasa si mejora asistencia al 90%?",
                "asistencia_proyectada": 90.0,
                "nota_proxima_practica": None,
                "reducir_conductas_negativas": False,
                "trabaja": None,
            }
        }


class SimulacionRequest(BaseModel):
    datos_base: DatosTiempoRealRequest       = Field(..., description="Situación actual")
    escenarios: list[EscenarioSimulacion]    = Field(..., min_length=1, max_length=5)


# ─────────────────────────────────────────
# RESPONSE
# ─────────────────────────────────────────

class ConfianzaPrediccion(BaseModel):
    nivel:                 NivelConfianza = Field(..., description="Nivel de confianza de la predicción")
    porcentaje_trimestre:  float          = Field(..., description="% del trimestre transcurrido")
    mensaje:               str            = Field(..., description="Explicación del nivel de confianza")


class ResultadoModelo(BaseModel):
    probabilidad_reprobar:  float        = Field(..., description="Probabilidad de reprobar (0-1)")
    nivel_riesgo:           NivelRiesgo  = Field(..., description="bajo/medio/alto/crítico")
    nota_estimada_final:    float        = Field(..., description="Nota final estimada (1-100)")
    clasificacion_estimada: Clasificacion = Field(..., description="ED/DA/DO/DP")
    factores_riesgo:        list[str]    = Field(..., description="Factores que aumentan el riesgo")
    factores_positivos:     list[str]    = Field(..., description="Factores protectores")
    confianza:              ConfianzaPrediccion = Field(..., description="Confiabilidad de la predicción")


class AnalisisGemini(BaseModel):
    explicacion:     str         = Field(..., description="Análisis en lenguaje natural")
    recomendaciones: list[str]   = Field(..., description="Acciones concretas para el docente")
    alerta_urgente:  bool        = Field(..., description="Si requiere intervención inmediata")
    mensaje_alerta:  Optional[str] = Field(None, description="Mensaje de alerta si es urgente")


class PrediccionResponse(BaseModel):
    estudiante_id:     int  = Field(..., description="ID del estudiante")
    materia:           str  = Field(..., description="Materia evaluada")
    codigo_materia:    str
    trimestre:         int
    semana_actual:     int  = Field(..., description="Semana del trimestre en que se hizo la predicción")
    modelo:            ResultadoModelo
    analisis:          Optional[AnalisisGemini] = None
    modelo_usado:      str
    gemini_disponible: bool


class ResultadoEscenario(BaseModel):
    descripcion:           str          = Field(..., description="Descripción del escenario")
    probabilidad_reprobar: float
    nivel_riesgo:          NivelRiesgo
    nota_estimada_final:   float
    cambio_probabilidad:   float        = Field(..., description="Delta vs situación actual (negativo=mejora)")
    cambio_nota:           float        = Field(..., description="Delta en nota estimada")
    conclusion:            str


class SimulacionResponse(BaseModel):
    estudiante_id:        int
    materia:              str
    semana_actual:        int
    situacion_actual:     ResultadoModelo
    escenarios:           list[ResultadoEscenario]
    recomendacion_gemini: Optional[str] = None


class HealthResponse(BaseModel):
    status:            str
    modelos_cargados:  bool
    gemini_disponible: bool
    version_modelo:    str
    tipo_modelo:       str = Field(default="tiempo_real_semanal")


class ReentrenarResponse(BaseModel):
    exitoso:           bool
    registros_usados:  int
    duracion_segundos: float
    metricas:          dict
    mensaje:           str