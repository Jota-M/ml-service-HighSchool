"""
schemas/prediccion.py — Modelos Pydantic v8.5
"""

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────

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
# CONFIG PERÍODO
# ─────────────────────────────────────────

class ConfigPeriodo(BaseModel):
    total_semanas: int = Field(
        default=13, ge=8, le=20,
        description="Total de semanas del período (13=trimestre, 10=bimestre)"
    )
    ponderaciones: dict[str, float] = Field(
        default={"SER": 10.0, "SAB": 45.0, "HAC": 40.0, "AUT": 5.0},
        description="Pesos por dimensión desde dimension_evaluacion (deben sumar 100)"
    )

    @field_validator("ponderaciones")
    @classmethod
    def validar_ponderaciones(cls, v: dict) -> dict:
        if not v:
            raise ValueError("Las ponderaciones no pueden estar vacías")
        total = sum(v.values())
        if abs(total - 100.0) > 0.5:
            raise ValueError(
                f"Las ponderaciones deben sumar 100, suman {total:.1f}. "
                f"Verificá dimension_evaluacion en la BD."
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "total_semanas": 13,
                "ponderaciones": {"SER": 10.0, "SAB": 45.0, "HAC": 40.0, "AUT": 5.0}
            }
        }
# ─────────────────────────────────────────
# MATERIAL DISPONIBLE
# ─────────────────────────────────────────

class MaterialDisponible(BaseModel):
    id:           int           = Field(..., description="ID del material en la BD")
    titulo:       str           = Field(..., description="Título del material")
    tipo:         str           = Field(..., description="Tipo: PDF, VIDEO, DOC, etc.")
    tipo_codigo:  str           = Field(..., description="Código del tipo: PDF, VIDEO, etc.")
    tema_titulo:  Optional[str] = Field(None, description="Tema al que pertenece")
    tema_id:      Optional[int] = Field(None, description="ID del tema en la BD")
    descripcion:  Optional[str] = Field(None, description="Descripción breve del material")
    es_destacado: bool          = Field(False, description="Si es material destacado")
    url:          Optional[str] = Field(None, description="URL de acceso al material")


# ─────────────────────────────────────────
# REQUEST — DATOS EN TIEMPO REAL v8.1
# ─────────────────────────────────────────

class DatosTiempoRealRequest(BaseModel):
    """
    Snapshot semanal del estudiante en una materia — v8.1.

    Node.js es responsable de calcular y mandar:
      - Los campos de historial (consultando trimestres anteriores en la BD)
      - Los campos de observaciones (sumando desde observacion_pedagogica)
      - Los campos de correlación (mirando otras materias del mismo estudiante)
      - nivel_educativo y horas_grado (desde el perfil del estudiante/grado)
      - regimen_pond (desde el año académico actual)

    Todos los campos nuevos tienen defaults → clientes legacy siguen funcionando.
    El modelo fue entrenado con -1 para "sin información histórica", así que
    mandar -1 es semánticamente correcto para el primer trimestre.
    """

    # ── Identificación ────────────────────────────────────────────────────────
    estudiante_id:  int = Field(..., description="ID del estudiante")
    materia:        str = Field(..., description="Nombre de la materia")
    codigo_materia: str = Field(..., description="Código (MAT, FIS, etc.)")
    trimestre:      int = Field(..., ge=1, le=3, description="Número del período (1, 2 o 3)")

    # ── Config del período ────────────────────────────────────────────────────
    config_periodo: ConfigPeriodo = Field(default_factory=ConfigPeriodo)

    # ── Tiempo actual en el período ───────────────────────────────────────────
    semana: int = Field(..., ge=1, description="Semana actual del período")

    # ── Asistencia ────────────────────────────────────────────────────────────
    asistencia_acumulada_pct: float = Field(..., ge=0, le=100)
    racha_inasistencias:      int   = Field(default=0, ge=0)
    max_racha_inasistencias:  int   = Field(default=0, ge=0)

    # ── Notas por dimensión (campos principales) ──────────────────────────────
    notas_sab: list[float] = Field(
        default=[],
        description="Notas de Saber — exámenes y pruebas (0-100 normalizadas)"
    )
    notas_hac: list[float] = Field(
        default=[],
        description="Notas de Hacer — prácticas, tareas, proyectos (0-100 normalizadas)"
    )
    nota_complementaria_pct: float = Field(
        default=0.0, ge=0, le=100,
        description="Aporte ponderado de SER+AUT (calculado por Node.js)"
    )
    peso_complementario: float = Field(
        default=0.15, ge=0, le=1,
        description="Peso total de dimensiones complementarias (ej: 0.10+0.05=0.15)"
    )

    # ── Notas legacy (compatibilidad hacia atrás) ─────────────────────────────
    notas_practicas: list[float] = Field(default=[], description="[Legacy] Usar notas_hac")
    notas_examenes:  list[float] = Field(default=[], description="[Legacy] Usar notas_sab")

    # ── Historial intertrimestral (v8.1) ──────────────────────────────────────
    nota_trim_ant: float = Field(
        default=-1.0,
        description="Nota final del trimestre anterior. -1 si es el primer trimestre del año."
    )
    asist_trim_ant: float = Field(
        default=-1.0,
        description="Asistencia promedio del trimestre anterior. -1 si no hay historial."
    )
    reprobo_trim_ant: int = Field(
        default=0, ge=0, le=1,
        description="1 si reprobó el trimestre anterior, 0 si no o si no hay historial."
    )
    racha_trims_riesgo: int = Field(
        default=0, ge=0,
        description="Cuántos trimestres consecutivos lleva en riesgo en esta materia."
    )
    mejor_nota_historica: float = Field(
        default=-1.0,
        description="Mejor nota que sacó en esta materia en cualquier trimestre previo. -1 si no hay."
    )
    tend_intertrimestral: int = Field(
        default=0,
        description="Tendencia entre el penúltimo y último trimestre: -1 bajando, 0 estable, 1 subiendo."
    )
    reprobo_misma_mat_ant: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Diferencia normalizada [-1,1] entre nota de esta materia "
                    "y promedio de otras el trimestre anterior. 0.0 si no hay historial."
    )
    # ── v8.6: SAB y HAC del trimestre anterior por dimensión ──────────────────
    sab_trim_ant: float = Field(
        default=-1.0,
        description="Promedio SAB del trimestre anterior. -1 si no hay historial."
    )
    hac_trim_ant: float = Field(
        default=-1.0,
        description="Promedio HAC del trimestre anterior. -1 si no hay historial."
    )
    # ── Observaciones pedagógicas (v8.1) ──────────────────────────────────────
    n_obs_conducta: int = Field(
        default=0, ge=0,
        description="Observaciones de conducta acumuladas este trimestre."
    )
    n_obs_socioem: int = Field(
        default=0, ge=0,
        description="Observaciones socioemocionales acumuladas este trimestre."
    )
    n_obs_urgentes: int = Field(
        default=0, ge=0,
        description="Observaciones urgentes acumuladas este trimestre."
    )
    n_logros: int = Field(
        default=0, ge=0,
        description="Logros destacados registrados por el docente este trimestre."
    )
    ratio_obs_negativas: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="(conducta+socioem+urgentes) / total_obs. 0 si no hay observaciones."
    )

    # ── Correlación entre materias (v8.1) ─────────────────────────────────────
    n_materias_riesgo_sim: int = Field(
        default=0, ge=0,
        description="Cuántas otras materias del estudiante están en riesgo este trimestre."
    )
    reprobo_mat_correlac: int = Field(
        default=0, ge=0, le=1,
        description="1 si reprueba una materia del mismo grupo curricular (ej: FIS si reprueba MAT)."
    )

    # ── Nivel educativo y carga horaria (v8.1) ────────────────────────────────
    nivel_educativo: int = Field(
        default=1, ge=0, le=1,
        description="0=primaria, 1=secundaria."
    )
    horas_grado: int = Field(
        default=168, ge=100, le=250,
        description="Carga horaria anual del grado según malla curricular (136/168/176/192)."
    )

    # ── Régimen de ponderación (v8.1) ─────────────────────────────────────────
    regimen_pond: int = Field(
        default=1, ge=0, le=3,
        description="Código del régimen ministerial: 0=2021, 1=2022-24, 2=2025, 3=2026."
    )

    # ── Materiales del repositorio ────────────────────────────────────────────
    materiales_disponibles: list[MaterialDisponible] = Field(default=[])

    # ── Validadores ───────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def validar_semana_vs_periodo(self) -> "DatosTiempoRealRequest":
        if self.semana > self.config_periodo.total_semanas:
            raise ValueError(
                f"semana={self.semana} excede total_semanas="
                f"{self.config_periodo.total_semanas} del período"
            )
        return self

    @field_validator("notas_sab", "notas_hac", "notas_practicas", "notas_examenes")
    @classmethod
    def validar_notas(cls, v: list) -> list:
        return [round(max(0.0, min(100.0, n)), 1) for n in v]

    # ── Propiedades de conveniencia ───────────────────────────────────────────

    @property
    def notas_saber(self) -> list[float]:
        return self.notas_sab if self.notas_sab else self.notas_examenes

    @property
    def notas_hacer(self) -> list[float]:
        return self.notas_hac if self.notas_hac else self.notas_practicas

    @property
    def todas_las_notas(self) -> list[float]:
        return self.notas_saber + self.notas_hacer

    class Config:
        json_schema_extra = {
            "example": {
                "estudiante_id": 42,
                "materia": "Matematica",
                "codigo_materia": "MAT",
                "trimestre": 2,
                "config_periodo": {
                    "total_semanas": 13,
                    "ponderaciones": {"SER": 10.0, "SAB": 45.0, "HAC": 40.0, "AUT": 5.0}
                },
                "semana": 6,
                "asistencia_acumulada_pct": 73.3,
                "racha_inasistencias": 0,
                "max_racha_inasistencias": 2,
                "notas_sab": [54.0, 68.0],
                "notas_hac": [72.0, 65.0, 58.0],
                "nota_complementaria_pct": 82.5,
                "peso_complementario": 0.15,
                "nota_trim_ant": 47.0,
                "asist_trim_ant": 71.0,
                "reprobo_trim_ant": 1,
                "racha_trims_riesgo": 1,
                "mejor_nota_historica": 54.0,
                "tend_intertrimestral": 0,
                "reprobo_misma_mat_ant": 1,
                "n_obs_conducta": 2,
                "n_obs_socioem": 0,
                "n_obs_urgentes": 0,
                "n_logros": 1,
                "ratio_obs_negativas": 0.667,
                "n_materias_riesgo_sim": 2,
                "reprobo_mat_correlac": 1,
                "nivel_educativo": 1,
                "horas_grado": 176,
                "regimen_pond": 2,
                "materiales_disponibles": []
            }
        }


# ─────────────────────────────────────────
# ESCENARIOS Y SIMULACIÓN
# ─────────────────────────────────────────

class EscenarioSimulacion(BaseModel):
    descripcion:           str            = Field(..., description="Descripción del escenario")
    asistencia_proyectada: Optional[float] = Field(None, ge=0, le=100)
    nota_proxima_practica: Optional[float] = Field(None, ge=0, le=100)
    nota_proximo_examen:   Optional[float] = Field(None, ge=0, le=100)
    semanas_adicionales:   Optional[int]   = Field(None, ge=1, le=5)


class SimulacionRequest(BaseModel):
    datos_base: DatosTiempoRealRequest
    escenarios: list[EscenarioSimulacion] = Field(..., min_length=1, max_length=5)


# ─────────────────────────────────────────
# RESPONSES
# ─────────────────────────────────────────

class ConfianzaPrediccion(BaseModel):
    nivel:              NivelConfianza
    porcentaje_periodo: float
    mensaje:            str


class ResultadoModelo(BaseModel):
    probabilidad_reprobar:  float
    nivel_riesgo:           NivelRiesgo
    nota_estimada_final:    float
    clasificacion_estimada: Clasificacion
    factores_riesgo:        list[str]
    factores_positivos:     list[str]
    confianza:              ConfianzaPrediccion


class RecursoRecomendado(BaseModel):
    material_id:  Optional[int] = None
    titulo:       str
    tipo:         str
    tema_titulo:  Optional[str] = None
    url:          Optional[str] = None
    search_query: Optional[str] = None
    razon:        str


class AnalisisGemini(BaseModel):
    explicacion:        str
    recomendaciones:    list[str]
    recursos_sugeridos: list[RecursoRecomendado] = Field(default=[])
    alerta_urgente:     bool
    mensaje_alerta:     Optional[str] = None


class PrediccionResponse(BaseModel):
    estudiante_id:     int
    materia:           str
    codigo_materia:    str
    trimestre:         int
    semana_actual:     int
    total_semanas:     int
    modelo:            ResultadoModelo
    analisis:          Optional[AnalisisGemini] = None
    modelo_usado:      str
    gemini_disponible: bool


class ResultadoEscenario(BaseModel):
    descripcion:           str
    probabilidad_reprobar: float
    nivel_riesgo:          NivelRiesgo
    nota_estimada_final:   float
    cambio_probabilidad:   float
    cambio_nota:           float
    conclusion:            str


class SimulacionResponse(BaseModel):
    estudiante_id:        int
    materia:              str
    semana_actual:        int
    total_semanas:        int
    situacion_actual:     ResultadoModelo
    escenarios:           list[ResultadoEscenario]
    recomendacion_gemini: Optional[str] = None


class HealthResponse(BaseModel):
    status:            str
    modelos_cargados:  bool
    gemini_disponible: bool
    version_modelo:    str
    tipo_modelo:       str = Field(default="tiempo_real_semanal_v8")
    n_features:        int = Field(default=29)


class ReentrenarResponse(BaseModel):
    exitoso:           bool
    registros_usados:  int
    duracion_segundos: float
    metricas:          dict
    mensaje:           str


# ─────────────────────────────────────────
# RECURSOS EXTERNOS  (v8.2)
# ─────────────────────────────────────────

class RecursoExternoItem(BaseModel):
    titulo:         str = Field(..., description="Título descriptivo del recurso")
    url:            str = Field(..., description="URL del recurso externo")
    origen_externo: str = Field(default="web", description="youtube | khan_academy | web")


class RecursosExternosRequest(BaseModel):
    """
    Payload que manda Node.js a POST /materiales/recursos-externos
    cuando no hay materiales internos (fecha_publicacion IS NULL)
    para el tema donde el estudiante tuvo nota baja.
    """
    tema_titulo:      str                 = Field(..., description="Título del tema con dificultad")
    tema_descripcion: Optional[str]       = Field(None, description="Descripción del tema")
    palabras_clave:   Optional[list[str]] = Field(None, description="Palabras clave del tema")
    nivel_dificultad: Optional[str]       = Field(None, description="Nivel de dificultad del tema")
    objetivos_unidad: Optional[str]       = Field(None, description="Objetivos de la unidad temática")
    nivel_educativo:  Optional[str]       = Field(
        None,
        description="Ej: 'Secundaria — 3° Grado'. Construido en Node.js desde nivel_academico + grado."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "tema_titulo":      "Ecuaciones de segundo grado",
                "tema_descripcion": "Resolución de ecuaciones cuadráticas por fórmula general y factorización",
                "palabras_clave":   ["discriminante", "fórmula cuadrática", "raíces"],
                "nivel_dificultad": "medio",
                "objetivos_unidad": "Resolver ecuaciones de segundo grado y aplicarlas en problemas contextualizados",
                "nivel_educativo":  "Secundaria — 3° Grado",
            }
        }


class RecursosExternosResponse(BaseModel):
    recursos:          list[RecursoExternoItem] = Field(
        default=[],
        description="Lista de recursos externos sugeridos por Gemini (máx. 3)"
    )
    gemini_disponible: bool = Field(
        description="True si Gemini respondió y devolvió al menos un recurso válido"
    )
    total:             int  = Field(
        description="Cantidad de recursos devueltos"
    )


# ─────────────────────────────────────────
# SIMULACIÓN ÓPTIMA  (v8.3)
# ─────────────────────────────────────────

class RestriccionesOptimo(BaseModel):
    """
    El cliente puede bloquear palancas que el estudiante no puede mejorar.
    Por ejemplo: si ya cerró asistencia o el examen ya fue tomado.
    """
    bloquear_practicas:  bool = False
    bloquear_examenes:   bool = False
    bloquear_asistencia: bool = False


class SimulacionOptimoRequest(BaseModel):
    datos_base:    DatosTiempoRealRequest
    objetivo_nota: float = Field(
        default=51.0, ge=1.0, le=100.0,
        description="Nota final mínima que se quiere alcanzar (default: 51 = aprobar)"
    )
    restricciones: RestriccionesOptimo = Field(default_factory=RestriccionesOptimo)


class AccionRequeridaResponse(BaseModel):
    componente:      str    # "practicas" | "examenes" | "asistencia"
    label:           str
    valor_actual:    float
    valor_necesario: float
    delta:           float
    impacto_nota:    float
    dificultad:      str    # "baja" | "media" | "alta"


class SimulacionOptimoResponse(BaseModel):
    objetivo_nota:        float
    nota_actual:          float
    nota_proyectada:      float
    alcanzable:           bool
    acciones:             list[AccionRequeridaResponse]
    nota_maxima_posible:  float
    mensaje:              str
    modo:                 str


# ─────────────────────────────────────────
# SIMULACIÓN ÓPTIMA V2  ← v8.6
# ─────────────────────────────────────────
 
class EvaluacionPendienteResponse(BaseModel):
    numero: int = Field(..., description="Número de la evaluación (ej: 4 = es la 4ta práctica)")
    tipo: str = Field(..., description="'practica' | 'examen'")
    nota_objetivo: float
    es_alcanzable: bool
 
 
class EscenarioDetalladoResponse(BaseModel):
    id: str
    titulo: str
    descripcion: str
    evaluaciones: list[EvaluacionPendienteResponse]
    nota_proyectada: float
    alcanzable: bool
    porcentaje_exito: float
    mensaje: str
 
 
class SimulacionOptimoV2Response(BaseModel):
    objetivo_nota: float
    nota_actual: float
    nota_maxima_posible: float
    semanas_restantes: int
    practicas_restantes_est: int
    examenes_restantes_est: int
    techo_practicas: float
    techo_examenes: float
    escenarios: list[EscenarioDetalladoResponse]
    ya_alcanza: bool
    imposible: bool
    mensaje_general: str
 
 
class SimulacionOptimoV2Request(BaseModel):
    datos_base: DatosTiempoRealRequest
    objetivo_nota: float = Field(
        default=51.0, ge=1.0, le=100.0,
        description="Nota final mínima que se quiere alcanzar (default: 51 = aprobar)"
    )
    restricciones: RestriccionesOptimo = Field(default_factory=RestriccionesOptimo)
    # Valores opcionales del docente — si vienen se usan directo, si no se estiman
    practicas_restantes: Optional[int] = Field(
        None, ge=0, le=8,
        description="Cuántas prácticas quedan. Si es None el backend estima por ritmo."
    )
    examenes_restantes: Optional[int] = Field(
        None, ge=0, le=2,
        description="Cuántos exámenes quedan. Si es None el backend estima por ritmo."
    )
 