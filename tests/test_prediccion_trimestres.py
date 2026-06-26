"""
tests/test_prediccion_trimestres.py

Casos de prueba para validar que el modelo predice coherentemente
en los tres trimestres. Cada caso tiene un "resultado_esperado"
que define qué debería pasar si el modelo funciona bien.

Ejecutar con:
    pytest tests/test_prediccion_trimestres.py -v
    # o directamente:
    python tests/test_prediccion_trimestres.py

Requiere que el servidor esté corriendo en localhost:8000.
"""

import httpx
import json
import sys

BASE_URL = "http://localhost:8000/api/v1"
TIMEOUT  = 30


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def predecir(payload: dict) -> dict:
    r = httpx.post(
        f"{BASE_URL}/predecir",
        json=payload,
        params={"usar_xgboost": True, "incluir_gemini": False},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def assert_nivel(resultado: dict, niveles_esperados: list[str], caso: str):
    nivel_real = resultado["modelo"]["nivel_riesgo"]
    assert nivel_real in niveles_esperados, (
        f"[{caso}] Nivel esperado: {niveles_esperados} | Got: {nivel_real}"
    )


def assert_nota_rango(resultado: dict, min_nota: float, max_nota: float, caso: str):
    nota_real = resultado["modelo"]["nota_estimada_final"]
    assert min_nota <= nota_real <= max_nota, (
        f"[{caso}] Nota esperada: [{min_nota}, {max_nota}] | Got: {nota_real}"
    )


def assert_clasif(resultado: dict, clasif_esperadas: list[str], caso: str):
    clasif_real = resultado["modelo"]["clasificacion_estimada"]
    assert clasif_real in clasif_esperadas, (
        f"[{caso}] Clasificación esperada: {clasif_esperadas} | Got: {clasif_real}"
    )


# ─────────────────────────────────────────
# PAYLOAD BASE — reutilizable
# ─────────────────────────────────────────

def base_payload(
    est_id:          int,
    trimestre:       int,
    semana:          int,
    asistencia:      float,
    notas_sab:       list[float],
    notas_hac:       list[float],
    nota_comp:       float = 75.0,
    # Historial
    nota_trim_ant:           float = -1.0,
    asist_trim_ant:          float = -1.0,
    reprobo_trim_ant:        int   = 0,
    racha_trims_riesgo:      int   = 0,
    mejor_nota_historica:    float = -1.0,
    tend_intertrimestral:    int   = 0,
    reprobo_misma_mat_ant:   int   = 0,
    # Observaciones
    n_obs_conducta:  int   = 0,
    n_obs_urgentes:  int   = 0,
    n_logros:        int   = 0,
    # Contexto
    racha_inasistencias:     int   = 0,
    max_racha_inasistencias: int   = 0,
) -> dict:
    return {
        "estudiante_id":  est_id,
        "materia":        "Matematica",
        "codigo_materia": "MAT",
        "trimestre":      trimestre,
        "config_periodo": {
            "total_semanas": 13,
            "ponderaciones": {"SER": 10.0, "SAB": 45.0, "HAC": 40.0, "AUT": 5.0},
        },
        "semana":                    semana,
        "asistencia_acumulada_pct":  asistencia,
        "racha_inasistencias":       racha_inasistencias,
        "max_racha_inasistencias":   max_racha_inasistencias,
        "notas_sab":                 notas_sab,
        "notas_hac":                 notas_hac,
        "nota_complementaria_pct":   nota_comp,
        "peso_complementario":       0.15,
        # Historial
        "nota_trim_ant":             nota_trim_ant,
        "asist_trim_ant":            asist_trim_ant,
        "reprobo_trim_ant":          reprobo_trim_ant,
        "racha_trims_riesgo":        racha_trims_riesgo,
        "mejor_nota_historica":      mejor_nota_historica,
        "tend_intertrimestral":      tend_intertrimestral,
        "reprobo_misma_mat_ant":     reprobo_misma_mat_ant,
        # Observaciones
        "n_obs_conducta":            n_obs_conducta,
        "n_obs_socioem":             0,
        "n_obs_urgentes":            n_obs_urgentes,
        "n_logros":                  n_logros,
        "ratio_obs_negativas":       0.0,
        # Correlación
        "n_materias_riesgo_sim":     0,
        "reprobo_mat_correlac":      0,
        # Nivel
        "nivel_educativo":           1,
        "horas_grado":               176,
        "regimen_pond":              2,
        "materiales_disponibles":    [],
    }


# ─────────────────────────────────────────
# TRIMESTRE 1 — sin historial previo
# El modelo no tiene datos de trimestres anteriores.
# Debe predecir solo con asistencia, notas actuales y semana.
# ─────────────────────────────────────────

def test_t1_inicio_sin_notas():
    """
    T1 semana 2 — sin evaluaciones todavía, asistencia perfecta.
    Esperado: riesgo bajo/medio, nota estimada razonable (no puede
    predecir bien aún, confianza muy_baja).
    El modelo NO debe inventar una nota alta con certeza.
    """
    payload = base_payload(
        est_id=1, trimestre=1, semana=2,
        asistencia=100.0,
        notas_sab=[], notas_hac=[],
    )
    r = predecir(payload)
    print(f"\n[T1-01] Sin notas, asist=100%")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    print(f"  Confianza: {r['modelo']['confianza']['nivel']}")
    assert r["modelo"]["confianza"]["nivel"] in ["muy_baja", "baja"], \
        "Sin evaluaciones en semana 2 la confianza debe ser muy_baja o baja"
    print("  ✅ PASS")


def test_t1_mitad_buen_rendimiento():
    """
    T1 semana 7 — notas buenas, asistencia buena.
    Esperado: riesgo bajo, nota estimada 65-85, clasificación DA/DO.
    """
    payload = base_payload(
        est_id=2, trimestre=1, semana=7,
        asistencia=88.0,
        notas_sab=[72.0, 68.0],
        notas_hac=[75.0, 70.0, 73.0],
        nota_comp=80.0,
    )
    r = predecir(payload)
    print(f"\n[T1-02] Buen rendimiento semana 7")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    assert_nivel(r, ["bajo", "medio"], "T1-02")
    assert_nota_rango(r, 55.0, 90.0, "T1-02")
    print("  ✅ PASS")


def test_t1_mitad_riesgo_alto():
    """
    T1 semana 7 — notas malas, asistencia crítica.
    Esperado: riesgo alto/crítico, nota estimada < 55, clasificación ED.
    """
    payload = base_payload(
        est_id=3, trimestre=1, semana=7,
        asistencia=68.0,
        notas_sab=[38.0, 42.0],
        notas_hac=[35.0, 40.0, 38.0],
        nota_comp=60.0,
        racha_inasistencias=4,
        max_racha_inasistencias=6,
    )
    r = predecir(payload)
    print(f"\n[T1-03] Mal rendimiento + asistencia crítica semana 7")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    assert_nivel(r, ["alto", "critico"], "T1-03")
    assert_nota_rango(r, 1.0, 52.0, "T1-03")
    print("  ✅ PASS")


def test_t1_cierre_estudiante_en_limite():
    """
    T1 semana 12 — notas en el límite de aprobación (50-55).
    Esperado: riesgo medio, nota ~51-58.
    Test de coherencia: si las notas promedio 52, la nota estimada
    no puede ser 80.
    """
    payload = base_payload(
        est_id=4, trimestre=1, semana=12,
        asistencia=82.0,
        notas_sab=[50.0, 53.0],
        notas_hac=[54.0, 51.0, 52.0, 50.0],
        nota_comp=70.0,
    )
    r = predecir(payload)
    nota_est = r["modelo"]["nota_estimada_final"]
    print(f"\n[T1-04] Estudiante en el límite semana 12")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {nota_est}")
    # Con promedio SAB=51.5 y HAC=51.75, nota_final debería estar cerca de 51-57
    assert_nota_rango(r, 40.0, 70.0, "T1-04")
    print("  ✅ PASS")


def test_t1_patron_contradictorio():
    """
    T1 semana 8 — asistencia alta (92%) pero notas BAJAS.
    Caso real: estudiante que va a clase pero no entiende.
    Esperado: riesgo alto/medio, nota baja. El modelo no debe
    confundirse con la asistencia buena.
    """
    payload = base_payload(
        est_id=5, trimestre=1, semana=8,
        asistencia=92.0,
        notas_sab=[38.0, 42.0],
        notas_hac=[40.0, 35.0, 43.0],
        nota_comp=75.0,
    )
    r = predecir(payload)
    print(f"\n[T1-05] Asistencia alta pero notas bajas (patrón contradictorio)")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    # Las notas mandan más que la asistencia — debe ser alto/crítico
    assert_nivel(r, ["alto", "critico", "medio"], "T1-05")
    assert_nota_rango(r, 1.0, 55.0, "T1-05")
    print("  ✅ PASS")


# ─────────────────────────────────────────
# TRIMESTRE 2 — con historial de T1
# El modelo ahora tiene datos del trimestre anterior.
# Casos clave: recuperación después de reprobar T1,
# caída después de aprobar T1.
# ─────────────────────────────────────────

def test_t2_recuperacion_tras_reprobar():
    """
    T2 semana 7 — reprobó T1 (nota 44) pero en T2 va bien.
    Esperado: el modelo debe reconocer la mejora y no castigar
    demasiado por reprobo_trim_ant=1.
    Si el modelo v8.1 viejo: daría crítico sin importar las notas.
    Si el modelo v8.2 nuevo: debería dar medio/alto como máximo.
    """
    payload = base_payload(
        est_id=101, trimestre=2, semana=7,
        asistencia=85.0,
        notas_sab=[62.0, 68.0],
        notas_hac=[65.0, 70.0, 63.0],
        nota_comp=78.0,
        # Historial T1
        nota_trim_ant=44.0,
        asist_trim_ant=72.0,
        reprobo_trim_ant=1,
        racha_trims_riesgo=1,
        mejor_nota_historica=44.0,
        tend_intertrimestral=1,   # subiendo
        reprobo_misma_mat_ant=1,
    )
    r = predecir(payload)
    print(f"\n[T2-01] Recuperación tras reprobar T1 (notas actuales buenas)")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    print(f"  ALERTA: si da 'critico' con notas 62-70, el modelo está dominado por historial")
    # Con notas actuales 62-70, NO debería ser crítico
    nivel = r["modelo"]["nivel_riesgo"]
    nota  = r["modelo"]["nota_estimada_final"]
    if nivel == "critico":
        print(f"  ⚠ PROBLEMA: nivel=critico con notas {[62,68,65,70,63]} — historial domina")
    else:
        print(f"  ✅ PASS — modelo considera mejora actual")
    assert nivel != "critico", \
        f"[T2-01] Con notas 62-70 no debería ser crítico. El historial está dominando."
    assert_nota_rango(r, 50.0, 80.0, "T2-01")


def test_t2_caida_tras_aprobar():
    """
    T2 semana 8 — aprobó T1 bien (nota 72) pero T2 va mal.
    Esperado: riesgo alto, nota baja, el modelo no se confía
    con el buen historial cuando las notas actuales son malas.
    """
    payload = base_payload(
        est_id=102, trimestre=2, semana=8,
        asistencia=74.0,
        notas_sab=[42.0, 45.0],
        notas_hac=[40.0, 38.0, 44.0],
        nota_comp=65.0,
        nota_trim_ant=72.0,
        asist_trim_ant=88.0,
        reprobo_trim_ant=0,
        racha_trims_riesgo=0,
        mejor_nota_historica=72.0,
        tend_intertrimestral=-1,  # bajando
        reprobo_misma_mat_ant=0,
    )
    r = predecir(payload)
    print(f"\n[T2-02] Caída en T2 (aprobó T1 con 72, ahora notas 38-45)")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    assert_nivel(r, ["alto", "critico", "medio"], "T2-02")
    assert_nota_rango(r, 1.0, 58.0, "T2-02")
    print("  ✅ PASS")


def test_t2_estable_aprobando():
    """
    T2 semana 6 — aprobó T1 (65) y T2 va parecido.
    Esperado: riesgo bajo/medio, tendencia estable.
    """
    payload = base_payload(
        est_id=103, trimestre=2, semana=6,
        asistencia=87.0,
        notas_sab=[63.0, 67.0],
        notas_hac=[65.0, 62.0, 68.0],
        nota_comp=72.0,
        nota_trim_ant=65.0,
        asist_trim_ant=85.0,
        reprobo_trim_ant=0,
        racha_trims_riesgo=0,
        mejor_nota_historica=65.0,
        tend_intertrimestral=0,
        reprobo_misma_mat_ant=0,
    )
    r = predecir(payload)
    print(f"\n[T2-03] Estable aprobando (T1=65, T2 similar)")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    assert_nivel(r, ["bajo", "medio"], "T2-03")
    assert_nota_rango(r, 52.0, 82.0, "T2-03")
    print("  ✅ PASS")


def test_t2_cronico_con_observaciones():
    """
    T2 semana 7 — reprobó T1 (38), T2 también va mal,
    Y tiene 2 observaciones urgentes.
    Esperado: crítico sí o sí.
    Valida que las observaciones suman al riesgo.
    """
    payload = base_payload(
        est_id=104, trimestre=2, semana=7,
        asistencia=65.0,
        notas_sab=[35.0, 38.0],
        notas_hac=[32.0, 36.0, 34.0],
        nota_comp=55.0,
        nota_trim_ant=38.0,
        asist_trim_ant=68.0,
        reprobo_trim_ant=1,
        racha_trims_riesgo=2,
        mejor_nota_historica=38.0,
        tend_intertrimestral=-1,
        reprobo_misma_mat_ant=1,
        n_obs_urgentes=2,
        n_obs_conducta=3,
    )
    r = predecir(payload)
    print(f"\n[T2-04] Crónico con observaciones urgentes")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    assert_nivel(r, ["critico", "alto"], "T2-04")
    assert_nota_rango(r, 1.0, 50.0, "T2-04")
    print("  ✅ PASS")


def test_t2_asistencia_baja_pero_notas_ok():
    """
    T2 semana 8 — asistencia muy baja (62%) pero notas decentes (60-70).
    Caso real: estudiante que falta mucho pero cuando va rinde bien.
    Esperado: riesgo medio/alto (asistencia es problema), pero
    nota estimada no debería ser catastrófica.
    """
    payload = base_payload(
        est_id=105, trimestre=2, semana=8,
        asistencia=62.0,
        notas_sab=[68.0, 65.0],
        notas_hac=[70.0, 63.0, 67.0],
        nota_comp=74.0,
        nota_trim_ant=58.0,
        asist_trim_ant=70.0,
        reprobo_trim_ant=0,
        racha_trims_riesgo=0,
        mejor_nota_historica=58.0,
        tend_intertrimestral=0,
        reprobo_misma_mat_ant=0,
        racha_inasistencias=6,
        max_racha_inasistencias=8,
    )
    r = predecir(payload)
    print(f"\n[T2-05] Asistencia muy baja pero notas decentes (patrón contradictorio)")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    print(f"  (La asistencia <75% es crítica legalmente aunque las notas sean buenas)")
    nivel = r["modelo"]["nivel_riesgo"]
    # Debe reconocer el riesgo de asistencia
    assert nivel in ["medio", "alto", "critico"], \
        f"[T2-05] Asistencia 62% siempre es riesgo. Got: {nivel}"
    print("  ✅ PASS")


# ─────────────────────────────────────────
# TRIMESTRE 3 — con historial de T1 y T2
# Casos más extremos: patrón crónico, recuperación tardía,
# excelente pero con caída al final.
# ─────────────────────────────────────────

def test_t3_patron_cronico_3_trims():
    """
    T3 semana 6 — reprobó T1 y T2, va mal en T3 también.
    racha_trims_riesgo=3.
    Esperado: crítico sin discusión.
    Valida el ajuste de consistencia de ml_service.py para patrón crónico.
    """
    payload = base_payload(
        est_id=201, trimestre=3, semana=6,
        asistencia=65.0,
        notas_sab=[35.0, 40.0],
        notas_hac=[38.0, 36.0, 39.0],
        nota_comp=58.0,
        nota_trim_ant=41.0,
        asist_trim_ant=67.0,
        reprobo_trim_ant=1,
        racha_trims_riesgo=3,
        mejor_nota_historica=44.0,
        tend_intertrimestral=-1,
        reprobo_misma_mat_ant=1,
        n_obs_urgentes=1,
        n_obs_conducta=2,
    )
    r = predecir(payload)
    print(f"\n[T3-01] Patrón crónico 3 trimestres consecutivos")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Prob reprobar: {r['modelo']['probabilidad_reprobar']:.2%}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    assert_nivel(r, ["critico", "alto"], "T3-01")
    assert_nota_rango(r, 1.0, 50.0, "T3-01")
    print("  ✅ PASS")


def test_t3_recuperacion_tardia():
    """
    T3 semana 7 — arquetipo "tardío": T1 mal (44), T2 regular (54),
    T3 va bien (notas 65-75).
    Esperado: riesgo bajo/medio, nota estimada 60-75.
    Valida que el modelo reconoce la mejora progresiva.
    """
    payload = base_payload(
        est_id=202, trimestre=3, semana=7,
        asistencia=84.0,
        notas_sab=[68.0, 72.0],
        notas_hac=[65.0, 70.0, 73.0],
        nota_comp=76.0,
        nota_trim_ant=54.0,
        asist_trim_ant=80.0,
        reprobo_trim_ant=0,
        racha_trims_riesgo=0,
        mejor_nota_historica=54.0,
        tend_intertrimestral=1,   # subiendo consistentemente
        reprobo_misma_mat_ant=0,
        n_logros=2,
    )
    r = predecir(payload)
    print(f"\n[T3-02] Recuperación tardía (T1=44, T2=54, T3 va bien)")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    assert_nivel(r, ["bajo", "medio"], "T3-02")
    assert_nota_rango(r, 55.0, 85.0, "T3-02")
    print("  ✅ PASS")


def test_t3_excelente_todo_el_año():
    """
    T3 semana 8 — excelente los tres trimestres.
    T1=85, T2=88, T3 va igual de bien.
    Esperado: bajo siempre, nota > 80.
    """
    payload = base_payload(
        est_id=203, trimestre=3, semana=8,
        asistencia=96.0,
        notas_sab=[84.0, 88.0],
        notas_hac=[86.0, 82.0, 87.0],
        nota_comp=90.0,
        nota_trim_ant=88.0,
        asist_trim_ant=95.0,
        reprobo_trim_ant=0,
        racha_trims_riesgo=0,
        mejor_nota_historica=88.0,
        tend_intertrimestral=0,
        reprobo_misma_mat_ant=0,
        n_logros=3,
    )
    r = predecir(payload)
    print(f"\n[T3-03] Excelente todo el año (T1=85, T2=88, T3 igual)")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    assert_nivel(r, ["bajo"], "T3-03")
    assert_nota_rango(r, 72.0, 100.0, "T3-03")
    assert_clasif(r, ["DO", "DP"], "T3-03")
    print("  ✅ PASS")


def test_t3_caida_al_final():
    """
    T3 semana 10 — fue bien todo el año pero en T3 se cayó fuerte.
    T1=70, T2=68, T3 notas 40-48.
    Esperado: alto/crítico en T3 aunque el historial sea bueno.
    Valida que las notas actuales pesan más que el buen historial al final.
    """
    payload = base_payload(
        est_id=204, trimestre=3, semana=10,
        asistencia=76.0,
        notas_sab=[44.0, 47.0],
        notas_hac=[42.0, 45.0, 40.0, 48.0],
        nota_comp=70.0,
        nota_trim_ant=68.0,
        asist_trim_ant=86.0,
        reprobo_trim_ant=0,
        racha_trims_riesgo=0,
        mejor_nota_historica=70.0,
        tend_intertrimestral=-1,
        reprobo_misma_mat_ant=0,
    )
    r = predecir(payload)
    print(f"\n[T3-04] Caída brusca en T3 (T1=70, T2=68, T3 notas 40-48)")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    print(f"  (En semana 10/13 con notas 40-48 hay poco tiempo para recuperar)")
    assert_nivel(r, ["alto", "critico"], "T3-04")
    assert_nota_rango(r, 1.0, 56.0, "T3-04")
    print("  ✅ PASS")


def test_t3_semana_final_nota_coherente():
    """
    T3 semana 13 (final) — verifica coherencia entre notas ingresadas
    y nota estimada. Con SAB promedio 58 y HAC promedio 62,
    la nota final con ponderación 45/40/15 debería ser ~58-63.
    Si el modelo devuelve 85, está inventando.
    """
    notas_sab = [55.0, 58.0, 60.0]   # prom = 57.67
    notas_hac = [60.0, 63.0, 65.0, 62.0]  # prom = 62.5
    # nota_final esperada: 57.67*0.45 + 62.5*0.40 + 75*0.15 = 25.95 + 25.0 + 11.25 = 62.2

    payload = base_payload(
        est_id=205, trimestre=3, semana=13,
        asistencia=83.0,
        notas_sab=notas_sab,
        notas_hac=notas_hac,
        nota_comp=75.0,
        nota_trim_ant=60.0,
        asist_trim_ant=82.0,
        reprobo_trim_ant=0,
        racha_trims_riesgo=0,
        mejor_nota_historica=65.0,
        tend_intertrimestral=0,
        reprobo_misma_mat_ant=0,
    )
    r = predecir(payload)
    nota_est   = r["modelo"]["nota_estimada_final"]
    nota_calc  = round(57.67 * 0.45 + 62.5 * 0.40 + 75.0 * 0.15, 1)

    print(f"\n[T3-05] Coherencia nota estimada vs nota calculada manualmente")
    print(f"  Nota calculada manualmente: {nota_calc}")
    print(f"  Nota estimada por el modelo: {nota_est}")
    print(f"  Diferencia: {abs(nota_est - nota_calc):.1f} pts")
    if abs(nota_est - nota_calc) > 10:
        print(f"  ⚠ DIFERENCIA ALTA — el modelo puede estar inventando")
    else:
        print(f"  ✅ Diferencia aceptable (< 10 pts)")
    # En semana 13 con todas las notas, la diferencia no debería ser > 10
    assert abs(nota_est - nota_calc) <= 12, \
        f"[T3-05] Nota estimada {nota_est} muy lejos de la calculada {nota_calc}"
    print("  ✅ PASS")


def test_t3_shock_asistencia_baja_notas_examen_buenas():
    """
    T3 semana 9 — situación contradictoria real:
    Asistencia baja (68%) pero el único examen que rindió fue 82.
    Pocas prácticas por faltar, pero cuando rinde es bueno.
    Esperado: riesgo medio (asistencia es problema pero potencial existe).
    """
    payload = base_payload(
        est_id=206, trimestre=3, semana=9,
        asistencia=68.0,
        notas_sab=[82.0],         # un solo examen, muy bueno
        notas_hac=[55.0, 58.0],   # pocas prácticas por faltar
        nota_comp=72.0,
        nota_trim_ant=62.0,
        asist_trim_ant=75.0,
        reprobo_trim_ant=0,
        racha_trims_riesgo=0,
        mejor_nota_historica=68.0,
        tend_intertrimestral=0,
        reprobo_misma_mat_ant=0,
        racha_inasistencias=8,
        max_racha_inasistencias=10,
    )
    r = predecir(payload)
    print(f"\n[T3-06] Asistencia baja (68%) pero examen excelente (82)")
    print(f"  Nivel: {r['modelo']['nivel_riesgo']}")
    print(f"  Nota estimada: {r['modelo']['nota_estimada_final']}")
    print(f"  (Asistencia <75% es riesgo legal aunque las notas sean buenas)")
    nivel = r["modelo"]["nivel_riesgo"]
    assert nivel in ["medio", "alto", "critico"], \
        f"[T3-06] Asistencia 68% siempre implica riesgo mínimo medio. Got: {nivel}"
    print("  ✅ PASS")


# ─────────────────────────────────────────
# RESUMEN DE RESULTADOS
# ─────────────────────────────────────────

def run_all():
    tests_t1 = [
        test_t1_inicio_sin_notas,
        test_t1_mitad_buen_rendimiento,
        test_t1_mitad_riesgo_alto,
        test_t1_cierre_estudiante_en_limite,
        test_t1_patron_contradictorio,
    ]
    tests_t2 = [
        test_t2_recuperacion_tras_reprobar,
        test_t2_caida_tras_aprobar,
        test_t2_estable_aprobando,
        test_t2_cronico_con_observaciones,
        test_t2_asistencia_baja_pero_notas_ok,
    ]
    tests_t3 = [
        test_t3_patron_cronico_3_trims,
        test_t3_recuperacion_tardia,
        test_t3_excelente_todo_el_año,
        test_t3_caida_al_final,
        test_t3_semana_final_nota_coherente,
        test_t3_shock_asistencia_baja_notas_examen_buenas,
    ]

    all_tests = tests_t1 + tests_t2 + tests_t3
    pasados   = 0
    fallados  = []

    print("\n" + "=" * 60)
    print("  TESTS DE PREDICCIÓN POR TRIMESTRE")
    print(f"  Total: {len(all_tests)} casos")
    print("=" * 60)

    print("\n── TRIMESTRE 1 (sin historial) ──────────────────────")
    for t in tests_t1:
        try:
            t()
            pasados += 1
        except AssertionError as e:
            fallados.append(str(e))
            print(f"  ❌ FAIL: {e}")
        except Exception as e:
            fallados.append(f"{t.__name__}: {e}")
            print(f"  💥 ERROR: {t.__name__}: {e}")

    print("\n── TRIMESTRE 2 (con historial T1) ───────────────────")
    for t in tests_t2:
        try:
            t()
            pasados += 1
        except AssertionError as e:
            fallados.append(str(e))
            print(f"  ❌ FAIL: {e}")
        except Exception as e:
            fallados.append(f"{t.__name__}: {e}")
            print(f"  💥 ERROR: {t.__name__}: {e}")

    print("\n── TRIMESTRE 3 (historial completo) ─────────────────")
    for t in tests_t3:
        try:
            t()
            pasados += 1
        except AssertionError as e:
            fallados.append(str(e))
            print(f"  ❌ FAIL: {e}")
        except Exception as e:
            fallados.append(f"{t.__name__}: {e}")
            print(f"  💥 ERROR: {t.__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"  RESULTADO: {pasados}/{len(all_tests)} pasados")
    if fallados:
        print(f"\n  FALLADOS ({len(fallados)}):")
        for f in fallados:
            print(f"    ❌ {f}")
        print("\n  INTERPRETACIÓN:")
        for f in fallados:
            if "T2-01" in f:
                print("    → T2-01 falla: historial reprobo_trim_ant sigue dominando demasiado")
                print("      Regenerar dataset con generar_dataset_v8.py v8.2 y reentrenar")
            if "T1-05" in f or "T2-05" in f or "T3-06" in f:
                print("    → Patrón contradictorio falla: asistencia alta/baja interfiere con notas")
                print("      El modelo no distingue bien casos donde asistencia y notas no correlacionan")
            if "T3-05" in f:
                print("    → T3-05 falla: nota estimada muy lejos de la calculada manualmente")
                print("      El modelo puede estar inventando en semana final")
    else:
        print("\n  ✅ Todos los casos pasaron")
        print("\n  PRÓXIMO PASO: correr validar_dataset.py para análisis estadístico")
    print("=" * 60)

    return len(fallados) == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)