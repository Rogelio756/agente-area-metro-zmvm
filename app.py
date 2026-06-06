"""
AquaInfer ZMVM — Agente Zona Metropolitana del Valle de México
28 municipios de Estado de México e Hidalgo
Predicción basada en heurística climática + Open-Meteo en tiempo real
Desplegado en Railway
"""

import os
import json
import threading
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Configuración ────────────────────────────────────────
BASE      = Path(__file__).parent
ZMVM_PATH = BASE / 'referencia_zmvm.csv'
STATE     = BASE / 'live_state_zmvm.json'
INTERVALO = int(os.environ.get('INTERVALO_SEG', 300))

zmvm = pd.read_csv(ZMVM_PATH)
print(f"ZMVM cargada: {len(zmvm)} municipios")


# ── Heurística de riesgo ─────────────────────────────────
def riesgo_heuristico(riesgo_base: int, lluvia_mm: float,
                      rain_7d: float, mes: int) -> int:
    """
    Ajusta el riesgo base estático con señales climáticas en tiempo real.
      +1 si lluvia hoy >= 20mm
      +1 si acumulado 7 días >= 50mm
      +1 si temporada lluvias (may-oct) y zona vulnerable (base >= 3)
      -1 si sin lluvia y fuera de temporada
    Resultado clampado [1, 5]
    """
    delta = 0
    if lluvia_mm >= 20:
        delta += 1
    if rain_7d >= 50:
        delta += 1
    if 5 <= mes <= 10 and riesgo_base >= 3:
        delta += 1
    if lluvia_mm == 0 and rain_7d < 5 and not (5 <= mes <= 10):
        delta -= 1
    return max(1, min(5, riesgo_base + delta))


def prob_lluvia_heuristica(lluvia_mm: float, rain_7d: float, mes: int) -> float:
    """
    Estima probabilidad de lluvia intensa mañana usando señales simples.
    """
    prob = 0.1
    if 5 <= mes <= 10:
        prob += 0.25
    if lluvia_mm >= 10:
        prob += 0.30
    elif lluvia_mm >= 5:
        prob += 0.15
    if rain_7d >= 40:
        prob += 0.15
    elif rain_7d >= 20:
        prob += 0.08
    return round(min(prob, 0.95), 2)


# ── Generación de alertas ────────────────────────────────
def alerta_watsonx(nombre, lluvia_mm, prob, nivel, rain_7d):
    try:
        api_key    = os.environ.get('WATSONX_API_KEY', '')
        project_id = os.environ.get('WATSONX_PROJECT_ID', '')
        if not api_key or not project_id:
            return None
        token = requests.post(
            'https://iam.cloud.ibm.com/identity/token',
            data={'grant_type': 'urn:ibm:params:oauth:grant-type:apikey', 'apikey': api_key},
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        ).json()['access_token']
        nivel_txt = {1:'muy bajo',2:'bajo',3:'medio',4:'alto',5:'muy alto'}.get(nivel,'medio')
        prompt = (f"Eres AquaInfer ZMVM. Genera alerta breve (max 2 oraciones) para {nombre}. "
                  f"Lluvia hoy: {lluvia_mm}mm, acumulado 7d: {rain_7d}mm, "
                  f"prob lluvia intensa manana: {prob*100:.0f}%, riesgo: {nivel}/5 ({nivel_txt}). "
                  f"Directo y util para ciudadanos. Sin markdown.")
        resp = requests.post(
            'https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29',
            headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
            json={'model_id': 'ibm/granite-13b-chat-v2', 'input': prompt,
                  'parameters': {'max_new_tokens': 120, 'temperature': 0.5},
                  'project_id': project_id},
            timeout=15
        ).json()
        return resp['results'][0]['generated_text'].strip()
    except Exception:
        return None


def alerta_claude(nombre, lluvia_mm, prob, nivel, rain_7d):
    try:
        import anthropic
        api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        if not api_key:
            return None
        client = anthropic.Anthropic(api_key=api_key)
        nivel_txt = {1:'muy bajo',2:'bajo',3:'medio',4:'alto',5:'muy alto'}.get(nivel,'medio')
        msg = client.messages.create(
            model="claude-haiku-4-5", max_tokens=120,
            messages=[{"role": "user", "content": (
                f"Eres AquaInfer ZMVM. Genera alerta breve (max 2 oraciones) para {nombre}. "
                f"Lluvia hoy: {lluvia_mm}mm, acumulado 7d: {rain_7d}mm, "
                f"prob lluvia intensa manana: {prob*100:.0f}%, riesgo: {nivel}/5 ({nivel_txt}). "
                f"Directo y util para ciudadanos. Sin markdown."
            )}]
        )
        return msg.content[0].text.strip()
    except Exception:
        return None


def alerta_reglas(nombre, lluvia_mm, prob, nivel, rain_7d):
    prob_pct  = round(prob * 100)
    nivel_txt = {1:'muy bajo',2:'bajo',3:'medio',4:'alto',5:'muy alto'}.get(nivel,'medio')
    if nivel >= 5:
        return (f"ALERTA MAXIMA en {nombre}. "
                f"Acumulado semanal {rain_7d}mm y {prob_pct}% prob lluvia intensa manana. "
                f"Riesgo MUY ALTO de inundacion.")
    elif nivel >= 4:
        return (f"Alerta en {nombre}. Lluvia {lluvia_mm}mm hoy, acumulado {rain_7d}mm en 7 dias. "
                f"Riesgo {nivel_txt} de inundacion. Mantente informado.")
    elif nivel >= 3:
        return (f"Aviso preventivo en {nombre}. {lluvia_mm}mm registrados hoy. "
                f"Probabilidad lluvia intensa manana: {prob_pct}%. Riesgo {nivel_txt}.")
    elif nivel >= 2:
        return f"{nombre}: Lluvia moderada ({lluvia_mm}mm). Riesgo {nivel_txt}. Sin alerta activa."
    else:
        return f"{nombre}: Condiciones normales. {lluvia_mm}mm hoy. Riesgo muy bajo."


def generar_alerta(nombre, lluvia_mm, prob, nivel, rain_7d):
    return (alerta_watsonx(nombre, lluvia_mm, prob, nivel, rain_7d) or
            alerta_claude(nombre, lluvia_mm, prob, nivel, rain_7d) or
            alerta_reglas(nombre, lluvia_mm, prob, nivel, rain_7d))


# ── Ciclo de predicción ──────────────────────────────────
def run_ciclo():
    now = datetime.now(timezone.utc)
    hoy = now.date()
    print(f"[{now.strftime('%H:%M:%S UTC')}] Actualizando ZMVM...")
    zones = {}

    for _, row in zmvm.iterrows():
        nombre      = row['municipio']
        lat         = row['lat']
        lon         = row['lon']
        riesgo_base = int(row['riesgo_base'])

        try:
            url = (f'https://api.open-meteo.com/v1/forecast'
                   f'?latitude={lat}&longitude={lon}'
                   f'&daily=precipitation_sum,temperature_2m_max,temperature_2m_min'
                   f'&timezone=America%2FMexico_City&past_days=30&forecast_days=1')
            d   = requests.get(url, timeout=10).json()['daily']
            ll  = d['precipitation_sum']

            lluvia_mm = round(ll[-1] or 0.0, 1)
            rain_7d   = round(sum(v or 0 for v in ll[-8:-1]), 1)
            mes       = hoy.month

            prob     = prob_lluvia_heuristica(lluvia_mm, rain_7d, mes)
            nivel    = riesgo_heuristico(riesgo_base, lluvia_mm, rain_7d, mes)
            alerta   = generar_alerta(nombre, lluvia_mm, prob, nivel, rain_7d)

            zones[nombre] = {
                'lluvia_mm'   : lluvia_mm,
                'prob_lluvia' : prob,
                'nivel_riesgo': nivel,
                'alerta'      : alerta,
                'fuente'      : 'heuristica_zmvm',
            }
            print(f"  {nombre}: {lluvia_mm}mm | riesgo {nivel}/5")

        except Exception as e:
            print(f"  ERROR {nombre}: {e}")
            zones[nombre] = {
                'lluvia_mm': 0.0, 'prob_lluvia': 0.0,
                'nivel_riesgo': riesgo_base,
                'alerta': 'Datos no disponibles temporalmente.',
                'fuente': 'heuristica_zmvm',
            }

    payload = {'timestamp': now.strftime('%Y-%m-%dT%H:%M:%SZ'), 'zones': zones}
    STATE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"  {len(zones)} municipios ZMVM actualizados OK")
    return payload


def loop_background():
    while True:
        try:
            run_ciclo()
        except Exception as e:
            print(f"ERROR loop: {e}")
        time.sleep(INTERVALO)


# ── FastAPI ──────────────────────────────────────────────
app = FastAPI(
    title="AquaInfer ZMVM API",
    description=(
        "Predicción de riesgo de inundación para la Zona Metropolitana "
        "del Valle de México. 28 municipios de Estado de México e Hidalgo. "
        "Heurística climática en tiempo real via Open-Meteo."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    threading.Thread(target=loop_background, daemon=True).start()


@app.get("/predict", summary="Predicciones en vivo — 28 municipios ZMVM")
def predict():
    """
    Devuelve timestamp + zones para los 28 municipios ZMVM.
    Se actualiza automáticamente cada 5 minutos.
    """
    if STATE.exists():
        return JSONResponse(json.loads(STATE.read_text(encoding='utf-8')))
    return JSONResponse(run_ciclo())


@app.get("/predict/{zona}", summary="Predicción para un municipio específico")
def predict_zona(zona: str):
    if STATE.exists():
        data = json.loads(STATE.read_text(encoding='utf-8'))
        if zona in data['zones']:
            return JSONResponse({
                'timestamp': data['timestamp'],
                'zona': zona,
                **data['zones'][zona]
            })
    return JSONResponse({'error': f'Municipio "{zona}" no encontrado'}, status_code=404)


@app.get("/zonas", summary="Lista de municipios disponibles")
def zonas():
    return {
        'total'  : len(zmvm),
        'metodo' : 'heuristica_climatica',
        'fuente_riesgo_base': 'CENAPRED / Atlas Nacional de Riesgos',
        'fuente_clima': 'Open-Meteo API (tiempo real)',
        'municipios': zmvm[['municipio','estado','riesgo_base']].to_dict(orient='records')
    }


@app.get("/health", summary="Health check")
def health():
    last = None
    if STATE.exists():
        data = json.loads(STATE.read_text(encoding='utf-8'))
        last = data.get('timestamp')
    return {'status': 'ok', 'last_update': last, 'zonas': len(zmvm), 'version': '1.0.0'}
