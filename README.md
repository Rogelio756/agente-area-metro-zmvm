# AquaInfer ZMVM 🗺️
**Agente de predicción de riesgo de inundación — Zona Metropolitana del Valle de México**

> Hackathon ConciencIA · Equipo AquaInfer · 2026
> Agente complementario de [AquaInfer CDMX](https://github.com/Rogelio756/hackaton_agentes_concienc_ia)

---

## ¿Qué hace?

Predice en tiempo real el riesgo de inundación para **28 municipios** de Estado de México e Hidalgo que conforman la Zona Metropolitana, usando:

- **Riesgo base** por municipio (fuente: CENAPRED / Atlas Nacional de Riesgos)
- **Clima en tiempo real** via Open-Meteo API (lluvia hoy, acumulado 7/30 días)
- **Heurística climática** que ajusta el riesgo según condiciones actuales
- **Agente 3** generador de alertas (Watsonx → Claude → Reglas)

---

## Endpoints

| Ruta | Descripción |
|------|-------------|
| `GET /predict` | JSON con los 28 municipios (actualiza cada 5 min) |
| `GET /predict/{municipio}` | Predicción para un municipio específico |
| `GET /zonas` | Lista completa con riesgo base por municipio |
| `GET /health` | Health check |

### Formato de respuesta

```json
{
  "timestamp": "2026-06-06T20:00:00Z",
  "zones": {
    "Nezahualcóyotl": {
      "lluvia_mm": 8.2,
      "prob_lluvia": 0.71,
      "nivel_riesgo": 5,
      "alerta": "ALERTA MAXIMA en Nezahualcóyotl...",
      "fuente": "heuristica_zmvm"
    }
  }
}
```

---

## Municipios cubiertos (28)

**Estado de México:** Ecatepec, Nezahualcóyotl, Chimalhuacán, Chalco, Valle de Chalco, La Paz, Ixtapaluca, Chicoloapan, Texcoco, Atenco, Tlalnepantla, Naucalpan, Atizapán, Cuautitlán Izcalli, Cuautitlán, Tultitlán, Coacalco, Tecámac, Nicolás Romero, Tultepec, Nextlalpan, Zumpango, Toluca, Metepec

**Hidalgo:** Tizayuca, Pachuca, Mineral de la Reforma, Zapotlán de Juárez

---

## Deploy en Railway

```bash
railway login
railway init
railway up
```

Variables opcionales (mejoran el texto de alerta):
```
WATSONX_API_KEY=...
WATSONX_PROJECT_ID=...
ANTHROPIC_API_KEY=...
```

---

## Agente hermano — CDMX
Para las 16 alcaldías de CDMX con modelos XGBoost ML entrenados:
👉 https://github.com/Rogelio756/hackaton_agentes_concienc_ia
