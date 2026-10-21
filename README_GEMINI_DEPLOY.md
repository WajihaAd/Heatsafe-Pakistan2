# HeatSafe Pakistan — Gemini Weather Dashboard (development build)

## Security first
The Gemini key posted in chat must be revoked. Create a NEW key in Google AI Studio. Never commit a key into GitHub. In Streamlit Cloud > Manage app > Settings > Secrets, add:

```toml
GEMINI_API_KEY = "PASTE_NEW_KEY_HERE"
```

## Deploy in Streamlit Community Cloud
Repository: WajihaAd/HeatSafe-Pakistan (your fork)
Branch: your NEW testing branch
Main file path: `streamlit_app/app.py`
App URL: any available name.

## Updating existing branch (safer)
Only replace/add these files, not the original data, model, Prefect or GitHub Actions:
- `streamlit_app/app.py`
- `streamlit_app/gemini_rag.py` (NEW)
- `streamlit_app/rag_engine.py` (retain or replace with included version)
- `streamlit_app/data_services.py` (MUST be in streamlit_app, NOT repository root)
- `streamlit_app/requirements.txt`
- `.streamlit/secrets.toml.example` (example only; no real key)

The full ZIP includes your previous repository snapshot and the files above, so it is a reference copy. **Do not overwrite newer changes in GitHub without comparing them.**

## What works and what doesn't
- Live current Karachi weather and seven-day forecast from Open-Meteo API.
- Daily weather outlook, hourly charts, UV, sunrise/sunset, wind and humidity.
- Open-Meteo modeled air-quality PM2.5 and US AQI when API available; NOT OpenAQ sensor data.
- Historical charts from committed weather dataset, existing experimental heat-risk model.
- Gemini answers for trusted-document questions if GEMINI_API_KEY is configured; retrieval-only fallback otherwise.
- Grounded Gemini live-weather explanation on demand.
- Prefect result publishing still NOT connected unless PUBLISHED_RESULTS_URL is configured and a publisher is deployed.
- Knowledge base currently contains only a few short source-linked summaries; it is NOT a verified full official-document RAG corpus.
- Gemini API may be rate limited, unavailable, or require a paid tier depending on your account.
- No guarantee of successful cloud deployment; check Streamlit logs.
- Predictions are research proxies, not validated medical outcomes or official heat warnings.

## Important data limitations
Current weather is an Open-Meteo API estimate, not an on-site observation. Forecast is a model projection, not observed future weather. Historical data covers a different period from the OpenAQ sensor data.

## Troubleshooting
If `ModuleNotFoundError: data_services`, verify file is `streamlit_app/data_services.py` on the deployed branch.
If Gemini fails, check Streamlit Secrets, model availability, API quota, and logs. Do not post keys.
If Streamlit cannot load the ML model due to sklearn version mismatch, report logs; do not change the model silently.
