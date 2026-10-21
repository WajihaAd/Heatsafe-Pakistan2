# HeatSafe Pakistan — Consolidated Project

This archive combines the original uploaded repository with the latest dashboard/RAG upgrade and the missing `streamlit_app/data_services.py` in its correct location. It is a development build, not a fully verified final submission.

## GitHub browser deployment
1. On your fork, create or select a NEW development branch. Do not merge to main yet.
2. Extract this ZIP. Upload the *contents*, preserving directory paths, to that branch. Do not upload the ZIP as a single file.
3. For files already in your fork, preserve newer work unless you intentionally choose this version. Most importantly check `streamlit_app/app.py`, `streamlit_app/rag_engine.py`, `streamlit_app/data_services.py`, `streamlit_app/requirements.txt`, `knowledge/guidance.json`.
4. Streamlit Community Cloud: Repository `WajihaAd/HeatSafe-Pakistan`, branch YOUR development branch, main file `streamlit_app/app.py`.
5. Inspect app logs on failure.

## Known limitations
- This is NOT a verified full official-document RAG corpus. Retrieval uses TF-IDF by default, with optional Groq LLM API if configured.
- Prefect-to-dashboard publishing is NOT automatically implemented; the dashboard can read public published results if configured.
- Cloud deployment has not been validated from this environment.
- The trained model has version compatibility risks; risk estimates are experimental.
- The app must not claim compliance with a Hugging Face Spaces hosting requirement when deployed only to Streamlit.
