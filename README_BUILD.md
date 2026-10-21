# HeatSafe Pakistan — Streamlit dashboard + evidence-grounded RAG

This is an **upgrade to the original starter**, not a medically validated heat alert system.

## What is implemented
- One five-page Streamlit application: Overview, Weather Trends, Ask HeatSafe AI, Data Quality, About.
- Real Open-Meteo current weather and committed historical weather charts.
- Existing scikit-learn model used for experimental next-hour risk if it loads and current weather passes freshness checks.
- RAG retrieval from `knowledge/guidance.json` with document IDs, sections, URLs, relevance cutoff and abstention.
- Optional LLM answer generation through a user-provided GROQ_API_KEY in Streamlit secrets; answer cites retrieved source IDs. Without a key, returns clearly labeled retrieval summaries.
- Optional FAISS + multilingual sentence-transformer embeddings (opt in by installing packages and setting USE_EMBEDDINGS=true). Default TF-IDF is intentionally lightweight for free hosting.
- Historical numerical queries include real row IDs and time ranges, though averages cite example IDs rather than a complete aggregation lineage.
- Optional public published-results reader; **Prefect publisher is NOT implemented yet**, so no claim of connected hourly results.
- Basic automated RAG tests.

## What is NOT yet complete
- No actual document ingestion from WHO/NIOSH PDFs; the small guidance JSON contains curated summaries, **not extracted verbatim sections**. Verify the linked primary documents and expand the knowledge base before claiming full document-grounded RAG compliance.
- No audited Urdu evaluation, model uncertainty, full gold row lineage, or evidence of successful scheduled Prefect runs.
- No durable Prefect publisher, no full 50k row compliance guarantee, no hackathon PDF/video evidence.
- A Streamlit site is **not** a Hugging Face Space.
- LLM and FAISS are optional because deployment memory, external API availability, and costs vary.

## Browser-only deployment
1. Open your GitHub fork. Create `streamlit_app/app.py`, `streamlit_app/rag_engine.py`, `streamlit_app/data_services.py`, `streamlit_app/requirements.txt`, `knowledge/guidance.json`, and `tests/test_rag.py` from this ZIP. Keep all existing repo files.
2. Open https://share.streamlit.io, log in with your GitHub account, select **your fork**, branch `main`, main file `streamlit_app/app.py`.
3. Deploy. Default RAG works without secrets.
4. Optional: in Streamlit App settings → Secrets, add `GROQ_API_KEY="..."` for LLM answers. Do **not** put keys in GitHub. For FAISS, uncomment two requirements and add `USE_EMBEDDINGS=true` in Secrets, but this may exceed free hosting resources.
5. Optional later: set `PUBLISHED_RESULTS_URL` to a **public HTTPS folder** with Prefect-published `latest_prediction.json` and `live_quality_report.json` after the publisher exists. Do not set it to an ephemeral Prefect filesystem path.
6. If the model fails to load, check exact scikit-learn version and compatible Python runtime; do not claim a prediction succeeded.

## Local tests (optional; GitHub Actions can run these)
`python -m unittest discover -s tests -v`

## Provenance
The existing source links in guidance.json are pointers to public health organizations; guidance texts are editorial summaries. Confirm text and exact section titles against source pages before formal judging.

## Expanding the knowledge base (genuine document ingestion)
Place authorized source text and a same-name `.json` metadata file in `knowledge/source_documents/`. See that folder's README. Run `python streamlit_app/ingest_knowledge.py` (e.g. via a GitHub Action) and commit the generated `knowledge/ingested_chunks.json`. The RAG engine loads it automatically. Default knowledge is only three short curated summaries, so add genuine sectioned source material for hackathon-quality citations.
