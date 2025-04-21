# Candidates Who'sWho: RAG Election Info App (still in development)

## Features
- Crawl and process election data (web, PDFs, multimodal)
- Hybrid retrieval (graph + vector)
- Neo4j + ChromaDB
- Gradio chat UI with filters/visuals
- Vercel deployment ready

## Setup
1. `pip install -r requirements.txt`
2. Set up `.env` with OpenAI, Neo4j, etc.
3. Run backend/API: `uvicorn backend/api/main:app --reload`
4. Run frontend: `python frontend/gradio_app.py`

## Data
- All crawled/processed data in `./data/crawled_data/`

## Deployment
- See `vercel.json` for config
