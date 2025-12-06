# LangChain Documentation Helper

Ingest LangChain docs into local Chroma (and optional Pinecone) using Tavily crawl/map/extract and Google GenAI embeddings.

## Quickstart
- Create a `.env` with `GOOGLE_API_KEY` and `TAVILY_API_KEY`. Add `PINECONE_API_KEY`/`PINECONE_INDEX_NAME` if you want remote storage.
- Optional tunables: `DOCS_BASE_URL`, `TAVILY_INSTRUCTIONS`, `TAVILY_MAX_DEPTH`, `TAVILY_URL_LIMIT`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `BATCH_SIZE`, `CHROMA_PERSIST_DIR`.
- Install deps: `pipenv install`.
- Run ingestion: `pipenv run python ingestion.py`.
- Chunks are stored in `chroma_db` and (if configured) Pinecone.
