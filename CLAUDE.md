# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-tenant AI chatbot REST API with dynamic RAG (Retrieval Augmented Generation). Serves three platforms — **ACHO**, **GenCampus**, **GenLive** — each with their own MongoDB databases, vector indices, and system prompts. Built with FastAPI + Python 3.12.

## Commands

```bash
# Setup
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env  # fill in credentials

# Infrastructure (MongoDB 7 + Redis 7)
docker-compose up -d

# Seed initial platforms
python -m scripts.seed_platforms

# Run dev server
uvicorn app.main:app --reload --port 8000

# Admin scripts
python -m scripts.update_collections   # sync collections and system prompts
python -m scripts.check_transcripts    # debug conversation logs
python -m scripts.check_userid         # debug user lookups
```

No test suite or linter is currently configured.

## Architecture

### Request Flow

```
POST /chat
  → auth.py: validate X-Platform-Id + X-API-Key (SHA256 hashed keys)
  → history_service.py: load recent turns from Redis (24h TTL) or MongoDB
  → rag/pipeline.py: semantic search in ChromaDB (per-platform/org namespace)
  → db/mongo_query.py: LLM generates MongoDB filter → execute against platform DB
  → chat_service.py: build prompt (system prompt + history + RAG results + DB data)
  → LLM provider (OpenAI / Anthropic / Gemini) → return answer + sources
  → history_service.py: persist to Redis + MongoDB
```

### Key Modules

| Path | Responsibility |
|------|---------------|
| `app/core/config.py` | Pydantic Settings — all env vars in one place |
| `app/core/auth.py` | API key validation, platform/org context extraction |
| `app/services/chat_service.py` | Orchestrates RAG + DB query + LLM call |
| `app/services/history_service.py` | Redis cache (active sessions) + MongoDB persistence |
| `app/rag/pipeline.py` | ChromaDB/FAISS vector store, per-platform namespacing |
| `app/rag/content_indexer.py` | Chunk + embed documents for indexing |
| `app/db/schema_introspector.py` | Samples MongoDB collections → markdown schema docs (disk-cached in `./data/schemas/`) |
| `app/db/mongo_query.py` | LLM-driven dynamic query generation |
| `app/models/platform.py` | `Platform`, `Organization`, `DBConnection`, `RAGDocument` Beanie models |
| `app/models/conversation.py` | `ChatTurn` — stored in meta MongoDB |

### Authentication

- **Platform requests**: `X-Platform-Id` + `X-API-Key` headers (keys stored SHA256-hashed)
- **Admin requests**: `X-Admin-Key` = `SECRET_KEY` env var
- Optional org-level isolation via `X-Org-Id`

### Data Stores

- **Meta MongoDB** (`META_MONGODB_URI`): Platform configs, organizations, chat history
- **Platform MongoDBs**: Client data per platform (events, users, courses, etc.)
- **Redis** (`REDIS_URL`): Active session cache, 24h TTL
- **ChromaDB** (`./data/chroma/`): Vector indices, namespaced per platform + org

### LLM Provider Switching

Controlled by `LLM_PROVIDER` env var (`openai` | `anthropic` | `gemini`). All three providers are wired in `chat_service.py` with the same interface. Embedding model is also provider-specific.

### Indexing Flow

`POST /platforms/{id}/index` triggers:
1. Schema introspection (samples 5 docs/collection, generates markdown)
2. Chunking + embedding all docs
3. Storage in ChromaDB under platform namespace

Index status endpoint: `GET /platforms/{id}/index/status`

## Environment Variables

See `.env.example` for all variables. Critical ones:

```
LLM_PROVIDER=openai            # openai | anthropic | gemini
OPENAI_API_KEY=...
META_MONGODB_URI=...           # API's own database
ACHO_MONGO_URI=...             # Platform databases
GENCAMPUS_MONGO_URI=...
GENLIVE_MONGO_URI=...
REDIS_URL=redis://localhost:6379
SECRET_KEY=...                 # Admin API key
```
