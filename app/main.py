"""
Chatbot API - Multi-plataforma con RAG dinámico sobre MongoDB
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from app.core.config import get_settings
from app.models.platform import Platform
from app.models.conversation import ChatTurn
from app.api import chat, platforms, recommendations, webhook
from app.api import widget
from app.api import documents

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    client = AsyncIOMotorClient(settings.meta_mongodb_uri)
    await init_beanie(
        database=client[settings.meta_mongodb_db],
        document_models=[Platform, ChatTurn],
    )
    app.state.mongo_client = client
    yield
    # Shutdown
    client.close()
    from app.db.mongo_pool import close_all
    close_all()


app = FastAPI(
    title="Chatbot API",
    description=(
        "API de chatbot multi-plataforma con RAG dinámico sobre MongoDB. "
        "Cada plataforma y organización tiene su propio índice de conocimiento."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(platforms.router)
app.include_router(recommendations.router)
app.include_router(widget.router)
app.include_router(webhook.router)
app.include_router(documents.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/")
async def root():
    return {
        "name": "Chatbot API",
        "docs": "/docs",
        "usage": {
            "chat": "POST /chat  [X-Platform-Id, X-API-Key, X-Org-Id headers]",
            "recommendations": "POST /recommendations  [X-Platform-Id, X-API-Key, X-Org-Id headers]",
            "admin": "POST /platforms  [X-Admin-Key header]",
        },
    }
