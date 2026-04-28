"""
Historial de chat:
- Redis: sesión activa (TTL 24h), acceso rápido para contexto de conversación
- MongoDB: persistencia permanente para recomendaciones y auditoría
"""
from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import redis.asyncio as aioredis

from app.core.config import get_settings

settings = get_settings()

HISTORY_TTL = timedelta(hours=24)
MAX_HISTORY_MESSAGES = 20


def _key(platform_id: str, user_id: str, session_id: str) -> str:
    return f"chat:history:{platform_id}:{user_id}:{session_id}"


async def _get_redis() -> aioredis.Redis:
    return aioredis.from_url(settings.redis_url, decode_responses=True)


# ─── Redis (sesión activa) ────────────────────────────────────────────────────

async def load_history(platform_id: str, user_id: str, session_id: str) -> list[dict]:
    """Carga el historial reciente desde Redis para una sesión específica."""
    r = await _get_redis()
    try:
        raw = await r.get(_key(platform_id, user_id, session_id))
        if raw:
            history = json.loads(raw)
            print(f"[history] loaded {len(history)} msgs from Redis (session={session_id})")
            return history
        # Fallback: cargar desde MongoDB si no está en Redis (sesión reanudada)
        print(f"[history] Redis miss for session={session_id}, falling back to MongoDB")
        history = await _load_session_from_mongo(platform_id, session_id)
        print(f"[history] loaded {len(history)} msgs from MongoDB (session={session_id})")
        return history
    except Exception as e:
        print(f"[history] redis load error: {e}")
        return []
    finally:
        await r.aclose()


async def _load_session_from_mongo(
    platform_id: str, session_id: str
) -> list[dict]:
    """Reconstruye el historial de una sesión desde MongoDB (para reanudar).
    Se busca solo por platform_id + session_id para tolerar cambios en user_id
    entre sesiones (ej: WhatsApp donde user_id puede variar entre llamadas).
    """
    from app.models.conversation import ChatTurn
    try:
        turns = await ChatTurn.find(
            ChatTurn.platform_id == platform_id,
            ChatTurn.session_id == session_id,
        ).sort(ChatTurn.created_at).to_list()

        history = []
        for t in turns:
            history.append({"role": "user", "content": t.user_message})
            history.append({"role": "assistant", "content": t.assistant_message})
        return history
    except Exception as e:
        print(f"[history] mongo session load error: {e}")
        return []


async def save_history(platform_id: str, user_id: str, session_id: str, history: list[dict]) -> None:
    """Guarda el historial en Redis para una sesión específica."""
    r = await _get_redis()
    try:
        trimmed = history[-MAX_HISTORY_MESSAGES:]
        await r.set(
            _key(platform_id, user_id, session_id),
            json.dumps(trimmed, ensure_ascii=False),
            ex=int(HISTORY_TTL.total_seconds()),
        )
    except Exception as e:
        print(f"[history] redis save error: {e}")
    finally:
        await r.aclose()


async def clear_history(platform_id: str, user_id: str, session_id: str | None = None) -> None:
    """Borra el historial de Redis. Si no se pasa session_id, borra todas las sesiones del usuario."""
    r = await _get_redis()
    try:
        if session_id:
            await r.delete(_key(platform_id, user_id, session_id))
        else:
            keys = await r.keys(f"chat:history:{platform_id}:{user_id}:*")
            if keys:
                await r.delete(*keys)
    finally:
        await r.aclose()


# ─── MongoDB (persistencia permanente) ───────────────────────────────────────

async def persist_turn(
    platform_id: str,
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
    user_name: str | None = None,
    org_id: str | None = None,
    collection_used: str | None = None,
    sources: list[dict] | None = None,
) -> None:
    """Persiste un turno de conversación en MongoDB."""
    from app.models.conversation import ChatTurn
    try:
        turn = ChatTurn(
            platform_id=platform_id,
            user_id=user_id,
            user_name=user_name,
            org_id=org_id,
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            collection_used=collection_used,
            sources=sources or [],
        )
        await turn.insert()
    except Exception as e:
        print(f"[history] mongo persist error: {e}")


async def get_user_history(
    platform_id: str,
    user_id: str,
    limit: int = 50,
) -> list[dict]:
    """Recupera el historial completo de un usuario desde MongoDB."""
    from app.models.conversation import ChatTurn
    turns = await ChatTurn.find(
        ChatTurn.platform_id == platform_id,
        ChatTurn.user_id == user_id,
    ).sort(-ChatTurn.created_at).limit(limit).to_list()

    return [
        {
            "user": t.user_message,
            "assistant": t.assistant_message,
            "collection": t.collection_used,
            "created_at": t.created_at.isoformat(),
        }
        for t in reversed(turns)
    ]


async def get_platform_history(
    platform_id: str,
    limit: int = 200,
) -> list[dict]:
    """Recupera conversaciones recientes de toda la plataforma (para recomendaciones)."""
    from app.models.conversation import ChatTurn
    turns = await ChatTurn.find(
        ChatTurn.platform_id == platform_id,
    ).sort(-ChatTurn.created_at).limit(limit).to_list()

    return [
        {
            "user_id": t.user_id,
            "user_name": t.user_name,
            "org_id": t.org_id,
            "user": t.user_message,
            "assistant": t.assistant_message,
            "collection": t.collection_used,
            "created_at": t.created_at.isoformat(),
        }
        for t in turns
    ]
