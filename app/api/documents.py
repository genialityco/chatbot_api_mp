"""
Endpoints para gestión de documentos RAG.
Autenticados con la API key de plataforma (X-Platform-Id + X-API-Key).
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.core.auth import get_platform_context_any_org

router = APIRouter(prefix="/documents", tags=["Documents"])

_reindex_running: set[str] = set()


async def _run_reindex(uri: str, database: str, platform_id: str, org_id: str | None, key: str) -> None:
    from app.rag.content_indexer import reindex_documents
    try:
        result = await reindex_documents(uri, database, platform_id, org_id)
        print(f"[documents] reindex completado: {result}")
    except Exception as exc:
        print(f"[documents] reindex error: {exc}")
    finally:
        _reindex_running.discard(key)


@router.post("/reindex")
async def trigger_reindex(
    background_tasks: BackgroundTasks,
    ctx: dict = Depends(get_platform_context_any_org),
):
    """
    Re-indexa la colección 'documents' en el RAG de la plataforma.
    Llamar desde GenCampus justo después de guardar un documento nuevo.
    No re-procesa cursos ni llama al LLM — es rápido.
    """
    platform = ctx["platform"]
    org_id: str | None = ctx.get("org_id")

    if not platform.db_connections:
        raise HTTPException(400, "La plataforma no tiene conexiones de base de datos configuradas.")

    conn = platform.db_connections[0]
    key = f"{platform.platform_id}:{org_id or '__global__'}"

    if key in _reindex_running:
        return {"status": "already_running", "message": "Ya hay una re-indexación en curso."}

    _reindex_running.add(key)
    background_tasks.add_task(
        _run_reindex,
        conn.uri,
        conn.database,
        platform.platform_id,
        org_id,
        key,
    )

    return {
        "status": "pending",
        "message": "Re-indexación de documentos iniciada en segundo plano.",
        "platform_id": platform.platform_id,
        "org_id": org_id,
    }
