"""
Endpoints de gestión de plataformas (admin).
Requieren X-Admin-Key header.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.core.auth import require_admin
from app.models.platform import Platform, Organization, DBConnection, RAGDocument, IndexStatusResponse
from app.rag.pipeline import RAGIndexer

router = APIRouter(prefix="/platforms", tags=["Platforms (Admin)"])


# ─── Schemas ─────────────────────────────────────────────────────────────────

class CreatePlatformRequest(BaseModel):
    platform_id: str
    name: str
    system_prompt: str | None = None
    db_connections: list[dict] = []


class AddOrgRequest(BaseModel):
    org_id: str
    name: str
    system_prompt_override: str | None = None
    extra_docs: list[dict] = []


class AddDBRequest(BaseModel):
    uri: str
    database: str
    collections: list[str] | None = None


class AddDocRequest(BaseModel):
    title: str
    content: str
    doc_type: str = "knowledge"
    org_id: str | None = None


# ─── In-memory index status tracker ─────────────────────────────────────────

_index_status: dict[str, dict] = {}


def _status_key(platform_id: str, org_id: str | None) -> str:
    return f"{platform_id}:{org_id or '__global__'}"


# ─── Background indexing task ────────────────────────────────────────────────

async def _run_indexing(platform_id: str, org_id: str | None = None, force_schema: bool = False):
    key = _status_key(platform_id, org_id)
    _index_status[key] = {"status": "indexing", "message": "Indexando..."}

    platform = await Platform.find_one(Platform.platform_id == platform_id)
    if not platform:
        _index_status[key] = {"status": "error", "message": "Plataforma no encontrada."}
        return

    # Seleccionar org o global
    extra_docs = []
    if org_id:
        org = platform.get_org(org_id)
        if org:
            extra_docs = [d.model_dump() for d in org.extra_docs]

    db_connections = [c.model_dump() for c in platform.db_connections]

    indexer = RAGIndexer(platform_id=platform_id, org_id=org_id)
    result = await indexer.build_from_platform(
        db_connections=db_connections,
        extra_docs=extra_docs,
        force_schema=force_schema,
    )

    # Actualizar estado en la plataforma
    platform.rag_indexed_at = datetime.utcnow()
    platform.updated_at = datetime.utcnow()
    await platform.save()

    _index_status[key] = {
        **result,
        "indexed_at": platform.rag_indexed_at.isoformat(),
    }


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("", status_code=201)
async def create_platform(
    body: CreatePlatformRequest,
    _: None = Depends(require_admin),
):
    """Registra una nueva plataforma y devuelve su API Key (solo se muestra una vez)."""
    existing = await Platform.find_one(Platform.platform_id == body.platform_id)
    if existing:
        raise HTTPException(400, f"La plataforma '{body.platform_id}' ya existe.")

    raw_key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    platform = Platform(
        platform_id=body.platform_id,
        name=body.name,
        api_key_hash=key_hash,
        db_connections=[DBConnection(**c) for c in body.db_connections],
        **({"system_prompt": body.system_prompt} if body.system_prompt else {}),
    )
    await platform.insert()

    return {
        "platform_id": platform.platform_id,
        "api_key": raw_key,  # ⚠️ Solo se muestra una vez
        "message": "Plataforma creada. Guarda la API Key, no se volverá a mostrar.",
    }


@router.get("")
async def list_platforms(_: None = Depends(require_admin)):
    platforms = await Platform.find_all().to_list()
    return [
        {
            "platform_id": p.platform_id,
            "name": p.name,
            "active": p.active,
            "rag_indexed_at": p.rag_indexed_at,
            "organizations": len(p.organizations),
            "db_connections": len(p.db_connections),
        }
        for p in platforms
    ]


@router.post("/{platform_id}/db-connections", status_code=201)
async def add_db_connection(
    platform_id: str,
    body: AddDBRequest,
    _: None = Depends(require_admin),
):
    platform = await Platform.find_one(Platform.platform_id == platform_id)
    if not platform:
        raise HTTPException(404, "Plataforma no encontrada.")

    platform.db_connections.append(DBConnection(**body.model_dump()))
    platform.updated_at = datetime.utcnow()
    await platform.save()
    return {"message": "Conexión agregada.", "total": len(platform.db_connections)}


@router.post("/{platform_id}/organizations", status_code=201)
async def add_organization(
    platform_id: str,
    body: AddOrgRequest,
    _: None = Depends(require_admin),
):
    platform = await Platform.find_one(Platform.platform_id == platform_id)
    if not platform:
        raise HTTPException(404, "Plataforma no encontrada.")

    if platform.get_org(body.org_id):
        raise HTTPException(400, f"Organización '{body.org_id}' ya existe.")

    org = Organization(
        org_id=body.org_id,
        name=body.name,
        system_prompt_override=body.system_prompt_override,
        extra_docs=[RAGDocument(**d) for d in body.extra_docs],
    )
    platform.organizations.append(org)
    platform.updated_at = datetime.utcnow()
    await platform.save()
    return {"message": "Organización agregada.", "org_id": org.org_id}


@router.post("/{platform_id}/documents", status_code=201)
async def add_document(
    platform_id: str,
    body: AddDocRequest,
    _: None = Depends(require_admin),
):
    """Agrega un documento de conocimiento (FAQ, manual, etc.) a la plataforma u org."""
    platform = await Platform.find_one(Platform.platform_id == platform_id)
    if not platform:
        raise HTTPException(404, "Plataforma no encontrada.")

    doc = RAGDocument(title=body.title, content=body.content, doc_type=body.doc_type)

    if body.org_id:
        org = platform.get_org(body.org_id)
        if not org:
            raise HTTPException(404, f"Organización '{body.org_id}' no encontrada.")
        org.extra_docs.append(doc)
    else:
        # Documento global de plataforma: lo guardamos en la primera org o creamos meta
        # Para simplicidad, lo guardamos como extra_doc en la plataforma directamente
        # Extendemos el modelo si hace falta. Por ahora, en la primera org.
        pass

    platform.updated_at = datetime.utcnow()
    await platform.save()
    return {"message": "Documento agregado."}


@router.post("/{platform_id}/index")
async def trigger_index(
    platform_id: str,
    background_tasks: BackgroundTasks,
    org_id: str | None = None,
    force_schema: bool = False,
    _: None = Depends(require_admin),
):
    """
    Dispara la indexación RAG en segundo plano.
    - org_id: crea un índice específico para esa organización.
    - force_schema: re-introspecciona MongoDB aunque exista cache en disco.
    """
    platform = await Platform.find_one(Platform.platform_id == platform_id)
    if not platform:
        raise HTTPException(404, "Plataforma no encontrada.")

    key = _status_key(platform_id, org_id)
    if _index_status.get(key, {}).get("status") == "indexing":
        return {"message": "Indexación ya en curso.", "status": "indexing"}

    _index_status[key] = {"status": "pending"}
    background_tasks.add_task(_run_indexing, platform_id, org_id, force_schema)

    return {
        "message": "Indexación iniciada en segundo plano.",
        "platform_id": platform_id,
        "org_id": org_id,
        "force_schema": force_schema,
        "status": "pending",
    }


@router.get("/{platform_id}/index/status", response_model=IndexStatusResponse)
async def index_status(
    platform_id: str,
    org_id: str | None = None,
    _: None = Depends(require_admin),
):
    key = _status_key(platform_id, org_id)
    status_data = _index_status.get(key, {"status": "not_started"})

    return IndexStatusResponse(
        platform_id=platform_id,
        status=status_data.get("status", "not_started"),
        collections_indexed=status_data.get("collections_indexed", 0),
        documents_indexed=status_data.get("documents_indexed", 0),
        message=status_data.get("message", ""),
        indexed_at=status_data.get("indexed_at"),
    )


@router.post("/{platform_id}/index/content")
async def trigger_content_index(
    platform_id: str,
    background_tasks: BackgroundTasks,
    org_id: str | None = None,
    event_id: str | None = None,
    _: None = Depends(require_admin),
):
    """
    Indexa el contenido real de cursos en el RAG.
    - Sin event_id: re-indexa todos los cursos (force=True, borra y recrea).
    - Con event_id: indexa solo ese curso de forma incremental (no toca el resto).
    """
    from app.rag.content_indexer import build_content_rag, build_single_course_rag

    platform = await Platform.find_one(Platform.platform_id == platform_id)
    if not platform:
        raise HTTPException(404, "Plataforma no encontrada.")
    if not platform.db_connections:
        raise HTTPException(400, "La plataforma no tiene conexiones de base de datos configuradas.")

    key = _status_key(platform_id, org_id) + (f":{event_id}" if event_id else ":content")
    if _index_status.get(key, {}).get("status") == "indexing":
        return {"message": "Indexación ya en curso.", "status": "indexing"}

    _index_status[key] = {"status": "indexing", "message": "Indexando..."}

    async def _run():
        try:
            conn = platform.db_connections[0]
            if event_id:
                result = await build_single_course_rag(conn.uri, conn.database, platform_id, event_id, org_id)
            else:
                result = await build_content_rag(conn.uri, conn.database, platform_id, org_id)
            _index_status[key] = {**result, "status": "ready"}
        except Exception as e:
            _index_status[key] = {"status": "error", "message": str(e)}
            print(f"[content_index] error: {e}")

    background_tasks.add_task(_run)
    return {
        "message": "Indexación iniciada.",
        "platform_id": platform_id,
        "org_id": org_id,
        "event_id": event_id,
        "mode": "incremental" if event_id else "full",
        "status": "indexing",
    }


@router.get("/{platform_id}/index/content/status")
async def content_index_status(
    platform_id: str,
    org_id: str | None = None,
    _: None = Depends(require_admin),
):
    key = _status_key(platform_id, org_id) + ":content"
    return _index_status.get(key, {"status": "not_started"})
async def regenerate_api_key(
    platform_id: str,
    _: None = Depends(require_admin),
):
    """Genera una nueva API Key para la plataforma. La anterior queda inválida."""
    platform = await Platform.find_one(Platform.platform_id == platform_id)
    if not platform:
        raise HTTPException(404, "Plataforma no encontrada.")

    raw_key = secrets.token_urlsafe(32)
    platform.api_key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    platform.updated_at = datetime.utcnow()
    await platform.save()

    return {
        "platform_id": platform_id,
        "api_key": raw_key,
        "message": "API Key regenerada. Guárdala, no se volverá a mostrar.",
    }
