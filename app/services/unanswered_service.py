"""
Servicio de registro de preguntas sin respuesta.

Flujo post-chat (background, no bloquea):
  1. Embebe la pregunta del usuario.
  2. Busca en ChromaDB unans si hay una pregunta similar (threshold 0.72).
     - MATCH  → incrementa hit_count en MongoDB.
     - NO MATCH → crea nuevo UnansweredQuestion y lo indexa.

Esto permite detectar qué temas se preguntan frecuentemente sin tener respuesta,
para priorizar qué documentar o indexar en RAG.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.models.unanswered import UnansweredQuestion
from app.rag.pipeline import get_embeddings

settings = get_settings()

UNANS_THRESHOLD: float = 0.72


def _ns(platform_id: str, org_id: str | None) -> str:
    key = f"{platform_id}:{org_id or '__global__'}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _unans_dir(platform_id: str, org_id: str | None) -> str:
    return os.path.join(settings.chroma_persist_dir, "unans", _ns(platform_id, org_id))


def _unans_name(platform_id: str, org_id: str | None) -> str:
    return f"unans_{_ns(platform_id, org_id)}"


def _open_store(persist_dir: str, collection_name: str) -> Chroma:
    os.makedirs(persist_dir, exist_ok=True)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _add_to_store(persist_dir: str, collection_name: str, text: str, doc_id: str, metadata: dict) -> None:
    store = _open_store(persist_dir, collection_name)
    store.add_documents(
        [Document(page_content=text, metadata=metadata)],
        ids=[doc_id],
    )


def _search_store(persist_dir: str, collection_name: str, query: str) -> tuple[Document, float, str] | None:
    if not os.path.exists(persist_dir):
        return None
    store = _open_store(persist_dir, collection_name)
    try:
        results = store.similarity_search_with_relevance_scores(query, k=1)
    except Exception:
        return None
    if not results:
        return None
    doc, score = results[0]
    print(f"[unans] similarity score={score:.4f} threshold={UNANS_THRESHOLD} match={'YES' if score >= UNANS_THRESHOLD else 'NO'} text='{doc.page_content[:50]}'")
    if score < UNANS_THRESHOLD:
        return None
    doc_id = doc.metadata.get("unans_id", "")
    return doc, score, doc_id


def _determine_reason(has_rag: bool, has_db_data: bool, is_personal: bool) -> str:
    """Determina la razón por la que no se pudo responder."""
    if is_personal and not has_db_data:
        return "no_user_data"
    if not has_rag and not has_db_data:
        return "no_rag_data"
    if has_rag and not has_db_data:
        return "no_db_data"
    return "no_data"


class UnansweredService:

    async def register(
        self,
        platform_id: str,
        org_id: str | None,
        question: str,
        intent: str,
        has_rag: bool,
        has_db_data: bool,
        is_personal: bool,
    ) -> None:
        """
        Registra una pregunta que no pudo ser respondida con datos concretos.
        Se llama desde chat_service como asyncio.create_task (no bloquea).
        """
        try:
            reason = _determine_reason(has_rag, has_db_data, is_personal)
            await self._learn(platform_id, org_id, question, intent, reason)
        except Exception as exc:
            print(f"[unans] error en register: {exc}")

    async def _learn(
        self,
        platform_id: str,
        org_id: str | None,
        question: str,
        intent: str,
        reason: str,
    ) -> None:
        result = await asyncio.to_thread(
            _search_store,
            _unans_dir(platform_id, org_id),
            _unans_name(platform_id, org_id),
            question,
        )

        if result is not None:
            _, score, unans_id = result
            print(f"[unans] match existente id={unans_id} score={score:.3f}")
            await self._increment(unans_id, question)
        else:
            print(f"[unans] nueva pregunta sin respuesta: '{question[:60]}'")
            await self._create(platform_id, org_id, question, intent, reason)

    async def _create(
        self,
        platform_id: str,
        org_id: str | None,
        question: str,
        intent: str,
        reason: str,
    ) -> None:
        unans_id = str(uuid.uuid4())
        item = UnansweredQuestion(
            platform_id=platform_id,
            org_id=org_id,
            question_canonical=question,
            question_variants=[question],
            hit_count=1,
            intent=intent,
            reason=reason,
            chroma_id=unans_id,
            created_at=datetime.utcnow(),
            last_asked=datetime.utcnow(),
        )
        await item.insert()

        await asyncio.to_thread(
            _add_to_store,
            _unans_dir(platform_id, org_id),
            _unans_name(platform_id, org_id),
            question,
            unans_id,
            {"unans_id": unans_id, "platform_id": platform_id, "org_id": org_id or ""},
        )

    async def _increment(self, unans_id: str, question: str) -> None:
        item = await UnansweredQuestion.find_one({"chroma_id": unans_id})
        if item is None:
            return
        item.hit_count += 1
        item.last_asked = datetime.utcnow()
        if question not in item.question_variants:
            item.question_variants.append(question)
        await item.save()
        print(f"[unans] hit_count={item.hit_count} para '{item.question_canonical[:60]}'")


unanswered_service = UnansweredService()
