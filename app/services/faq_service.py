"""
Servicio de auto-aprendizaje de FAQs.

Flujo post-chat (background, no bloquea):
  1. Embebe la pregunta del usuario.
  2. Busca en ChromaDB faq_cand si hay una pregunta similar (threshold 0.72).
     - MATCH  → incrementa hit_count en MongoDB; si hit_count >= FAQ_MIN_HITS promueve a confirmed.
     - NO MATCH → crea nuevo FAQItem(candidate) y lo indexa en faq_cand.

Flujo pre-chat (pre-check antes del LLM):
  1. Busca en ChromaDB faq_conf (solo confirmadas, threshold 0.82).
     - MATCH  → devuelve la respuesta cacheada (sin llamar al LLM).
     - NO MATCH → None (sigue el pipeline normal).
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
from app.models.faq import FAQItem
from app.rag.pipeline import get_embeddings

settings = get_settings()

FAQ_MIN_HITS: int = 3


# ─── Detector de respuestas socráticas ───────────────────────────────────────

def _is_socratic_response(answer: str) -> bool:
    """
    Devuelve True si la respuesta es una pregunta de clarificación socrática
    en lugar de una respuesta real con contenido. No debe aprenderse como FAQ.
    """
    stripped = answer.strip()
    # Respuesta muy corta que termina en pregunta
    if len(stripped) < 300 and stripped.endswith("?"):
        return True
    # Frases típicas del agente socrático
    _socratic_hints = [
        "¿podrías indicarme",
        "¿podrías decirme",
        "para poder orientarte",
        "para orientarte mejor",
        "¿en qué área",
        "¿qué tema específico",
        "¿sobre qué tema",
        "¿me podrías decir",
        "¿me podrías indicar",
    ]
    lower = stripped.lower()
    return any(hint in lower for hint in _socratic_hints)
# Similitud mínima para agrupar una pregunta con una candidata existente.
# Permisivo para capturar variantes semánticas del mismo tema.
FAQ_CAND_THRESHOLD: float = 0.72
# Similitud mínima para servir una respuesta desde el cache de confirmadas.
# Más estricto para evitar falsos positivos al responder sin LLM.
FAQ_CONF_THRESHOLD: float = 0.82


# ─── Namespace / directorio helpers ──────────────────────────────────────────

def _ns(platform_id: str, org_id: str | None) -> str:
    key = f"{platform_id}:{org_id or '__global__'}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _cand_dir(platform_id: str, org_id: str | None) -> str:
    return os.path.join(settings.chroma_persist_dir, "faq", f"cand_{_ns(platform_id, org_id)}")


def _conf_dir(platform_id: str, org_id: str | None) -> str:
    return os.path.join(settings.chroma_persist_dir, "faq", f"conf_{_ns(platform_id, org_id)}")


def _cand_name(platform_id: str, org_id: str | None) -> str:
    return f"faq_cand_{_ns(platform_id, org_id)}"


def _conf_name(platform_id: str, org_id: str | None) -> str:
    return f"faq_conf_{_ns(platform_id, org_id)}"


# ─── ChromaDB store helpers ───────────────────────────────────────────────────

def _open_store(persist_dir: str, collection_name: str) -> Chroma:
    os.makedirs(persist_dir, exist_ok=True)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
        collection_name=collection_name,
        # Cosine distance da scores entre 0-1 predecibles para similitud semántica.
        # L2 (default) produce scores bajos/negativos con embeddings de alta dimensión.
        collection_metadata={"hnsw:space": "cosine"},
    )


def _add_to_store(persist_dir: str, collection_name: str, text: str, doc_id: str, metadata: dict) -> None:
    store = _open_store(persist_dir, collection_name)
    store.add_documents(
        [Document(page_content=text, metadata=metadata)],
        ids=[doc_id],
    )


def _search_store(
    persist_dir: str,
    collection_name: str,
    query: str,
    threshold: float,
) -> tuple[Document, float, str] | None:
    """Devuelve (doc, score, doc_id) del mejor match o None."""
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
    print(f"[faq] similarity score={score:.4f} threshold={threshold} match={'YES' if score >= threshold else 'NO'} text='{doc.page_content[:50]}'")
    if score < threshold:
        return None
    doc_id = doc.metadata.get("faq_id", "")
    return doc, score, doc_id


# ─── FAQService ───────────────────────────────────────────────────────────────

class FAQService:

    async def check_faq(
        self,
        question: str,
        platform_id: str,
        org_id: str | None,
    ) -> str | None:
        """
        Pre-check: busca en el índice de confirmadas.
        Devuelve la respuesta cacheada si hay match, o None si debe seguir el pipeline.
        """
        result = await asyncio.to_thread(
            _search_store,
            _conf_dir(platform_id, org_id),
            _conf_name(platform_id, org_id),
            question,
            FAQ_CONF_THRESHOLD,
        )
        if result is None:
            return None

        _, score, faq_id = result
        print(f"[faq] hit confirmed FAQ id={faq_id} score={score:.3f}")

        # Actualizar hit_count y last_used en background
        asyncio.create_task(self._update_usage(faq_id))

        item = await FAQItem.find_one({"chroma_cand_id": faq_id})
        if item:
            return item.answer
        return None

    async def register_qa(
        self,
        platform_id: str,
        org_id: str | None,
        question: str,
        answer: str,
    ) -> None:
        """
        Post-chat: registra el par pregunta/respuesta para aprendizaje.
        Se llama desde chat_service como asyncio.create_task (no bloquea la respuesta).
        Ignora respuestas socráticas (clarificaciones sin contenido real).
        """
        if _is_socratic_response(answer):
            print(f"[faq] skip — respuesta socrática detectada")
            return
        try:
            await self._learn(platform_id, org_id, question, answer)
        except Exception as exc:
            print(f"[faq] error en register_qa: {exc}")

    # ─── Lógica de aprendizaje ────────────────────────────────────────────────

    async def _learn(
        self,
        platform_id: str,
        org_id: str | None,
        question: str,
        answer: str,
    ) -> None:
        result = await asyncio.to_thread(
            _search_store,
            _cand_dir(platform_id, org_id),
            _cand_name(platform_id, org_id),
            question,
            FAQ_CAND_THRESHOLD,
        )

        if result is not None:
            _, score, faq_id = result
            print(f"[faq] match candidata id={faq_id} score={score:.3f}")
            await self._increment_and_maybe_promote(faq_id, question, answer, platform_id, org_id)
        else:
            print(f"[faq] nueva candidata: '{question[:60]}'")
            await self._create_candidate(platform_id, org_id, question, answer)

    async def _create_candidate(
        self,
        platform_id: str,
        org_id: str | None,
        question: str,
        answer: str,
    ) -> None:
        faq_id = str(uuid.uuid4())
        item = FAQItem(
            platform_id=platform_id,
            org_id=org_id,
            question_canonical=question,
            question_variants=[question],
            answer=answer,
            hit_count=1,
            status="candidate",
            chroma_cand_id=faq_id,
            created_at=datetime.utcnow(),
        )
        await item.insert()

        await asyncio.to_thread(
            _add_to_store,
            _cand_dir(platform_id, org_id),
            _cand_name(platform_id, org_id),
            question,
            faq_id,
            {"faq_id": faq_id, "platform_id": platform_id, "org_id": org_id or ""},
        )

    async def _increment_and_maybe_promote(
        self,
        faq_id: str,
        question: str,
        answer: str,
        platform_id: str,
        org_id: str | None,
    ) -> None:
        item = await FAQItem.find_one({"chroma_cand_id": faq_id})
        if item is None:
            return

        item.hit_count += 1
        item.answer = answer  # mantener la respuesta más reciente
        if question not in item.question_variants:
            item.question_variants.append(question)

        if item.status == "candidate" and item.hit_count >= FAQ_MIN_HITS:
            item.status = "confirmed"
            await item.save()
            print(f"[faq] promovida a confirmed id={faq_id} hits={item.hit_count}")
            await self._index_confirmed(item)
        else:
            await item.save()

    async def _index_confirmed(self, item: FAQItem) -> None:
        """Agrega la FAQ confirmada al índice faq_conf."""
        await asyncio.to_thread(
            _add_to_store,
            _conf_dir(item.platform_id, item.org_id),
            _conf_name(item.platform_id, item.org_id),
            item.question_canonical,
            item.chroma_cand_id,
            {
                "faq_id": item.chroma_cand_id,
                "platform_id": item.platform_id,
                "org_id": item.org_id or "",
            },
        )

    async def _update_usage(self, faq_id: str) -> None:
        item = await FAQItem.find_one({"chroma_cand_id": faq_id})
        if item:
            item.hit_count += 1
            item.last_used = datetime.utcnow()
            await item.save()


# Singleton para importar desde chat_service
faq_service = FAQService()
