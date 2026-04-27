"""
RAG pipeline por plataforma + organización.
Construye y mantiene un vector store por (platform_id, org_id).
Utiliza LangChain con ChromaDB o FAISS.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS

from app.core.config import get_settings
from app.db.schema_introspector import introspect_database, schema_to_text

settings = get_settings()


# ─── Namespace helpers ───────────────────────────────────────────────────────

def _namespace(platform_id: str, org_id: str | None = None) -> str:
    key = f"{platform_id}:{org_id or '__global__'}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _chroma_dir(platform_id: str, org_id: str | None = None) -> str:
    ns = _namespace(platform_id, org_id)
    return os.path.join(settings.chroma_persist_dir, ns)


# ─── Embeddings factory ──────────────────────────────────────────────────────

def get_embeddings() -> Embeddings:
    if settings.llm_provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key,
        )
    # openai / anthropic (anthropic no tiene embeddings propios)
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )


# ─── Document preparation ────────────────────────────────────────────────────

def _prepare_documents(
    raw_docs: list[dict[str, str]],
    extra_docs: list[dict[str, str]] | None = None,
) -> list[Document]:
    """
    Convierte dicts crudos en Document de LangChain y los splittea.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    all_docs: list[Document] = []

    for raw in raw_docs:
        text = raw.pop("text")
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_docs.append(Document(page_content=chunk, metadata={**raw}))

    for extra in (extra_docs or []):
        text = extra.pop("text", extra.get("content", ""))
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_docs.append(Document(
                page_content=chunk,
                metadata={k: v for k, v in extra.items() if k != "content"},
            ))

    return all_docs


# ─── Vector store factory ────────────────────────────────────────────────────

def _build_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    platform_id: str,
    org_id: str | None = None,
    force: bool = False,
) -> Chroma | FAISS:
    if settings.vector_store_type == "chroma":
        persist_dir = _chroma_dir(platform_id, org_id)
        os.makedirs(persist_dir, exist_ok=True)
        collection_name = _namespace(platform_id, org_id)

        if force and os.path.exists(persist_dir):
            import chromadb
            client = chromadb.PersistentClient(path=persist_dir)
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass

        # Si ya existe la colección y no es force, agregar incrementalmente
        existing = _load_vector_store(embeddings, platform_id, org_id)
        if existing and not force:
            existing.add_documents(documents)
            print(f"[rag] incremental add: {len(documents)} docs to existing index")
            return existing

        vs = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=collection_name,
        )
        return vs

    # FAISS — no soporta incremental nativo, siempre reconstruye
    vs = FAISS.from_documents(documents=documents, embedding=embeddings)
    faiss_dir = os.path.join("./data/faiss", _namespace(platform_id, org_id))
    os.makedirs(faiss_dir, exist_ok=True)
    vs.save_local(faiss_dir)
    return vs


def _load_vector_store(
    embeddings: Embeddings,
    platform_id: str,
    org_id: str | None = None,
) -> Chroma | FAISS | None:
    if settings.vector_store_type == "chroma":
        persist_dir = _chroma_dir(platform_id, org_id)
        if not os.path.exists(persist_dir):
            return None
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=_namespace(platform_id, org_id),
        )

    faiss_dir = os.path.join("./data/faiss", _namespace(platform_id, org_id))
    if not os.path.exists(faiss_dir):
        return None
    return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)


# ─── Public API ──────────────────────────────────────────────────────────────

class RAGIndexer:
    """Construye el índice RAG para una plataforma/organización."""

    def __init__(
        self,
        platform_id: str,
        org_id: str | None = None,
    ):
        self.platform_id = platform_id
        self.org_id = org_id
        self.embeddings = get_embeddings()
        self._stats: dict[str, Any] = {}

    async def build_from_platform(
        self,
        db_connections: list[dict],
        extra_docs: list[dict] | None = None,
        force_schema: bool = False,
    ) -> dict[str, Any]:
        """
        Introspecciona todas las DBs de la plataforma y construye el índice.
        db_connections: [{"uri": "...", "database": "...", "collections": [...]}]

        Si force_schema=True, re-introspecciona aunque exista cache en disco.
        """
        schema_chunks: list[dict[str, str]] = []
        collections_indexed = 0
        errors = []

        for conn in db_connections:
            try:
                db_info = introspect_database(
                    uri=conn["uri"],
                    database=conn["database"],
                    collections=conn.get("collections"),
                    force=force_schema,
                )
                chunks = schema_to_text(db_info)
                schema_chunks.extend(chunks)
                collections_indexed += len(db_info["collections"])
            except Exception as exc:
                errors.append({"database": conn.get("database"), "error": str(exc)})

        # Preparar extra docs (FAQs, manuales, etc.)
        extra_prepared = []
        for doc in (extra_docs or []):
            extra_prepared.append({
                "text": doc.get("content", ""),
                "doc_type": doc.get("doc_type", "knowledge"),
                "title": doc.get("title", ""),
                **doc.get("metadata", {}),
            })

        if not schema_chunks and not extra_prepared:
            return {
                "status": "error",
                "message": "No se pudo extraer información de ninguna base de datos.",
                "errors": errors,
            }

        documents = _prepare_documents(schema_chunks, extra_prepared)
        _build_vector_store(documents, self.embeddings, self.platform_id, self.org_id, force=force_schema)

        return {
            "status": "ready",
            "collections_indexed": collections_indexed,
            "documents_indexed": len(documents),
            "errors": errors,
        }


class RAGRetriever:
    """Recupera contexto relevante del índice para una consulta."""

    def __init__(
        self,
        platform_id: str,
        org_id: str | None = None,
    ):
        self.platform_id = platform_id
        self.org_id = org_id
        self.embeddings = get_embeddings()
        self._vs: Chroma | FAISS | None = None

    def _get_store(self) -> Chroma | FAISS:
        if self._vs is None:
            self._vs = _load_vector_store(self.embeddings, self.platform_id, self.org_id)
            # Fallback a índice global de la plataforma si no hay por org
            if self._vs is None and self.org_id:
                self._vs = _load_vector_store(self.embeddings, self.platform_id, None)
            if self._vs is None:
                raise ValueError(
                    f"No existe índice RAG para platform={self.platform_id} org={self.org_id}. "
                    "Ejecuta el endpoint POST /platforms/{id}/index primero."
                )
        return self._vs

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        k = top_k or settings.rag_top_k
        store = self._get_store()

        results = store.similarity_search_with_relevance_scores(query, k=k)
        return [
            doc for doc, score in results
            if score >= settings.rag_score_threshold
        ]

    def retrieve_as_context(self, query: str) -> tuple[str, list[dict]]:
        """
        Devuelve (context_text, sources) para inyectar en el prompt.
        Incluye metadata de jerarquía (curso → módulo → actividad) para que el LLM
        pueda distinguir entre cursos completos y actividades específicas.
        """
        docs = self.retrieve(query)
        if not docs:
            return "", []

        context_parts = []
        sources = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            doc_type = meta.get("doc_type", "")
            course_name = meta.get("name", "")

            # Encabezado de jerarquía según el tipo de documento
            if doc_type == "course_summary" and course_name:
                header = f"[Fuente {i}] CURSO: {course_name}"
            elif doc_type == "knowledge":
                header = f"[Fuente {i}] CONOCIMIENTO GENERAL"
            else:
                header = f"[Fuente {i}]"

            context_parts.append(f"{header}\n{doc.page_content}")
            sources.append({
                "index": i,
                "source": meta.get("source", ""),
                "doc_type": doc_type,
                "collection": meta.get("collection", ""),
                "name": course_name,
            })

        return "\n\n---\n\n".join(context_parts), sources
