from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field
from beanie import Document, Indexed


# ─── Stored in Meta MongoDB ──────────────────────────────────────────────────

class DBConnection(BaseModel):
    """Conexión a una base de datos MongoDB de una plataforma."""
    uri: str
    database: str
    collections: list[str] | None = None  # None = todas las colecciones


class RAGDocument(BaseModel):
    """Documento extra de conocimiento (FAQs, manuales, políticas, etc.)"""
    title: str
    content: str
    doc_type: str = "knowledge"      # schema | knowledge | faq | manual
    metadata: dict[str, Any] = Field(default_factory=dict)


class Organization(BaseModel):
    org_id: str
    name: str
    extra_docs: list[RAGDocument] = Field(default_factory=list)
    system_prompt_override: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Platform(Document):
    """Plataforma registrada en el sistema."""
    platform_id: Indexed(str, unique=True)  # type: ignore
    name: str
    api_key_hash: str                        # hash de la API key
    db_connections: list[DBConnection] = Field(default_factory=list)
    organizations: list[Organization] = Field(default_factory=list)
    system_prompt: str = (
        "Eres un asistente experto en la plataforma {platform_name}. "
        "Responde siempre en el mismo idioma que el usuario. "
        "Usa la información de contexto para dar respuestas precisas."
    )
    rag_index_id: str | None = None          # ID del índice vectorial
    rag_indexed_at: datetime | None = None
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "platforms"

    def get_org(self, org_id: str) -> Organization | None:
        return next((o for o in self.organizations if o.org_id == org_id), None)

    def get_system_prompt(self, org_id: str | None = None) -> str:
        if org_id:
            org = self.get_org(org_id)
            if org and org.system_prompt_override:
                return org.system_prompt_override
        try:
            return self.system_prompt.format(platform_name=self.name)
        except KeyError:
            # El prompt tiene placeholders que no son platform_name (ej: gencampus URLs)
            return self.system_prompt


# ─── Request / Response schemas ──────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # user | assistant
    content: str


class ChatRequest(BaseModel):
    message: str
    user_id: str                              # ID del usuario en la plataforma
    user_name: str | None = None              # Nombre del usuario (opcional)
    org_id: str | None = None                 # ID de organización (opcional)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    platform_id: str
    org_id: str | None = None


class IndexStatusResponse(BaseModel):
    platform_id: str
    status: str       # pending | indexing | ready | error
    indexed_at: datetime | None = None
    collections_indexed: int = 0
    documents_indexed: int = 0
    message: str = ""
