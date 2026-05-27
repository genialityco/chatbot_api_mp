from datetime import datetime
from typing import Any
from pydantic import Field
from beanie import Document, Indexed


class FAQItem(Document):
    """
    FAQ aprendida automáticamente por clustering semántico de preguntas frecuentes.
    status=candidate hasta que hit_count >= FAQ_MIN_HITS, luego status=confirmed.
    """
    platform_id: str
    org_id: str | None = None

    question_canonical: str          # Pregunta representativa del cluster
    question_variants: list[str] = Field(default_factory=list)  # Variantes vistas
    answer: str                      # Última respuesta del LLM para este cluster

    hit_count: int = 1               # Veces que se ha visto una pregunta similar
    status: str = "candidate"        # candidate | confirmed

    chroma_cand_id: str | None = None  # ID del doc en ChromaDB faq_cand

    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: datetime | None = None

    class Settings:
        name = "faq_items"
        indexes = [
            [("platform_id", 1), ("org_id", 1), ("status", 1)],
            [("platform_id", 1), ("org_id", 1), ("hit_count", -1)],
        ]
