from datetime import datetime
from typing import Any
from pydantic import Field
from beanie import Document


class ChatTurn(Document):
    """
    Turno de conversación persistido en MongoDB.
    Permite reconstruir historial y generar recomendaciones.
    """
    platform_id: str
    user_id: str
    user_name: str | None = None
    org_id: str | None = None
    session_id: str

    user_message: str
    assistant_message: str

    # Contexto usado para responder (colección consultada, fuentes)
    collection_used: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "chat_history"
        indexes = [
            [("platform_id", 1), ("user_id", 1), ("created_at", -1)],
            [("platform_id", 1), ("created_at", -1)],
        ]
