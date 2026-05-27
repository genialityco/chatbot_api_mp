from datetime import datetime
from pydantic import Field
from beanie import Document


class UnansweredQuestion(Document):
    """
    Pregunta que el chatbot no pudo responder con datos concretos.
    Se agrupa semánticamente igual que FAQItem pero para el caso negativo.
    reason indica por qué no se pudo responder.
    """
    platform_id: str
    org_id: str | None = None

    question_canonical: str
    question_variants: list[str] = Field(default_factory=list)

    hit_count: int = 1
    intent: str = "platform_query"   # intent clasificado al momento de la pregunta
    reason: str = "no_data"          # no_rag_data | no_db_data | no_db_collection | no_user_data

    chroma_id: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_asked: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "unanswered_questions"
        indexes = [
            [("platform_id", 1), ("org_id", 1), ("hit_count", -1)],
            [("platform_id", 1), ("org_id", 1), ("reason", 1)],
        ]
