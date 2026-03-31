"""
Actualiza las db_connections y system_prompt de las plataformas existentes.

Uso:
    python -m scripts.update_collections
"""
from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from app.core.config import get_settings
from app.models.platform import Platform

settings = get_settings()

PLATFORM_CONFIG = {
    "acho": {
        "collections": settings.schema_allowed_collections,
        "system_prompt": (
            "Eres un asistente experto en la plataforma ACHO. "
            "Ayudas con información sobre eventos, asistentes, ponentes y certificaciones. "
            "Responde siempre en el mismo idioma que el usuario. "
            "Usa la información de contexto para dar respuestas precisas sobre los datos de ACHO."
        ),
    },
    "gencampus": {
        "collections": settings.gencampus_allowed_collections,
        "system_prompt": (
            "Eres un asistente experto en la plataforma GenCampus, una plataforma de aprendizaje en línea. "
            "Ayudas a los usuarios con información sobre sus cursos, actividades, módulos, quizzes y progreso académico. "
            "Responde siempre en el mismo idioma que el usuario. "
            "Usa la información de contexto para dar respuestas precisas sobre el contenido educativo de GenCampus."
        ),
    },
    "genlive": {
        "collections": settings.schema_allowed_collections,
        "system_prompt": (
            "Eres un asistente experto en la plataforma GenLive. "
            "Ayudas con información sobre eventos en vivo, transmisiones y participantes. "
            "Responde siempre en el mismo idioma que el usuario. "
            "Usa la información de contexto para dar respuestas precisas sobre eventos en vivo."
        ),
    },
}


async def update():
    client = AsyncIOMotorClient(settings.meta_mongodb_uri)
    await init_beanie(database=client[settings.meta_mongodb_db], document_models=[Platform])

    platforms = await Platform.find_all().to_list()

    for platform in platforms:
        cfg = PLATFORM_CONFIG.get(platform.platform_id)
        if not cfg:
            print(f"[SKIP] {platform.platform_id} — sin config definida")
            continue

        for conn in platform.db_connections:
            conn.collections = cfg["collections"]
        platform.system_prompt = cfg["system_prompt"]
        await platform.save()
        print(f"[OK] {platform.platform_id} → {len(cfg['collections'])} colecciones")

    client.close()


if __name__ == "__main__":
    asyncio.run(update())
