"""
Script de seed para registrar las plataformas iniciales en el sistema.

Uso:
    python -m scripts.seed_platforms

Las API keys generadas se muestran UNA SOLA VEZ. Guárdalas en un lugar seguro.
"""
from __future__ import annotations

import asyncio
import hashlib
import secrets
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from app.core.config import get_settings
from app.models.platform import Platform, DBConnection

settings = get_settings()

# ─── Definición de plataformas ───────────────────────────────────────────────

PLATFORMS = [
    {
        "platform_id": "acho",
        "name": "ACHO",
        "system_prompt": (
            "Eres un asistente experto en la plataforma ACHO. "
            "Responde siempre en el mismo idioma que el usuario. "
            "Usa la información de contexto para dar respuestas precisas sobre los datos de ACHO."
        ),
        "db_connections": [
            {
                "uri": settings.acho_mongo_uri,
                "database": settings.acho_mongo_db,
                "collections": settings.schema_allowed_collections,
            }
        ],
    },
    {
        "platform_id": "gencampus",
        "name": "GenCampus",
        "system_prompt": (
            "Eres un asistente experto en la plataforma GenCampus, una plataforma de aprendizaje en línea. "
            "Ayudas a los usuarios con información sobre sus cursos, actividades, módulos, quizzes y progreso académico. "
            "Responde siempre en el mismo idioma que el usuario. "
            "Usa la información de contexto para dar respuestas precisas sobre el contenido educativo de GenCampus."
        ),
        "db_connections": [
            {
                "uri": settings.gencampus_mongo_uri,
                "database": settings.gencampus_mongo_db,
                "collections": settings.gencampus_allowed_collections,
            }
        ],
    },
    {
        "platform_id": "genlive",
        "name": "GenLive",
        "system_prompt": (
            "Eres un asistente experto en la plataforma GenLive. "
            "Responde siempre en el mismo idioma que el usuario. "
            "Usa la información de contexto para dar respuestas precisas sobre eventos en vivo."
        ),
        "db_connections": [
            {
                "uri": settings.genlive_mongo_uri,
                "database": settings.genlive_mongo_db,
                "collections": settings.schema_allowed_collections,
            }
        ],
    },
]


# ─── Seed ────────────────────────────────────────────────────────────────────

async def seed():
    client = AsyncIOMotorClient(settings.meta_mongodb_uri)
    await init_beanie(
        database=client[settings.meta_mongodb_db],
        document_models=[Platform],
    )

    print(f"\n{'='*55}")
    print(f"  Seed de plataformas → {settings.meta_mongodb_db}")
    print(f"{'='*55}\n")

    generated_keys: list[dict] = []

    for p_data in PLATFORMS:
        existing = await Platform.find_one(Platform.platform_id == p_data["platform_id"])

        if existing:
            print(f"[SKIP]  '{p_data['platform_id']}' ya existe, omitiendo.")
            continue

        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        platform = Platform(
            platform_id=p_data["platform_id"],
            name=p_data["name"],
            api_key_hash=key_hash,
            system_prompt=p_data["system_prompt"],
            db_connections=[DBConnection(**c) for c in p_data["db_connections"]],
        )
        await platform.insert()

        generated_keys.append({
            "platform_id": p_data["platform_id"],
            "name": p_data["name"],
            "api_key": raw_key,
        })
        print(f"[OK]    '{p_data['platform_id']}' creada.")

    if generated_keys:
        print(f"\n{'='*55}")
        print("  ⚠️  API KEYS GENERADAS — guárdalas, no se volverán a mostrar")
        print(f"{'='*55}")
        for entry in generated_keys:
            print(f"\n  Plataforma : {entry['name']} ({entry['platform_id']})")
            print(f"  API Key    : {entry['api_key']}")
        print(f"\n{'='*55}\n")
    else:
        print("\nNo se crearon plataformas nuevas.\n")

    client.close()


if __name__ == "__main__":
    asyncio.run(seed())
