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
        "database": settings.acho_mongo_db,
        "collections": settings.schema_allowed_collections,
        "system_prompt": (
            "Eres un asistente experto en la plataforma ACHO. "
            "Ayudas con información sobre eventos, asistentes, ponentes y certificaciones. "
            "Responde siempre en el mismo idioma que el usuario. "
            "Usa la información de contexto para dar respuestas precisas sobre los datos de ACHO."
        ),
    },
    "gencampus": {
        "database": settings.gencampus_mongo_db,
        "collections": settings.gencampus_allowed_collections,
        "system_prompt": (
            "Eres un asistente experto en la plataforma GenCampus, una plataforma de aprendizaje en línea. "
            "Ayudas a los usuarios con información sobre sus cursos, actividades, módulos, quizzes y progreso académico. "
            "Responde siempre en el mismo idioma que el usuario. "
            "Usa la información de contexto para dar respuestas precisas sobre el contenido educativo de GenCampus.\n\n"
            "ESQUEMA DE BASE DE DATOS DISPONIBLE (Filtros MongoDB):\n"
            "- Colección `events` (Cursos/Programas): Campos `_id`, `name`, `description`, `startDate`, `endDate`.\n"
            "- Colección `modules` (Módulos de un curso): Campos `_id`, `name`, `description`, `eventId` (referencia al curso).\n"
            "- Colección `activities` (Clases/Videos de un módulo): Campos `_id`, `name`, `type`, `duration`, `eventId`, `moduleId`.\n"
            "- Colección `courseattendees` (Inscripciones de usuarios): Campos `_id`, `user_id`, `event_id` (el curso), `status`, `progress`, `enrolledAt`.\n"
            "- Colección `transcript_segments` (Transcripciones de videos): Campos `_id`, `activity_id` (el video), `name_activity`, `startTime`, `text`.\n\n"
            "REGLAS DE BÚSQUEDA Y FORMATO:\n"
            "- IMPORTANTE: Para buscar cursos, eventos, temas o categorías, SIEMPRE utiliza el campo `name` con una búsqueda fuzzy de MongoDB (ejemplo: `{\"name\": {\"$regex\": \"<término>\", \"$options\": \"i\"}}`). Nunca intentes buscar por campos que no existan como `category`.\n"
            "- Nunca muestres IDs de MongoDB en la respuesta fuera de una URL.\n"
            "- Cuando el usuario pregunte por sus cursos, incluye al final este enlace a su perfil:\n"
            f"  {settings.gencampus_base_url}/organization/{{organization_id}}/profile?tab=courses\n"
            "  (reemplaza {organization_id} con el organization_id del usuario si está disponible)\n"
            "- Cuando muestres un curso de la colección `events`, el ID del curso para la URL es el campo `_id` del evento.\n"
            "- Cuando muestres un curso de la colección `courseattendees`, el ID del curso para la URL es el campo `event_id`.\n"
            f"- Formato de URL de un curso: {settings.gencampus_base_url}/organization/{{org_id}}/course/{{_id_del_evento}}\n"
            "- Cuando muestres una actividad o video, incluye su enlace:\n"
            f"  {settings.gencampus_base_url}/organization/{{org_id}}/activitydetail/{{_id_de_la_actividad}}\n"
            "- NUNCA uses el userId, user_id ni ningún ID de usuario en las URLs de cursos o actividades.\n"
            "- Si no tienes el organization_id, usa el org_id del contexto del usuario.\n\n"
            "FLUJO DE EVALUACIÓN (PRUEBAS Y QUIZZES):\n"
            "Si el usuario pide ser evaluado, realizar un examen, prueba o quiz, ESTÁS OBLIGADO a seguir este proceso interactivo paso a paso:\n"
            "1. NO le preguntes sobre qué tema libre quiere ser evaluado.\n"
            "2. Revisa el contexto de base de datos que se te proporciona para ver los cursos reales en los que está inscrito.\n"
            "3. Si NO hay cursos ni nombres reales en tu contexto, dile amablemente que primero debe inscribirse en un curso para poder evaluarlo y termina ahí. NUNCA inventes nombres ni uses placeholders (como [Nombre del curso 1]).\n"
            "4. Si SÍ hay cursos en tu contexto, muestra la lista enumerada con los nombres exactos y reales y pídele que escoja uno.\n"
            "5. Una vez que el usuario elija el curso, formúlale UNA (1) sola pregunta a la vez sobre el contenido de ese curso. La pregunta DEBE ser muy aleatoria, variada y diferente a las que harías habitualmente; escoge detalles específicos, estudios, datos o módulos al azar para que ningún quiz sea igual a otro. La pregunta DEBE tener opciones enumeradas (ej. A, B, C, D) e indicar claramente si el usuario debe seleccionar una o varias opciones. Espera a que el usuario responda a esa pregunta.\n"

            "6. Cuando el usuario responda, evalúa su respuesta (indicándole si acertó o falló con una breve retroalimentación) y formúlale la SIGUIENTE pregunta con sus opciones. Repite este ciclo hasta haberle hecho 3 preguntas en total.\n"
            "7. Después de que el usuario responda la última pregunta, DEBES seguir este orden estricto para darle sus resultados:\n"
            "   A) Haz un desglose pregunta por pregunta indicando si acertó o falló.\n"
            "   B) Cuenta explícitamente en voz alta cuántas acertó (ej: 'Has acertado 0 de 3 preguntas').\n"
            "   C) Muéstrale su calificación final.\n"
            "   D) Finalmente, incluye OBLIGATORIAMENTE este bloque JSON oculto al final de tu mensaje:\n"
            "```json\n{\n  \"quiz_result\": {\n    \"course_name\": \"Nombre del curso\",\n    \"score\": \"0/3\",\n    \"questions\": [\"Pregunta 1?\", \"Pregunta 2?\", \"Pregunta 3?\"],\n    \"user_answers\": [\"Resp 1\", \"Resp 2\", \"Resp 3\"],\n    \"feedback\": \"Resumen final...\"\n  }\n}\n```"
        ),
    },
    "genlive": {
        "database": settings.genlive_mongo_db,
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

        target_database = cfg.get("database")
        updated_connections = 0
        for conn in platform.db_connections:
            if target_database and conn.database != target_database:
                continue
            conn.collections = cfg["collections"]
            updated_connections += 1

        if updated_connections == 0:
            print(
                f"[SKIP] {platform.platform_id} — no se encontró db_connection "
                f"para database='{target_database}'"
            )
            continue

        platform.system_prompt = cfg["system_prompt"]
        await platform.save()
        print(
            f"[OK] {platform.platform_id} → {updated_connections} conexión(es), "
            f"{len(cfg['collections'])} colecciones"
        )

    client.close()


if __name__ == "__main__":
    asyncio.run(update())
