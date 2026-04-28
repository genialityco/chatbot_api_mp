"""
Servicio de recomendaciones por WhatsApp.
Envía mensajes de plantilla con cursos recomendados basados en historial y cursos inscritos.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
from bson import ObjectId
from pymongo import MongoClient

from app.core.config import get_settings
from app.models.platform import Platform
from app.services.recommendation_service import RecommendationService, _extract_keywords

settings = get_settings()

_WA_API_URL = "https://graph.facebook.com/v18.0/{phone_id}/messages"
_TEMPLATE_NAME = "rec_cursos_gencampus"
_TEMPLATE_LANG = "es"

# Número de destino hardcodeado para pruebas
_TEST_PHONE = "3133057451"


def _get_enrolled_event_ids(conn_dict: dict, user_id: str) -> set[str]:
    """Retorna los IDs de eventos en los que el usuario ya está inscrito."""
    try:
        client = MongoClient(
            conn_dict["uri"],
            serverSelectionTimeoutMS=8000,
            readPreference="secondaryPreferred",
        )
        db = client[conn_dict["database"]]
        # courseattendees.user_id es string
        docs = list(db["courseattendees"].find(
            {"user_id": user_id},
            {"event_id": 1}
        ))
        client.close()
        return {str(d["event_id"]) for d in docs if d.get("event_id")}
    except Exception as e:
        print(f"[wa_rec] error fetching enrolled ids: {e}")
        return set()


def _get_org_id_for_user(conn_dict: dict, user_id: str) -> tuple[str | None, str | None]:
    """Busca el org_id y el nombre de la organización del usuario."""
    try:
        client = MongoClient(
            conn_dict["uri"],
            serverSelectionTimeoutMS=8000,
            readPreference="secondaryPreferred",
        )
        db = client[conn_dict["database"]]
        try:
            query_id = ObjectId(user_id)
        except Exception:
            query_id = user_id
        doc = db["organizationusers"].find_one(
            {"user_id": query_id},
            {"organization_id": 1}
        )
        if not doc or not doc.get("organization_id"):
            client.close()
            return None, None

        org_id = str(doc["organization_id"])

        # Buscar nombre de la organización
        org_doc = db["organizations"].find_one(
            {"_id": doc["organization_id"]},
            {"name": 1}
        )
        org_name = org_doc.get("name", "GenCampus") if org_doc else "GenCampus"
        client.close()
        return org_id, org_name
    except Exception as e:
        print(f"[wa_rec] error fetching org_id: {e}")
        return None, None


async def _send_template(phone: str, user_name: str, org_name: str, course_urls: list[str]) -> None:
    """Envía el template rec_cursos_gencampus.
    Header: {{1}} = nombre usuario
    Body:   {{1}} = nombre organización, {{2}} {{3}} {{4}} = URLs de cursos
    """
    urls = (course_urls + [""] * 3)[:3]

    url = _WA_API_URL.format(phone_id=settings.whatsapp_phone_id)
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "template",
        "template": {
            "name": _TEMPLATE_NAME,
            "language": {"code": _TEMPLATE_LANG},
            "components": [
                {
                    "type": "header",
                    "parameters": [
                        {"type": "text", "text": user_name},  # {{1}} header: nombre usuario
                    ],
                },
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": org_name},   # {{1}} body: nombre organización
                        {"type": "text", "text": urls[0]},    # {{2}}
                        {"type": "text", "text": urls[1]},    # {{3}}
                        {"type": "text", "text": urls[2]},    # {{4}}
                    ],
                },
            ],
        },
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            print(f"[wa_rec] template send error {resp.status_code}: {resp.text}")
        else:
            print(f"[wa_rec] template sent to {phone}: {resp.status_code}")


async def send_wa_recommendations(platform: Platform, user_id: str, user_name: str) -> dict[str, Any]:
    """
    Genera recomendaciones para el usuario y envía el template de WA.
    Excluye cursos en los que el usuario ya está inscrito.
    """
    conn_dict = platform.db_connections[0].model_dump() if platform.db_connections else None
    if not conn_dict:
        return {"status": "error", "message": "No hay conexión de base de datos configurada."}

    # 1. Obtener cursos ya inscritos para excluirlos
    enrolled_ids = await asyncio.to_thread(_get_enrolled_event_ids, conn_dict, user_id)
    print(f"[wa_rec] enrolled_ids={enrolled_ids}")

    # 2. Obtener org_id y nombre de la organización
    org_id, org_name = await asyncio.to_thread(_get_org_id_for_user, conn_dict, user_id)
    print(f"[wa_rec] org_id={org_id} org_name={org_name}")

    # 3. Generar recomendaciones usando el servicio existente
    rec_service = RecommendationService(platform=platform, org_id=org_id)
    recommendations, keywords = await rec_service.recommend_for_user(user_id, limit=10)

    # 4. Filtrar los ya inscritos y tomar las 3 mejores URLs
    course_urls: list[str] = []
    for rec in recommendations:
        if rec.collection != "events":
            continue
        if rec.item_id in enrolled_ids:
            continue
        if rec.url:
            course_urls.append(rec.url)
        if len(course_urls) >= 3:
            break

    if not course_urls:
        return {"status": "no_recommendations", "message": "No se encontraron recomendaciones nuevas para el usuario."}

    # 5. Enviar template
    org_name = "GenCampus"
    await _send_template(_TEST_PHONE, user_name or "estudiante", org_name or "GenCampus", course_urls)

    return {
        "status": "sent",
        "phone": _TEST_PHONE,
        "user_id": user_id,
        "keywords": keywords,
        "course_urls": course_urls,
    }
