from fastapi import APIRouter, BackgroundTasks, Query, HTTPException, Request, Depends

from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

from app.core.auth import get_platform_context
from app.core.config import get_settings
from app.services.whatsapp_service import markdown_to_whatsapp, send_text_message

router = APIRouter(prefix="/webhook", tags=["webhook"])

settings = get_settings()

# Plataforma hardcodeada para este prototipo de WhatsApp
_WA_PLATFORM_ID = "gencampus"


@router.get("")
async def verify_webhook(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge"),
):
    if mode == "subscribe" and token == settings.meta_verify_token:
        return int(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    """Recibe mensajes de WhatsApp y procesa la respuesta en background."""
    body = await request.json()
    print(f"[webhook] POST recibido: {body}")

    # Extraer mensaje del payload de Meta
    try:
        entry = body["entry"][0]
        change = entry["changes"][0]["value"]
        messages = change.get("messages")
        if not messages:
            return {"status": "ok"}
        msg = messages[0]
        sender = msg["from"]
        text = msg.get("text", {}).get("body", "")
        if not text:
            return {"status": "ok"}
        # ID del número de WhatsApp Business que recibió el mensaje
        phone_number_id = change.get("metadata", {}).get("phone_number_id")
    except (KeyError, IndexError):
        return {"status": "ok"}

    background_tasks.add_task(_process_and_reply, sender, text, phone_number_id)
    return {"status": "ok"}


async def _process_and_reply(phone: str, message: str, phone_number_id: str | None = None) -> None:
    """Llama al ChatService y envía la respuesta al usuario de WhatsApp."""
    from app.models.platform import Platform
    from app.services.chat_service import ChatService
    from app.services.socratic_agent import SocraticAgent

    try:
        platform = await Platform.find_one(Platform.platform_id == _WA_PLATFORM_ID)
        if not platform:
            print(f"[webhook] platform '{_WA_PLATFORM_ID}' not found")
            return

        # ─── Resolver organización por número de WhatsApp Business ────
        org_id: str | None = None
        if phone_number_id:
            org = platform.get_org_by_whatsapp_phone(phone_number_id)
            if org:
                org_id = org.org_id
                print(f"[webhook] org resuelto por phone_number_id={phone_number_id} → org_id={org_id}")
            else:
                print(f"[webhook] phone_number_id={phone_number_id} sin org asignada, usando contexto global")
        # ──────────────────────────────────────────────────────────────

        # ─── Búsqueda de usuario por teléfono ─────────────────────────
        possible_phones = [phone, f"+{phone}"]
        if len(phone) >= 10:
            last_10 = phone[-10:]
            possible_phones.append(last_10)
            try:
                possible_phones.append(float(last_10))
            except ValueError:
                pass

        user_id_str = phone
        user_name = None

        gencampus_conn = next((c for c in platform.db_connections if c.database == settings.gencampus_mongo_db), None)

        if gencampus_conn:
            client = AsyncIOMotorClient(gencampus_conn.uri)
            db = client[gencampus_conn.database]

            # Buscar usuario dentro de la org si ya la conocemos, o en toda la BD
            query: dict = {'properties.phone': {'$in': possible_phones}}
            if org_id:
                query['organization_id'] = org_id

            org_user = await db.organizationusers.find_one(query)

            # Si no encontró con org_id, intentar sin filtro de org (fallback)
            if not org_user and org_id:
                org_user = await db.organizationusers.find_one({'properties.phone': {'$in': possible_phones}})

            if org_user and 'user_id' in org_user:
                user_id_str = str(org_user['user_id'])
                if 'properties' in org_user:
                    user_name = f"{org_user['properties'].get('names', '')} {org_user['properties'].get('lastNames', '')}".strip()
                # Si org_id no se resolvió por phone_number_id, tomarlo del documento del usuario
                if not org_id and org_user.get('organization_id'):
                    org_id = str(org_user['organization_id'])
                    print(f"[webhook] org_id resolved from organizationusers: {org_id}")
            else:
                await send_text_message(phone, "Lo siento, no estás autorizado para usar este canal ya que tu número no está registrado en la plataforma.")
                client.close()
                return

            client.close()
        # ──────────────────────────────────────────────────────────────

        system_prompt = platform.get_system_prompt()
        db_connections = [c.model_dump() for c in platform.db_connections]

        if getattr(platform, "socratic_mode", False):
            service = SocraticAgent(
                platform_id=_WA_PLATFORM_ID,
                org_id=None,
                base_prompt=system_prompt,
                db_connections=db_connections,
            )
        else:
            service = ChatService(
                platform_id=_WA_PLATFORM_ID,
                org_id=None,
                system_prompt=system_prompt,
                db_connections=db_connections,
            )

        result = await service.chat(
            message=message,
            user_id=user_id_str,
            user_name=user_name,
            org_id=org_id,
            session_id=session_id,
        )

        answer = markdown_to_whatsapp(result["answer"])
        await send_text_message(phone, answer)

    except Exception as exc:
        print(f"[webhook] error processing message from {phone}: {exc}")


# ─── Endpoint de recomendaciones por WA ──────────────────────────────────────

class WARecommendationRequest(BaseModel):
    user_id: str
    user_name: str | None = None


@router.post("/recommendations/wa")
async def send_wa_recommendation(
    body: WARecommendationRequest,
    ctx: dict = Depends(get_platform_context),
):
    """
    Genera recomendaciones para un usuario y las envía por WhatsApp
    usando el template rec_cursos_gencampus.
    """
    from app.services.wa_recommendation_service import send_wa_recommendations

    platform = ctx["platform"]
    result = await send_wa_recommendations(
        platform=platform,
        user_id=body.user_id,
        user_name=body.user_name,
    )

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return result
