from fastapi import APIRouter, BackgroundTasks, Query, HTTPException, Request

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
            # Son notificaciones de estado (leído, entregado) — ignorar
            return {"status": "ok"}
        msg = messages[0]
        sender = msg["from"]
        text = msg.get("text", {}).get("body", "")
        if not text:
            return {"status": "ok"}
    except (KeyError, IndexError):
        # Payload inesperado — devolver 200 para que Meta no reintente
        return {"status": "ok"}

    background_tasks.add_task(_process_and_reply, sender, text)
    return {"status": "ok"}


async def _process_and_reply(phone: str, message: str) -> None:
    """Llama al ChatService y envía la respuesta al usuario de WhatsApp."""
    from app.models.platform import Platform
    from app.services.chat_service import ChatService

    try:
        platform = await Platform.find_one(Platform.platform_id == _WA_PLATFORM_ID)
        if not platform:
            print(f"[webhook] platform '{_WA_PLATFORM_ID}' not found")
            return

        service = ChatService(
            platform_id=_WA_PLATFORM_ID,
            org_id=None,
            system_prompt=platform.get_system_prompt(),
            db_connections=[c.model_dump() for c in platform.db_connections],
        )

        result = await service.chat(
            message=message,
            user_id=phone,
            session_id=f"wa_{phone}",
        )

        answer = markdown_to_whatsapp(result["answer"])
        await send_text_message(phone, answer)

    except Exception as exc:
        print(f"[webhook] error processing message from {phone}: {exc}")
