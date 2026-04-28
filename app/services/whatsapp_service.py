import re

import httpx

from app.core.config import get_settings

settings = get_settings()

_WA_API_URL = "https://graph.facebook.com/v18.0/{phone_id}/messages"


def markdown_to_whatsapp(text: str) -> str:
    """Convierte formato Markdown básico al formato de WhatsApp."""
    # Eliminar bloques HTML completos (tarjetas, divs, etc.)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    # [texto](url) → texto: url  (WhatsApp no renderiza markdown links, pero sí URLs planas)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'\1: \2', text)
    # **negrita** → *negrita*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    # __negrita__ → *negrita*
    text = re.sub(r"__(.+?)__", r"*\1*", text)
    # _cursiva_ → _cursiva_ (WhatsApp ya usa este formato, sin cambio)
    # ### Encabezados → *Encabezado*
    text = re.sub(r"^#{1,3}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
    # Bloques de código → sin backticks
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).replace("```", ""), text)
    # Limpiar líneas vacías excesivas
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def send_text_message(to: str, text: str) -> None:
    """Envía un mensaje de texto a un número de WhatsApp vía Meta Cloud API."""
    url = _WA_API_URL.format(phone_id=settings.whatsapp_phone_id)
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            print(f"[whatsapp] send error {resp.status_code}: {resp.text}")
