"""
Clasificador de intención liviano.

Intents:
  - greeting     → saludo, despedida, agradecimiento ("hola", "gracias", "bye")
  - chitchat     → conversación general sin necesidad de datos ("¿cómo estás?")
  - general_knowledge → pregunta conceptual que se responde con conocimiento general
  - platform_query    → necesita datos de la plataforma (catálogo, cursos, eventos)
  - personal_query    → necesita datos del usuario (progreso, inscripciones, certificados)

Flujo:
  1. Reglas rápidas (sin LLM) para greeting/chitchat obvios.
  2. Si no hay match, LLM clasifica entre los intents restantes.
"""
from __future__ import annotations

import re

_GREETING_PATTERNS = [
    r"^\s*(hola|hello|hi|hey|buenas|buenos días|buenas tardes|buenas noches|buen día)\b",
    r"^\s*(gracias|thank you|thanks|muchas gracias|de nada|perfecto|ok|okey|okay|entendido|listo)\s*[!.]*\s*$",
    r"^\s*(bye|adiós|adios|hasta luego|chao|chau|nos vemos)\b",
    r"^\s*(👋|😊|🙏|✅)\s*$",
]
_GREETING_RE = re.compile("|".join(_GREETING_PATTERNS), re.IGNORECASE)

_CHITCHAT_PATTERNS = [
    r"\bcómo estás\b", r"\bcomo estas\b", r"\bqué tal\b", r"\bque tal\b",
    r"\bcómo te llamas\b", r"\bquién eres\b", r"\bque eres\b",
    r"\bcuántos años tienes\b", r"\beres humano\b", r"\beres un bot\b",
]
_CHITCHAT_RE = re.compile("|".join(_CHITCHAT_PATTERNS), re.IGNORECASE)

_PERSONAL_KW = [
    "mis ", "mi ", "mío", "mía", "tengo", "he tomado", "he completado",
    "mi progreso", "mis certificados", "mis inscripciones", "estoy inscrito",
    "my ", "i have", "i am enrolled",
    "evalúame", "evaluame", "examen", "quiz", "prueba", "evaluarme",
]

_PLATFORM_KW = [
    "cursos", "curso", "evento", "eventos", "actividad", "actividades",
    "módulo", "modulo", "módulos", "modulos", "programa", "diplomado",
    "certificación", "catalogo", "catálogo", "disponibles", "oferta",
    "horario", "inscribir", "inscripción", "precio", "costo",
]


def _rule_based_intent(message: str) -> str | None:
    """Devuelve intent si es detectable por reglas, None si necesita LLM."""
    if _GREETING_RE.search(message):
        return "greeting"
    if _CHITCHAT_RE.search(message):
        return "chitchat"
    return None


async def classify_intent(message: str) -> str:
    """
    Clasifica la intención del mensaje.
    Primero reglas rápidas, luego LLM si no hay match claro.
    """
    rule = _rule_based_intent(message)
    if rule:
        print(f"[intent] rule-based → {rule}")
        return rule

    result = await _llm_classify(message)
    print(f"[intent] llm → {result}")
    return result


async def _llm_classify(message: str) -> str:
    from google import genai
    from google.genai import types
    import warnings
    from app.core.config import get_settings

    settings = get_settings()
    prompt = (
        "Clasifica la siguiente pregunta en UNA de estas categorías. "
        "Responde SOLO con la categoría, sin explicación:\n\n"
        "- greeting: saludo, despedida, agradecimiento, confirmación corta\n"
        "- chitchat: conversación casual sin necesidad de datos\n"
        "- general_knowledge: pregunta conceptual que se responde con conocimiento general (no necesita base de datos)\n"
        "- platform_query: necesita datos de la plataforma (cursos, eventos, actividades, módulos disponibles)\n"
        "- personal_query: necesita datos específicos del usuario (progreso, inscripciones, certificados, evaluaciones)\n\n"
        "Ejemplos:\n"
        "  'hola' → greeting\n"
        "  '¿cómo estás?' → chitchat\n"
        "  '¿qué es la diabetes?' → general_knowledge\n"
        "  '¿qué cursos tienen sobre endocrinología?' → platform_query\n"
        "  '¿cuál es mi progreso?' → personal_query\n"
        "  'evalúame sobre mis cursos' → personal_query\n\n"
        f"Pregunta: \"{message}\"\n\nCategoría:"
    )

    client = genai.Client(api_key=settings.gemini_api_key)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            response = await client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
        intent = response.text.strip().lower().split()[0]
        valid = {"greeting", "chitchat", "general_knowledge", "platform_query", "personal_query"}
        return intent if intent in valid else "platform_query"
    except Exception as e:
        print(f"[intent] llm classify error: {e}")
        return "platform_query"


def intent_needs_db(intent: str) -> bool:
    return intent in ("platform_query", "personal_query")


def intent_is_personal(intent: str) -> bool:
    return intent == "personal_query"


def intent_needs_llm(intent: str) -> bool:
    """greeting y chitchat pueden responderse sin pipeline completo."""
    return intent not in ("greeting", "chitchat")
