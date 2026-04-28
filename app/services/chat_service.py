"""
Servicio de chat.
Combina historial de conversación + contexto RAG + datos reales de MongoDB + LLM.
Para Gemini usa el SDK nativo (google-genai) para evitar problemas de serialización.
"""
from __future__ import annotations

import asyncio
import json
import re
import urllib.parse
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from app.core.config import get_settings
from app.rag.pipeline import get_retriever
from app.db.mongo_query import fetch_collection_data, docs_to_context, generate_filter_async, search_transcript_segments
from app.db.schema_introspector import load_schema_cache
from app.services.history_service import load_history, save_history, persist_turn

settings = get_settings()


# ─── LLM factory ─────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.2) -> BaseChatModel | None:
    """Devuelve LangChain LLM para openai/anthropic. Para gemini devuelve None (usa SDK nativo)."""
    provider = settings.llm_provider
    if provider == "openai":
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=temperature,
            openai_api_key=settings.openai_api_key,
            streaming=False,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=temperature,
            anthropic_api_key=settings.anthropic_api_key,
        )
    return None  # gemini usa SDK nativo


async def _invoke_gemini(messages: list[dict], temperature: float = 0.2) -> str:
    """Llama a Gemini directamente con el SDK nativo google-genai."""
    import warnings
    from google import genai
    from google.genai import types

    print(f"[gemini] invoking with {len(messages)} messages")
    client = genai.Client(api_key=settings.gemini_api_key)

    system_instruction = None
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
        elif msg["role"] == "model":
            contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=contents,
            config=config,
        )
    print(f"[gemini] response received, text length={len(response.text)}")
    return response.text


async def _invoke_gemini_stream(messages: list[dict], temperature: float = 0.2) -> AsyncGenerator[str, None]:
    """Llama a Gemini con streaming nativo."""
    import warnings
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)

    system_instruction = None
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
        elif msg["role"] == "model":
            contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
        response_stream = await client.aio.models.generate_content_stream(
            model=settings.gemini_model,
            contents=contents,
            config=config,
        )
        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text


async def _invoke_llm(messages: list[dict], temperature: float = 0.2) -> str:
    """Invoca el LLM configurado con lista de mensajes en formato dict."""
    try:
        if settings.llm_provider == "gemini":
            return await _invoke_gemini(messages, temperature)

        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "model":
                lc_messages.append(AIMessage(content=msg["content"]))

        llm = get_llm(temperature)
        response = await llm.ainvoke(lc_messages)
        return response.content
    except Exception as e:
        print(f"[llm] invoke error: {e}")
        raise


async def _invoke_llm_stream(messages: list[dict], temperature: float = 0.2) -> AsyncGenerator[str, None]:
    """Invoca el LLM configurado devolviendo un generador de streaming."""
    try:
        if settings.llm_provider == "gemini":
            async for chunk in _invoke_gemini_stream(messages, temperature):
                yield chunk
            return

        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "model":
                lc_messages.append(AIMessage(content=msg["content"]))

        llm = get_llm(temperature)
        async for chunk in llm.astream(lc_messages):
            yield chunk.content
    except Exception as e:
        print(f"[llm] stream invoke error: {e}")
        raise


# ─── Prompt template ─────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """{system_prompt}

## Contexto del usuario
- ID: `{user_id}`
- Nombre: {user_name}
- OrgId: {org_id}

## DATOS REALES DE LA BASE DE DATOS
{data_context}

## Cómo interpretar los datos
- `progress: 0` significa 0% de avance — el usuario está inscrito pero no ha comenzado.
- `progress: 100` significa completado al 100%.
- `status: ACTIVE` significa inscripción activa.
- `certificationHours` indica horas de certificación obtenidas.
- `attended: true` indica asistencia confirmada.
- Si hay datos en `events` o `activities`, úsalos para describir el curso o evento.

## Instrucciones
- Responde SIEMPRE usando los datos de arriba, aunque sean parciales.
- Si `progress` es 0, dilo claramente: "tienes 0% de avance" o "aún no has comenzado".
- Si el usuario pregunta por recomendaciones, usa los cursos/eventos disponibles para sugerir.
- Responde en el mismo idioma que el usuario.
- NO digas que no tienes información si los datos están presentes arriba.
- Nunca muestres IDs de MongoDB en la respuesta.
- IMPORTANTE: Cuando haya tarjetas HTML disponibles, úsalas inyectando tu contenido en {{{{RESUMEN_AQUI}}}}. NO inventes URLs que no estén en los datos.
- Tu respuesta debe ser natural, aportando el contexto general y la respuesta a la pregunta del usuario.

## Cómo citar fuentes
- Cuando uses datos de la base de datos, menciona el nombre del curso, evento o actividad de forma natural en la respuesta (ej: "en el curso *Diplomado en Endocrinología* tienes un 0% de avance").
- No generes ni repitas URLs en el cuerpo de la respuesta; los enlaces se adjuntan automáticamente al final.

## Instrucciones adicionales
{extra_instructions}

## Corrección ortográfica
- Si el usuario escribe con faltas de ortografía, errores de tildes o términos médicos/educativos mal escritos, CORRÍGELOS AUTOMÁTICAMENTE en tu respuesta.
- Ejemplos comunes: "cancer" → "cáncer", "mama" → "mamá", "informacion" → "información", "evaluacion" → "evaluación", "hipertension" → "hipertensión", "sintomas" → "síntomas".
- Mantén la corrección natural y no menciones que estás corrigiendo errores.

## Tutor Proactivo — Cierre de respuesta
- Finaliza SIEMPRE con 1 o 2 sugerencias concretas y naturales basadas en los datos disponibles arriba.
- Las sugerencias deben invitar al usuario a explorar el siguiente paso lógico: ver módulos, evaluar progreso, buscar un tema relacionado, hacer un quiz, etc.
- Varía el estilo: a veces una pregunta directa, a veces una invitación. Nunca uses la misma frase de cierre dos veces.
- Basa las sugerencias ÚNICAMENTE en los datos presentes; no inventes opciones inexistentes.
- Ejemplos de cierres naturales: "¿Quieres que revise cuántos módulos le quedan?", "Si te interesa, puedo mostrarte las actividades de ese diplomado.", "También puedo buscarte cursos similares disponibles ahora."
"""

NO_CONTEXT_TEMPLATE = """{system_prompt}

## Corrección ortográfica
- Si el usuario escribe con faltas de ortografía, errores de tildes o términos médicos/educativos mal escritos, CORRÍGELOS AUTOMÁTICAMENTE en tu respuesta.
- Ejemplos comunes: "cancer" → "cáncer", "mama" → "mamá", "informacion" → "información", "evaluacion" → "evaluación", "hipertension" → "hipertensión", "sintomas" → "síntomas".
- Mantén la corrección natural y no menciones que estás corrigiendo errores.

## Tutor Proactivo — Cierre de respuesta
- Aunque no encuentres datos específicos, orienta al usuario sugiriendo cómo puede explorar la plataforma o qué información puede pedirte.
- Cierra con 1 pregunta o sugerencia concreta que lo invite a continuar la conversación.

No se encontró información relevante en la base de datos para esta consulta.
Responde indicando que no tienes datos disponibles sobre este tema, pero guía al usuario hacia lo que sí puede explorar.
## IMPORTANTE
No se encontró información relevante en la base de datos para esta consulta.
- NO inventes cursos, eventos, actividades ni datos que no existan.
- Si no tienes datos concretos, dilo claramente y ofrece ayudar con una pregunta más específica.
- Responde solo con conocimiento general si la pregunta es conceptual (no sobre el catálogo).
"""


def build_prompt(
    system_prompt: str,
    data_context: str,
    user_id: str,
    user_name: str | None,
    org_id: str | None,
    history: list[dict],
    extra_instructions: str | None = None,
    history_summary: str | None = None,
) -> list[dict]:
    """Construye lista de mensajes en formato dict {role, content}."""
    has_data = bool(data_context) and "No se encontraron" not in data_context
    print(f"[build_prompt] has_data={has_data}, data_context_len={len(data_context)}")

    if has_data:
        system_content = SYSTEM_TEMPLATE.format(
            system_prompt=system_prompt,
            data_context=data_context,
            user_id=user_id,
            user_name=user_name or "desconocido",
            org_id=org_id or "N/A",
            base_url=settings.gencampus_base_url,
            extra_instructions=extra_instructions or "",
        )
    else:
        system_content = NO_CONTEXT_TEMPLATE.format(system_prompt=system_prompt)

    # Inyectar resumen de conversación anterior si existe
    if history_summary:
        system_content += f"\n\n## CONTEXTO DE CONVERSACIÓN ANTERIOR\n{history_summary}"

    print(f"[build_prompt] using={'SYSTEM_TEMPLATE' if has_data else 'NO_CONTEXT_TEMPLATE'} summary={'yes' if history_summary else 'no'}")

    messages: list[dict] = [{"role": "system", "content": system_content}]
    for msg in history:
        role = "user" if msg.get("role") == "user" else "model"
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        messages.append({"role": role, "content": content})
    return messages


# ─── Message conversion ──────────────────────────────────────────────────────

def _to_langchain_history(
    history: list[dict[str, str]],
) -> list[HumanMessage | AIMessage]:
    result = []
    for msg in history:
        content = msg.get("content", "")
        # Asegurar que el contenido sea siempre string
        if not isinstance(content, str):
            content = str(content)
        if msg.get("role") == "user":
            result.append(HumanMessage(content=content))
        elif msg.get("role") == "assistant":
            result.append(AIMessage(content=content))
    return result


# ─── Chat service ─────────────────────────────────────────────────────────────

async def _save_quiz_result(uri: str, database: str, platform_id: str, user_id: str, org_id: str | None, quiz_data: dict) -> None:
    """Guarda el resultado del quiz generado por el LLM en una colección dedicada."""
    from pymongo import MongoClient
    import datetime

    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            retryWrites=True,
        )
        db = client[database]

        col_name = f"quizzes_{platform_id}"

        doc = {
            "user_id": user_id,
            "org_id": org_id,
            "created_at": datetime.datetime.now(datetime.timezone.utc),
            "quiz_data": quiz_data.get("quiz_result", quiz_data)
        }

        db[col_name].insert_one(doc)
        print(f"[quiz] Quiz result saved successfully in {col_name} for user {user_id}")
    except Exception as e:
        print(f"[quiz] Error saving quiz result: {e}")
    finally:
        client.close()

class ChatService:
    def __init__(
        self,
        platform_id: str,
        org_id: str | None,
        system_prompt: str,
        db_connections: list[dict] | None = None,
    ):
        self.platform_id = platform_id
        self.org_id = org_id
        self.system_prompt = system_prompt
        self.db_connections = db_connections or []
        self.retriever = get_retriever(platform_id, org_id)
        self.llm = get_llm()

    async def chat(
        self,
        message: str,
        user_id: str,
        user_name: str | None = None,
        org_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        session_id = session_id or str(uuid.uuid4())

        # Resolver org_id desde la DB si no viene en el request
        if not org_id and self.db_connections:
            from app.services.socratic_agent import _resolve_org_id
            org_id = await asyncio.to_thread(_resolve_org_id, self.db_connections[0], user_id) or org_id
            if org_id:
                print(f"[chat] resolved org_id from db: {org_id}")

        user_id_field = "user_id" if self.db_connections and self._uses_snake_case() else "userId"

        # Detectar si la pregunta es personal o una consulta general del catálogo
        _personal_kw = [
            "mis", "mi", "mío", "mía", "tengo", "he tomado", "he completado",
            "mi progreso", "mis certificados", "mis inscripciones", "estoy inscrito",
            "my", "i have", "i am enrolled",
            "evaluame", "evalúame", "examen", "quiz", "prueba", "evaluado", "evaluarme", "preguntas"
        ]
        
        pattern_personal = r'\b(?:' + '|'.join(_personal_kw) + r')\b'
        is_personal = bool(re.search(pattern_personal, message.lower()))

        if is_personal:
            ctx_parts = [f"{user_id_field}: {user_id}"]
            if user_name:
                ctx_parts.append(f"nombre: {user_name}")
            # NOTA: no incluir org_id en el enriquecimiento — se usa solo para URLs, no para filtros MongoDB
            enriched_message = f"{message} [{', '.join(ctx_parts)}]"
        else:
            # Consulta general — sin contexto de usuario
            enriched_message = message

        # ── Paso 1: RAG + historial + clasificación de intención en paralelo ──────────────────────────────
        # Fast-track: si usa ciertas palabras clave muy específicas, forzar DB sin preguntar al LLM
        force_db_kw = ["evaluame", "evalúame", "examen", "quiz", "prueba", "evaluado", "evaluarme", "preguntas", "progreso", "inscrito", "video", "videos"]
        pattern_force_db = r'\b(?:' + '|'.join(force_db_kw) + r')\b'
        fast_track_db = bool(re.search(pattern_force_db, message.lower()))

        if fast_track_db:
            (rag_context, sources), history = await asyncio.gather(
                asyncio.to_thread(self._rag_retrieve, message),
                load_history(self.platform_id, user_id, session_id),
            )
            needs_db = True
        else:
            (rag_context, sources), history, needs_db = await asyncio.gather(
                asyncio.to_thread(self._rag_retrieve, message),
                load_history(self.platform_id, user_id, session_id),
                _requires_db_query(message),
            )

        print(f"[chat] Intent classifier: needs_db={needs_db} (fast_track={fast_track_db})")

        # ── Paso 2: generar filtro + cargar schema cache en paralelo ─────────
        data_parts: list[str] = []
        if rag_context:
            data_parts.append("### CONOCIMIENTO DE LA PLATAFORMA (RAG)\n" + rag_context)
            
        docs: list[dict] = []
        transcript_docs: list[dict] = []
        primary: str | None = None
        conn = self.db_connections[0] if self.db_connections else {}
        is_video_summary = False
        is_activity_question = False

        if self.db_connections and needs_db:
            try:
                # RAG may suggest a collection, but we rely heavily on heuristics for the content RAG
                suggested_primary = sources[0].get("collection", "") if sources else ""

                _activity_kw = ["actividad", "actividades", "clase", "clases", "lección", "lecciones"]
                _module_kw = ["módulo", "modulo", "módulos", "modulos"]
                _course_kw = ["curso", "cursos", "evento", "eventos", "programa", "diplomado", "certificación", "simposio", "congreso"]
                _video_kw = ["video", "videos", "grabación", "grabacion", "transcripción", "transcripcion", "conferencia", "conferencias"]
                _user_kw = ["usuario", "mi", "mis", "progreso", "inscrito"]
                
                msg_lower = message.lower()
                
                def _has_kw(kws: list[str]) -> bool:
                    pattern = r'\b(?:' + '|'.join(kws) + r')\b'
                    return bool(re.search(pattern, msg_lower))

                # 1. Determinar colección principal en base a la intención del usuario
                if is_personal or _has_kw(_user_kw):
                    primary = "courseattendees"
                elif _has_kw(_video_kw):
                    primary = "transcript_segments"
                elif _has_kw(_activity_kw):
                    primary = "activities"
                elif _has_kw(_module_kw):
                    primary = "modules"
                elif _has_kw(_course_kw):
                    primary = "events"
                else:
                    # Fallback al RAG o a events por defecto
                    primary = suggested_primary or "events"

                is_video_summary = primary == "transcript_segments" and _has_kw(["resumen", "resumir", "resúmen"])
                is_activity_question = primary in ("activities", "modules")

                if primary:
                    # Interceptar consultas dependientes de eventos en GenCampus
                    if self.platform_id == "gencampus" and primary in ("modules", "activities", "courseattendees"):
                        # Si el usuario menciona "curso" o "evento", buscar primero en events para obtener el ID
                        _course_kw_dep = ["curso", "evento", "programa", "diplomado", "certificación"]
                        if _has_kw(_course_kw_dep):
                            print(f"[chat] resolving event dependency for {primary} based on message")
                            
                            # Extraer el nombre del curso de la pregunta usando el extractor de temas genérico
                            course_topic = await _extract_topic(message, "el nombre del curso o evento")
                            if course_topic and course_topic.lower() not in _course_kw_dep:
                                from bson import ObjectId
                                # Usar fuzzy search en events para ese topic
                                event_docs = await asyncio.to_thread(
                                    _run_query, conn, "events", 
                                    {"name": {"$regex": course_topic, "$options": "i"}}, 
                                    None, 2
                                )
                                if event_docs:
                                    event_ids = [str(d.get("_id", "")) for d in event_docs if d.get("_id")]
                                    if event_ids:
                                        enriched_message += f" [Encontramos el curso {course_topic}, su event_id/eventId es: {event_ids[0]}]"
                                        print(f"[chat] resolved event_id={event_ids[0]} for course mention '{course_topic}'")

                    collections_list = conn.get("collections")
                    print(f"[chat] loading cache with collections={collections_list}")
                    cached = load_schema_cache(conn["uri"], conn["database"], collections_list)
                    # Si no encuentra con collections específico, intentar con None (todas las colecciones)
                    if not cached and collections_list:
                        print(f"[chat] cache miss with specific collections, retrying with None")
                        cached = load_schema_cache(conn["uri"], conn["database"], None)
                    col_info = (cached or {}).get("collections", {}).get(primary, {})
                    print(f"[chat] cached_available={bool(cached)} primary={primary} col_info_keys={list(col_info.keys())}")
                    schema_fields = list(col_info.get("schema", {}).keys())
                    enum_values = col_info.get("enum_values", {})

                    # Generar filtro async solo si no es transcript_segments
                    if primary == "transcript_segments":
                        docs = await _search_transcripts_async(conn, message)
                    else:
                        # Intento de generar filtro
                        mongo_filter = await generate_filter_async(enriched_message, primary, schema_fields, enum_values)
                        print(f"[chat] schema_fields={schema_fields} mongo_filter_before={json.dumps(mongo_filter, default=str)}")
                        
                        # Si sabemos que estamos buscando un evento en dependencias, forzar o inyectar el event_id en el filtro
                        if self.platform_id == "gencampus" and primary in ("modules", "activities") and "eventId es:" in enriched_message:
                            match = re.search(r"eventId es:\s*([a-fA-F0-9]{24})", enriched_message)
                            if match:
                                event_id_val = match.group(1)
                                # Si no hay filtro, lo creamos
                                if not mongo_filter:
                                    mongo_filter = {"$or": [{"eventId": event_id_val}, {"event_id": event_id_val}]}
                                else:
                                    # Limpiamos cualquier búsqueda errónea por nombre/título y priorizamos el evento
                                    mongo_filter.pop("name", None)
                                    mongo_filter.pop("title", None)
                                    if not mongo_filter:
                                        mongo_filter = {"$or": [{"eventId": event_id_val}, {"event_id": event_id_val}]}
                                    else:
                                        # Si había más cosas en el filtro, las anidamos con un AND
                                        mongo_filter = {"$and": [{"$or": [{"eventId": event_id_val}, {"event_id": event_id_val}]}, mongo_filter]}
                                print(f"[chat] forced event_id filter created from enriched message: {mongo_filter}")

                        # Expandir búsquedas de texto a palabras clave individuales
                        mongo_filter = _expand_text_filter_to_keywords(mongo_filter, schema_fields)
                        print(f"[chat] mongo_filter_after_fuzzy_options={json.dumps(mongo_filter, default=str)}")
                        # Validar que el filtro solo use campos que existen en el schema (solo si tenemos schema)
                        if schema_fields:
                            mongo_filter = _validate_filter_fields(mongo_filter, set(schema_fields))
                            print(f"[chat] mongo_filter_after_validation={json.dumps(mongo_filter, default=str)}")
                        else:
                            print(f"[chat] WARNING: no schema_fields available, skipping filter validation")
                        if self.platform_id == "gencampus" and primary == "events":
                            mongo_filter = _expand_events_text_filter_for_gencampus(mongo_filter)
                        # Detectar si pide el último/más reciente → ordenar por fecha desc, limit 1
                        _latest_kw = ["último", "ultimo", "más reciente", "mas reciente", "más nuevo", "mas nuevo", "latest", "newest"]
                        is_latest = any(kw in message.lower() for kw in _latest_kw)
                        date_fields = ["datetime_from", "datetime_to", "startDate", "start_date", "createdAt", "created_at", "date"]
                        sort_field = next((f for f in date_fields if f in schema_fields), None)
                        # Fallback: ordenar por _id desc (ObjectId tiene timestamp embebido)
                        if is_latest and not sort_field:
                            sort_field = "_id"
                        sort = [(sort_field, -1)] if is_latest and sort_field else None
                        limit = 1 if is_latest else None
                        print(f"[chat] is_latest={is_latest} sort_field={sort_field} sort={sort} limit={limit}")
                        # Limitar la respuesta general a un máximo de 5 documentos
                        docs = await asyncio.to_thread(_run_query, conn, primary, mongo_filter, sort, limit)
                        if docs and limit is None:
                            docs = docs[:5]
                    print(f"[chat] primary={primary} docs={len(docs)}")

                    if docs:
                        if primary == "transcript_segments":
                            if is_video_summary:
                                data_parts.append(_transcripts_to_summary_context(docs))
                            else:
                                data_parts.append(_transcripts_to_context(docs))
                        else:
                            data_parts.append(docs_to_context(primary, docs, org_id or ""))
                            related = await _fetch_related_async(docs, conn, cached, enriched_message, org_id or "")
                            if related:
                                data_parts.append(related)

                    # Si la pregunta es sobre actividades, también buscar en transcript_segments
                    if is_activity_question and primary == "activities" and docs:
                        print(f"[chat] activity question detected, also searching transcript_segments")
                        transcript_docs = await _search_transcripts_async(conn, message)
                        if transcript_docs:
                            transcript_docs = transcript_docs[:5]
                            data_parts.append(_transcripts_to_context(transcript_docs))
                            print(f"[chat] found {len(transcript_docs)} transcript segments related to activities")
            except Exception as e:
                import traceback
                print(f"[chat] CRITICAL ERROR IN DB PHASE: {e}")
                traceback.print_exc()

        data_context = "\n\n".join(data_parts) if data_parts else "No se encontraron datos relevantes en la base de datos para esta consulta."
        print(f"[chat] data_context_len={len(data_context)}")

        # ── Paso 4: invocar LLM ───────────────────────────────────────────────
        extra_instructions = ""
        
        # Preparar las plantillas HTML de las tarjetas para que el LLM inyecte sus resúmenes
        templates_context = ""
        if self.platform_id == "gencampus":
            templates_html = ""
            if docs and primary in ("events", "activities", "transcript_segments"):
                templates_html += await _build_gencampus_cards_template_async(conn, docs, primary, org_id or "")
            if transcript_docs and primary != "transcript_segments":
                templates_html += await _build_gencampus_cards_template_async(conn, transcript_docs, "transcript_segments", org_id or "")
            
            if templates_html:
                templates_context = f"\n\n## PLANTILLAS DE TARJETAS HTML DISPONIBLES\nEl sistema ha preparado las siguientes tarjetas visuales. PARA CADA VIDEO O CURSO QUE MENCIONES, DEBES imprimir EXACTAMENTE su bloque HTML correspondiente, y reemplazar el texto '{{{{RESUMEN_AQUI}}}}' con tu explicación elaborada. NO uses markdown para las tarjetas, imprime el HTML tal cual.\n\n{templates_html}\n\n"

        if is_video_summary:
            extra_instructions = (
                "- Si el usuario pide un resumen de un video específico, usa solo las transcripciones encontradas "
                "para generar un resumen claro y puntual del contenido del video, e inyéctalo en la tarjeta HTML correspondiente."
            )
        elif primary == "transcript_segments" or transcript_docs:
            extra_instructions = (
                "- Basándote en los segmentos de transcripción y actividades provistas, genera una respuesta "
                "elaborada y contextualizada. "
                "- Para cada video, usa SU TARJETA HTML EXACTA proporcionada arriba, reemplazando {{{{RESUMEN_AQUI}}}} con tu explicación de qué se habla en el video. "
                "- Dentro de tu explicación, puedes mencionar los tiempos (ej. [00:15:30]) integrados en tu narrativa. "
                "- NO generes listas repetitivas de tiempos fuera de la tarjeta."
            )
        elif templates_context:
            extra_instructions = (
                "- Para cada curso o actividad que recomiendes o describas, IMPRIME su bloque HTML completo correspondiente "
                "proporcionado arriba, reemplazando {{{{RESUMEN_AQUI}}}} con tu descripción o resumen."
            )
            
        # Añadir las plantillas al data_context
        full_context = data_context + templates_context

        # Comprimir historial largo: resumen de turnos antiguos + ventana reciente
        from app.services.history_service import compress_history_for_prompt
        history_summary, recent_history = await compress_history_for_prompt(history)

        messages = build_prompt(
            self.system_prompt,
            full_context,
            user_id,
            user_name,
            org_id,
            recent_history,
            extra_instructions=extra_instructions,
            history_summary=history_summary or None,
        )
        print(f"[chat] prompt_messages={len(messages)} history_turns={len(recent_history)//2} summary={'yes' if history_summary else 'no'}")
        messages.append({"role": "user", "content": message})
        
        # Aumentar la temperatura y añadir un ruido algorítmico si estamos en modo evaluación
        # para forzar la aleatoriedad en las preguntas
        chat_temperature = 0.2
        if is_personal and any(kw in message.lower() for kw in ["evaluame", "evalúame", "examen", "quiz", "prueba", "evaluado", "evaluarme", "preguntas"]):
            chat_temperature = 0.6
            import time
            messages[-1]["content"] += f"\n\n[System note: Seed temporal para forzar variedad en este quiz: {time.time()}]"
        
        answer_text = await _invoke_llm(messages, temperature=chat_temperature)
        print(f"[chat] answer_text={answer_text[:200] if answer_text else 'EMPTY'}")

        # ── Paso 5: guardar historial Redis + persistir en MongoDB ───────────
        history_turns_count = len(history)  # capturar antes de agregar el turno actual

        # Interceptar resultado de quiz si el LLM emitió el bloque JSON
        quiz_save_coro = None
        quiz_match = re.search(r"```json\s*(\{.*?\"quiz_result\".*?\})\s*```", answer_text, re.DOTALL)
        if quiz_match:
            quiz_json_str = quiz_match.group(1)
            try:
                quiz_data = json.loads(quiz_json_str)
                # Preparar corutina para guardar el quiz de forma asíncrona segura
                uri = self.db_connections[0]["uri"] if self.db_connections else settings.meta_mongodb_uri
                database = self.db_connections[0]["database"] if self.db_connections else settings.meta_mongodb_db
                quiz_save_coro = _save_quiz_result(uri, database, self.platform_id, user_id, org_id, quiz_data)
                
                # Remover el bloque json del texto que verá el usuario
                answer_text = answer_text[:quiz_match.start()] + answer_text[quiz_match.end():]
                answer_text = answer_text.strip()
            except Exception as e:
                print(f"[chat] error parsing quiz json: {e}")

        # Como las tarjetas ahora las construye e inyecta el propio LLM (o las omitimos de la pre-concatenación),
        # solo formateamos las fechas
        answer_text = _format_dates_in_text(answer_text)

        # Adjuntar sección de fuentes si el LLM no la incluyó ya
        if "Para profundizar" not in answer_text and "Fuente:" not in answer_text:
            fuentes = _build_fuentes_section(docs, primary, sources, org_id)
            if fuentes:
                answer_text = answer_text.rstrip() + fuentes

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": _strip_html_for_history(answer_text)})
        collection_used = sources[0].get("collection") if sources else None
        
        # Guardar historial + persistir + generar follow-ups en paralelo
        tasks_to_gather = [
            save_history(self.platform_id, user_id, session_id, history),
            persist_turn(
                platform_id=self.platform_id,
                user_id=user_id,
                session_id=session_id,
                user_message=message,
                assistant_message=answer_text,
                user_name=user_name,
                org_id=org_id,
                collection_used=collection_used,
                sources=sources,
            ),
            _generate_follow_ups(message, answer_text, data_context, self.platform_id),
        ]
        if quiz_save_coro:
            tasks_to_gather.append(quiz_save_coro)

        gather_results = await asyncio.gather(*tasks_to_gather, return_exceptions=True)
        follow_up_questions: list[str] = (
            gather_results[2] if isinstance(gather_results[2], list) else []
        )

        return {
            "answer": answer_text,
            "session_id": session_id,
            "sources": [sources[0]] if sources else [],
            "platform_id": self.platform_id,
            "org_id": self.org_id,
            "sources_used": {
                "rag_chunks": len(sources),
                "mongodb_collection": primary,
                "mongodb_docs": len(docs),
                "history_turns": history_turns_count,
            },
            "follow_up_questions": follow_up_questions,
        }

    def _rag_retrieve(self, message: str) -> tuple[str, list[dict]]:
        """RAG síncrono envuelto para correr en thread pool."""
        try:
            return self.retriever.retrieve_as_context(message)
        except ValueError:
            return "", []

    def _uses_snake_case(self) -> bool:
        """Detecta si la plataforma usa snake_case para IDs (ej: gencampus)."""
        from app.core.config import get_settings
        s = get_settings()
        conn = self.db_connections[0] if self.db_connections else {}
        gencampus_cols = set(s.gencampus_allowed_collections)
        platform_cols = set(conn.get("collections") or [])
        return bool(platform_cols & gencampus_cols)


async def _search_transcripts_async(conn: dict, query: str) -> list[dict]:
    """Extrae el tema con LLM y busca en transcript_segments.text."""
    topic = await _extract_topic(query)
    print(f"[transcript] extracted topic='{topic}'")
    return await asyncio.to_thread(
        search_transcript_segments, conn["uri"], conn["database"], topic
    )


async def _extract_topic(query: str, instruction: str = "el tema principal, enfermedad, tratamiento o concepto a buscar en videos") -> str:
    """Extrae un tema o nombre de la pregunta del usuario."""
    from google import genai
    from google.genai import types
    import warnings

    prompt = (
        f"Extrae EXACTAMENTE {instruction} de la siguiente pregunta. "
        f"Ignora palabras coloquiales, verbos (explican, dicen), conectores, y no incluyas palabras como 'videos', 'minutos', 'conferencias' o 'clases'. "
        f"Responde SOLO con el término clave principal (ej: 'hipotiroidismo', 'cáncer de mama').\n\n"
        f"Pregunta: \"{query}\"\n\n"
        f"Extracción:"
    )

    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )
    return response.text.strip().strip('"').strip("'")


async def _requires_db_query(query: str) -> bool:
    """
    Usa el LLM para determinar si la pregunta del usuario requiere buscar 
    datos en vivo desde la base de datos de la plataforma.
    """
    from google import genai
    from google.genai import types
    import warnings

    prompt = (
        "Analiza la siguiente pregunta y determina si el asistente necesita buscar información específica "
        "en una base de datos educativa (catálogo de cursos, progreso, actividades, usuarios, videos o certificados). "
        "Responde SOLO 'SI' o 'NO'.\n\n"
        "Ejemplos que SÍ requieren base de datos:\n"
        "- ¿Cuál es mi progreso en el curso de endocrinología?\n"
        "- Busca un video donde mencionen el ayuno intermitente.\n"
        "- ¿Qué módulos tiene el curso ENDIMET?\n"
        "- ¿Cuáles cursos hay disponibles sobre la diabetes?\n"
        "- Quiero que me evalúes o me hagas un examen de mis cursos.\n"
        "- Hazme una prueba.\n\n"
        "Ejemplos que NO requieren base de datos (se responden con conocimiento general o de la documentación):\n"
        "- Hola, ¿cómo estás?\n"
        "- Explícame qué es el hipotiroidismo y sus síntomas.\n"
        "- ¿Para qué sirve la insulina?\n"
        "- Escríbeme un correo de agradecimiento.\n\n"
        f"Pregunta: \"{query}\"\n\n"
        "Respuesta:"
    )

    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            response = await client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
        ans = response.text.strip().lower()
        return "si" in ans or "sí" in ans
    except Exception as e:
        print(f"[intent] _requires_db_query failed: {e}")
        return True # Por seguridad, asumir que sí necesita DB si hay error


async def _generate_follow_ups(
    user_message: str,
    answer: str,
    data_context: str,
    platform_id: str,
    n: int = 3,
) -> list[str]:
    """
    Genera preguntas de seguimiento contextuales reutilizando data_context ya recuperado.
    Corre en paralelo con save_history, sin llamadas adicionales a DB.
    """
    from google import genai
    from google.genai import types
    import warnings

    ctx_snippet = data_context[:2000] if len(data_context) > 2000 else data_context
    answer_snippet = answer[:400] if len(answer) > 400 else answer

    prompt = (
        f"Eres un asistente experto de la plataforma '{platform_id}'.\n"
        f"El usuario preguntó: \"{user_message}\"\n"
        f"Respondiste: \"{answer_snippet}\"\n\n"
        f"Contexto de datos disponibles:\n{ctx_snippet}\n\n"
        f"Genera exactamente {n} preguntas de seguimiento cortas, naturales y en español "
        f"que el usuario podría querer hacer a continuación, basadas ÚNICAMENTE en los datos "
        f"de contexto de arriba. Sin numeración ni viñetas, una pregunta por línea."
    )

    _settings = get_settings()
    client = genai.Client(api_key=_settings.gemini_api_key)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            response = await client.aio.models.generate_content(
                model=_settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3),
            )
        lines = [ln.strip() for ln in response.text.strip().split("\n") if ln.strip()]
        return lines[:n]
    except Exception as exc:
        print(f"[follow_ups] error: {exc}")
        return []


_LINKABLE_COLLECTIONS = {"events", "activities", "courseattendees", "modules"}


def _build_fuentes_section(
    docs: list[dict],
    primary: str | None,
    rag_sources: list[dict],
    org_id: str | None,
) -> str:
    """Construye '📚 Para profundizar:' con los recursos reales encontrados."""
    base = settings.gencampus_base_url

    lines: list[str] = []
    seen: set[str] = set()

    if docs and primary in _LINKABLE_COLLECTIONS:
        for doc in docs[:5]:
            name = (
                doc.get("name") or doc.get("title") or
                doc.get("eventName") or doc.get("name_event") or
                doc.get("eventTitle") or doc.get("courseName") or
                doc.get("course_name")
            )

            url: str | None = None
            if org_id:
                if primary == "events":
                    eid = doc.get("_id")
                    if eid:
                        url = f"{base}/organization/{org_id}/course/{eid}"
                elif primary == "activities":
                    aid = doc.get("_id")
                    if aid:
                        url = f"{base}/organization/{org_id}/activitydetail/{aid}"
                elif primary in ("courseattendees", "modules"):
                    eid = doc.get("eventId") or doc.get("event_id")
                    if eid:
                        url = f"{base}/organization/{org_id}/course/{eid}"

            if not name and not url:
                continue

            key = url or name
            if key in seen:
                continue
            seen.add(key)

            label = name or "Ver recurso"
            lines.append(f"- [{label}]({url})" if url else f"- {label}")

    if lines:
        return "\n\n📚 *Para profundizar:*\n" + "\n".join(lines)

    return ""


def _strip_html_for_history(text: str) -> str:
    """Quita tags HTML de la respuesta antes de guardar en historial.
    El LLM no necesita re-leer HTML propio para mantener contexto conversacional;
    el texto limpio es suficiente y evita que el historial crezca innecesariamente.
    """
    if not text:
        return text
    cleaned = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def _format_seconds(seconds: float) -> str:
    """Format seconds into hh:mm:ss"""
    if seconds is None:
        return "00:00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _transcripts_to_context(activities: list[dict]) -> str:
    if not activities:
        return "No se encontraron videos relacionados con ese tema."
    lines = [f"### Videos encontrados ({len(activities)} actividades)\n"]
    for i, act in enumerate(activities, 1):
        lines.append(f"[{i}] Actividad: {act['name_activity']}")
        lines.append(f"    ID: {act['activity_id']}")
        for seg in act["segments"]:
            time_str = _format_seconds(seg.get('startTime', 0))
            lines.append(f"    [{time_str}] {seg['text'].strip()}")
        lines.append("")
    return "\n".join(lines)


def _transcripts_to_summary_context(activities: list[dict]) -> str:
    if not activities:
        return "No se encontraron videos relacionados con ese tema."
    lines = ["### Transcripciones para resumen de video específico\n"]
    for i, act in enumerate(activities, 1):
        lines.append(f"[{i}] Actividad: {act['name_activity']}")
        for seg in act["segments"]:
            lines.append(seg["text"].strip())
        lines.append("")
    lines.append("Resumen esperado: usa el texto anterior para generar un resumen breve y claro del contenido del video.")
    return "\n".join(lines)


def _format_dates_in_text(text: str) -> str:
    if not text:
        return text

    month_names = {
        1: "ene", 2: "feb", 3: "mar", 4: "abr",
        5: "may", 6: "jun", 7: "jul", 8: "ago",
        9: "sep", 10: "oct", 11: "nov", 12: "dic"
    }

    def _replace_iso_full(match):
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            # Ignoramos hora y minutos para el texto general
            # Retorna formato: 28 de mar de 2025
            dt = datetime(year, month, day)
            return f"{dt.day} de {month_names[dt.month]} de {dt.year}"
        except ValueError:
            return match.group(0)

    def _replace_iso(match):
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            dt = datetime(year, month, day)
            return f"{dt.day} de {month_names[dt.month]} de {dt.year}"
        except ValueError:
            return match.group(0)

    def _replace_dmy(match):
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            dt = datetime(year, month, day)
            return f"{dt.day} de {month_names[dt.month]} de {dt.year}"
        except ValueError:
            return match.group(0)

    def _replace_spanish_month(match):
        day = int(match.group(1))
        month_name = match.group(2).lower()
        year = int(match.group(3))
        month_map = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
            "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
            "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
        }
        month = month_map.get(month_name)
        if not month:
            return match.group(0)
        try:
            dt = datetime(year, month, day)
            return f"{dt.day} de {month_names[dt.month]} de {dt.year}"
        except ValueError:
            return match.group(0)

    # Reemplazar fechas tipo ISO 8601 con T y Z (2025-03-28T16:43:00)
    text = re.sub(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})T\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?Z?\b", _replace_iso_full, text)
    # Reemplazar fechas estándar (2025-03-28)
    text = re.sub(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", _replace_iso, text)
    text = re.sub(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", _replace_dmy, text)
    text = re.sub(
        r"\b(\d{1,2}) de (enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre) de (\d{4})\b",
        _replace_spanish_month,
        text,
        flags=re.IGNORECASE,
    )
    return text


def _get_image_from_event_sync(uri: str, database: str, event_id: str) -> str | None:
    from pymongo import MongoClient
    from bson import ObjectId
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client[database]
        
        # Try to parse as ObjectId or use as string
        query_id = event_id
        if len(event_id) == 24 and event_id.isalnum():
            try:
                query_id = ObjectId(event_id)
            except Exception:
                pass
                
        event_doc = db["events"].find_one({"_id": query_id}) or db["events"].find_one({"_id": str(event_id)})
        if event_doc:
            if isinstance(event_doc.get("styles"), dict) and event_doc["styles"].get("event_image"):
                return event_doc["styles"]["event_image"]
            return event_doc.get("image") or event_doc.get("image_url")
    except Exception as e:
        print(f"[card_html] Error fetching event image: {e}")
    return None

def _get_image_for_activity_sync(uri: str, database: str, activity_id: str) -> str | None:
    from pymongo import MongoClient
    from bson import ObjectId
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client[database]
        
        query_id = activity_id
        if len(activity_id) == 24 and activity_id.isalnum():
            try:
                query_id = ObjectId(activity_id)
            except Exception:
                pass
                
        activity_doc = db["activities"].find_one({"_id": query_id}) or db["activities"].find_one({"_id": str(activity_id)})
        if activity_doc:
            event_id = activity_doc.get("eventId") or activity_doc.get("event_id")
            if event_id:
                return _get_image_from_event_sync(uri, database, event_id)
    except Exception as e:
        print(f"[card_html] Error fetching activity related image: {e}")
    return None

async def _build_gencampus_cards_template_async(conn: dict, docs: list[dict], collection: str, org_id: str) -> str:
    """Genera plantillas HTML para que el LLM inyecte su propio resumen en {{RESUMEN_AQUI}}."""
    if not docs or not org_id:
        return ""

    base_url = settings.gencampus_base_url
    cards_html = []
    
    for doc in docs:
        image_url = None
        if isinstance(doc.get("styles"), dict):
            image_url = doc["styles"].get("event_image")
        image_url = image_url or doc.get("styles.event_image") or doc.get("image") or doc.get("image_url")

        if not image_url:
            if collection == "transcript_segments":
                act_id = doc.get("activity_id")
                if act_id:
                    image_url = await asyncio.to_thread(_get_image_for_activity_sync, conn["uri"], conn["database"], str(act_id))
            elif collection == "activities":
                evt_id = doc.get("eventId") or doc.get("event_id")
                if evt_id:
                    image_url = await asyncio.to_thread(_get_image_from_event_sync, conn["uri"], conn["database"], str(evt_id))

        if collection == "events":
            title = doc.get("name") or doc.get("title") or "Evento"
            subtitle = "Evento"
            url = f"{base_url}/organization/{org_id}/course/{doc.get('_id')}"
            segments_html = ""
        elif collection == "transcript_segments":
            activity_name = doc.get("name_activity") or "Video"
            title = activity_name
            subtitle = "Video / Transcripción"
            
            segments = doc.get("segments", [])
            base_activity_url = f"{base_url}/organization/{org_id}/activitydetail/{doc.get('activity_id')}"
            
            if segments:
                first_time = segments[0].get("startTime", 0)
                fragments_json = urllib.parse.quote(json.dumps(segments))
                url = f"{base_activity_url}?t={first_time}&fragments={fragments_json}"
            else:
                url = base_activity_url
            
            segments_list = ""
            for seg in segments:
                t = seg.get("startTime", 0)
                time_str = _format_seconds(t)
                text = seg.get("text", "").strip()
                
                # Make each timestamp a hyperlink to its exact time
                segment_url = base_activity_url
                if segments:
                    segment_url = f"{base_activity_url}?t={t}&fragments={fragments_json}"
                    
                segments_list += f"<li style=\"margin-bottom:4px;\"><a href=\"{segment_url}\" target=\"_blank\" rel=\"noopener\" style=\"color:#2563eb;text-decoration:none;\"><strong>[{time_str}]</strong></a> {text}</li>"
            
            segments_html = f"<div style=\"margin-top:10px;font-size:13px;color:#374151;background:#f9fafb;padding:8px;border-radius:6px;\"><ul style=\"margin:0;padding-left:20px;\">{segments_list}</ul></div>" if segments_list else ""
        else:
            activity_name = doc.get("name") or doc.get("title") or "Actividad"
            event_name = (
                doc.get("event_name") or doc.get("course_name") or doc.get("name_event") or doc.get("eventTitle") or doc.get("eventName")
            )
            title = activity_name
            subtitle = f"Evento: {event_name}" if event_name else "Actividad"
            url = f"{base_url}/organization/{org_id}/activitydetail/{doc.get('_id')}"
            segments_html = ""

        date_label = ""
        for key in ("startDate", "datetime_from", "datetime_to", "date", "start_date", "createdAt", "created_at"):
            if doc.get(key):
                try:
                    value = doc.get(key)
                    if isinstance(value, str):
                        formatted = _format_dates_in_text(value)
                    elif isinstance(value, datetime):
                        month_names = {
                            1: "ene", 2: "feb", 3: "mar", 4: "abr", 5: "may", 6: "jun",
                            7: "jul", 8: "ago", 9: "sep", 10: "oct", 11: "nov", 12: "dic"
                        }
                        formatted = f"{value.day} de {month_names[value.month]} de {value.year}"
                    else:
                        formatted = str(value)
                    date_label = f"Fecha: {formatted}"
                    break
                except Exception:
                    continue

        image_html = f"<img src=\"{image_url}\" alt=\"Imagen\" style=\"width:100%;height:100%;object-fit:cover;border-radius:12px 0 0 12px;\"/>" if image_url else ""
        image_container = f"<div style=\"flex: 0 0 200px; max-width: 200px; background:#f1f5f9; border-radius:12px 0 0 12px;\">{image_html}</div>" if image_url else ""
        date_html = f"<div style=\"color:#555;font-size:0.95rem;margin-top:4px;\">{date_label}</div>" if date_label else ""
        
        # El bloque donde el LLM inyectará su resumen dinámico
        dynamic_summary_html = f"<div style=\"margin-top:10px;font-size:13px;color:#374151;line-height:1.4;\">{{{{RESUMEN_AQUI}}}}</div>"
        
        card = (
            f"<!-- TARJETA HTML PARA: {title} -->\n"
            f"<div style=\"display:flex;border:1px solid #e2e8f0;border-radius:14px;overflow:hidden;max-width:680px;box-shadow:0 2px 12px rgba(15,23,42,.08);background:#ffffff;margin-bottom:1rem;\">"
            f"{image_container}"
            f"<div style=\"flex:1;padding:16px;display:flex;flex-direction:column;justify-content:center;\">"
            f"<div style=\"font-size:18px;font-weight:700;color:#111827;\">{title}</div>"
            f"<div style=\"color:#4b5563;font-size:14px;margin-top:6px;\">{subtitle}</div>"
            f"{date_html}"
            f"{segments_html}"
            f"{dynamic_summary_html}"
            f"<div style=\"margin-top:12px;\"><a href=\"{url}\" target=\"_blank\" rel=\"noopener\" style=\"display:inline-block;padding:8px 14px;background:#2563eb;color:#ffffff;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;\">Ver en GenCampus</a></div>"
            f"</div></div>"
        )
        cards_html.append(card)

    return "\n\n".join(cards_html)


def _run_query(conn: dict, collection: str, mongo_filter: dict, sort: list | None = None, limit: int | None = None) -> list[dict]:
    """Ejecuta el find en MongoDB con el filtro ya generado."""
    from app.db.mongo_query import _serialize, _REQUIRE_FILTER, MAX_DOCS, ReadOnlyCollection
    from app.db.mongo_pool import get_db

    if not mongo_filter and collection in _REQUIRE_FILTER:
        print(f"[chat] skipping {collection} — requires filter")
        return []

    try:
        db = get_db(conn["uri"], conn["database"])
        col = ReadOnlyCollection(db[collection])
        n = limit or MAX_DOCS
        collation = {"locale": "es", "strength": 1} if _has_regex_in_filter(mongo_filter) else None
        cursor = col._col.find(mongo_filter, limit=n, collation=collation)
        if sort:
            cursor = cursor.sort(sort)
        docs = [_serialize(doc) for doc in cursor]
        print(f"[mongo_query] collection={collection} filter={mongo_filter} docs={len(docs)}")
        return docs
    except Exception as e:
        print(f"[mongo_query] query error: {e}")
        return []


def _has_regex_in_filter(filter_dict: dict) -> bool:
    """Verifica si el filtro contiene alguna búsqueda con $regex."""
    for key, value in filter_dict.items():
        if key.startswith("$"):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and _has_regex_in_filter(item):
                        return True
            elif isinstance(value, dict):
                if _has_regex_in_filter(value):
                    return True
        elif isinstance(value, dict) and "$regex" in value:
            return True
    return False


def _expand_text_filter_to_keywords(mongo_filter: dict, valid_fields: list[str]) -> dict:
    """
    Aplica opciones de búsqueda fuzzy a filtros de texto usando regex de MongoDB.
    Mantiene la frase completa en lugar de expandir a palabras clave individuales.
    Ej: {"description": {"$regex": "cancer de mama"}} 
    -> {"description": {"$regex": "cancer de mama", "$options": "i"}}
    """
    if not isinstance(mongo_filter, dict):
        return mongo_filter

    result = {}
    for key, value in mongo_filter.items():
        if key.startswith("$"):
            # Operadores MongoDB
            if key == "$or" and isinstance(value, list):
                result["$or"] = [_expand_text_filter_to_keywords(item, valid_fields) for item in value]
            elif key == "$and" and isinstance(value, list):
                result["$and"] = [_expand_text_filter_to_keywords(item, valid_fields) for item in value]
            else:
                result[key] = value
        elif isinstance(value, dict) and "$regex" in value:
            # Campo con búsqueda de regex - aplicar opciones fuzzy
            regex_pattern = value["$regex"]
            options = value.get("$options", "")
            
            # Agregar case-insensitive si no está presente
            if "i" not in options:
                options += "i"
            
            result[key] = {"$regex": regex_pattern, "$options": options}
        else:
            result[key] = value
    
    return result


def _extract_search_keywords(text: str) -> list[str]:
    """
    Extrae palabras clave de un texto de búsqueda.
    Aplica corrección ortográfica usando diccionario estático de términos comunes.
    Filtra palabras cortas y stopwords comunes.
    """
    import re
    
    # Stopwords en español e inglés
    stopwords = {
        "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero", "que", "con", "para", "por", "sin", "sobre", "entre", "desde", "hasta",
        "the", "and", "or", "but", "with", "for", "from", "to", "in", "on", "at", "by", "of", "a", "an"
    }
    
    # Diccionario de correcciones ortográficas comunes (faltas de tildes y errores frecuentes)
    corrections = {
        # Falta de tildes
        "cancer": "cáncer",
        "mama": "mamá", 
        "papel": "papel",
        "cafe": "café",
        "universidad": "universidad",
        "facil": "fácil",
        "dificil": "difícil",
        "musica": "música",
        "gracias": "gracias",
        "tambien": "también",
        "estan": "están",
        "estan": "están",
        "aqui": "aquí",
        "ahi": "ahí",
        "alla": "allá",
        "aun": "aún",
        "como": "cómo",
        "cuando": "cuándo",
        "donde": "dónde",
        "porque": "porque",
        "que": "qué",
        "quien": "quién",
        "cual": "cuál",
        "cuales": "cuáles",
        
        # Errores de escritura comunes
        "actividad": "actividad",
        "actividades": "actividades",
        "curso": "curso",
        "cursos": "cursos",
        "clase": "clase",
        "clases": "clases",
        "leccion": "lección",
        "lecciones": "lecciones",
        "video": "video",
        "videos": "videos",
        "transcripcion": "transcripción",
        "transcripciones": "transcripciones",
        "grabacion": "grabación",
        "grabaciones": "grabaciones",
        "conferencia": "conferencia",
        "conferencias": "conferencias",
        
        # Más correcciones comunes
        "recursos": "recursos",
        "material": "material",
        "materiales": "materiales",
        "informacion": "información",
        "contenido": "contenido",
        "contenidos": "contenidos",
        "evaluacion": "evaluación",
        "evaluaciones": "evaluaciones",
        "certificado": "certificado",
        "certificados": "certificados",
        "progreso": "progreso",
        "avance": "avance",
        "completo": "completo",
        "completado": "completado",
        "inscrito": "inscrito",
        "inscrita": "inscrita",
        "inscripcion": "inscripción",
        "registro": "registro",
        
        # Términos médicos comunes
        "diabetes": "diabetes",
        "hipertension": "hipertensión",
        "cancer": "cáncer",
        "infarto": "infarto",
        "alergia": "alergia",
        "alergias": "alergias",
        "asma": "asma",
        "bronquitis": "bronquitis",
        "neumonia": "neumonía",
        "gripe": "gripe",
        "resfriado": "resfriado",
        "dolor": "dolor",
        "fiebre": "fiebre",
        "tos": "tos",
        "estornudo": "estornudo",
        "estornudos": "estornudos",
        "sintomas": "síntomas",
        "tratamiento": "tratamiento",
        "tratamientos": "tratamientos",
        "medicamento": "medicamento",
        "medicamentos": "medicamentos",
        "diagnostico": "diagnóstico",
        "prevencion": "prevención",
        "salud": "salud",
        "enfermedad": "enfermedad",
        "enfermedades": "enfermedades",
    }
    
    # Extraer palabras (letras y números, mínimo 4 caracteres)
    words = re.findall(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑ0-9]{4,}\b', text.lower())
    
    # Aplicar correcciones ortográficas
    corrected_words = []
    for word in words:
        corrected_words.append(corrections.get(word, word))
    
    # Filtrar stopwords y duplicados
    keywords = [word for word in corrected_words if word not in stopwords]
    
    return list(set(keywords))  # Eliminar duplicados


def _expand_events_text_filter_for_gencampus(mongo_filter: dict) -> dict:
    """
    Para GenCampus: si events viene filtrado por `description`, `category` o `title` con regex,
    ampliar a búsqueda en `description` o `name`.
    """
    if not isinstance(mongo_filter, dict):
        return mongo_filter

    text_regex = None
    target_field = None
    
    for field in ("description", "category", "title", "name"):
        val = mongo_filter.get(field)
        if isinstance(val, dict) and "$regex" in val:
            text_regex = val
            target_field = field
            break

    if not text_regex or not target_field:
        return mongo_filter

    remaining = {k: v for k, v in mongo_filter.items() if k != target_field}
    or_clause = [
        {"name": text_regex},
        {"description": text_regex},
        {"category": text_regex}
    ]

    existing_or = remaining.pop("$or", None)
    if isinstance(existing_or, list) and existing_or:
        or_clause = existing_or + or_clause

    text_filter = {"$or": or_clause}
    if not remaining:
        return text_filter
    return {"$and": [remaining, text_filter]}


def _validate_filter_fields(mongo_filter: dict, valid_fields: set) -> dict:
    """
    Valida y limpia un filtro MongoDB eliminando campos que no existen en el schema.
    Mantiene los operadores MongoDB ($or, $and, $regex, etc).
    """
    if not isinstance(mongo_filter, dict) or not mongo_filter:
        return mongo_filter
    
    result = {}
    for key, value in mongo_filter.items():
        # Mantener operadores MongoDB (comienzan con $)
        if key.startswith("$"):
            if key == "$or" and isinstance(value, list):
                cleaned_or = []
                for item in value:
                    cleaned = _validate_filter_fields(item, valid_fields)
                    if cleaned:
                        cleaned_or.append(cleaned)
                if cleaned_or:
                    result["$or"] = cleaned_or
            elif key == "$and" and isinstance(value, list):
                cleaned_and = []
                for item in value:
                    cleaned = _validate_filter_fields(item, valid_fields)
                    if cleaned:
                        cleaned_and.append(cleaned)
                if cleaned_and:
                    result["$and"] = cleaned_and
            else:
                result[key] = value
        # Validar campos reales
        elif key in valid_fields:
            result[key] = value
        # Ignorar campos no válidos (como 'nombre' o 'org_id' en courseattendees)
    
    return result


async def _fetch_related_async(
    docs: list[dict],
    conn: dict,
    cached: dict | None,
    query_hint: str,
    org_id: str = "",
) -> str:
    RELATIONS = {
        "eventId":  "events",
        "event_id": "events",
        "memberId": "members",
        "speakerId": "speakers",
    }

    # Agrupar todos los IDs por colección destino
    col_ids: dict[str, list[str]] = {}
    for doc in docs:
        for field, target_col in RELATIONS.items():
            raw_id = doc.get(field)
            if not raw_id:
                continue
            if target_col not in col_ids:
                col_ids[target_col] = []
            if raw_id not in col_ids[target_col]:
                col_ids[target_col].append(raw_id)

    if not col_ids:
        return ""

    # Buscar todos los IDs de cada colección en paralelo
    tasks = []
    labels = []
    for target_col, ids in col_ids.items():
        schema_fields = list(
            (cached or {}).get("collections", {})
            .get(target_col, {}).get("schema", {}).keys()
        )
        tasks.append(_fetch_related_col_multi(conn, target_col, ids, schema_fields))
        labels.append(target_col)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    parts = []
    for label, result in zip(labels, results):
        if isinstance(result, list) and result:
            parts.append(docs_to_context(label, result, org_id))
            print(f"[chat] enriched {label} ({len(result)} docs)")
    return "\n\n".join(parts)


async def _fetch_related_col_multi(
    conn: dict, collection: str, ids: list[str], schema_fields: list[str]
) -> list[dict]:
    """Busca múltiples documentos relacionados por lista de IDs."""
    from app.db.mongo_query import _serialize, MAX_DOCS
    from app.db.mongo_pool import get_db
    from bson import ObjectId

    def _query():
        try:
            col = get_db(conn["uri"], conn["database"])[collection]
            try:
                oids = [ObjectId(i) for i in ids]
                docs = list(col.find({"_id": {"$in": oids}}, limit=MAX_DOCS))
                if docs:
                    return [_serialize(d) for d in docs]
            except Exception:
                pass
            docs = list(col.find({"_id": {"$in": ids}}, limit=MAX_DOCS))
            if docs:
                return [_serialize(d) for d in docs]
            if "event_id" in schema_fields:
                docs = list(col.find({"event_id": {"$in": ids}}, limit=MAX_DOCS))
                return [_serialize(d) for d in docs]
            return []
        except Exception as e:
            print(f"[chat] _fetch_related_col_multi error {collection}: {e}")
            return []

    return await asyncio.to_thread(_query)


async def _fetch_related_col(
    conn: dict, collection: str, raw_id: str, schema_fields: list[str]
) -> list[dict]:
    return await _fetch_related_col_multi(conn, collection, [raw_id], schema_fields)
