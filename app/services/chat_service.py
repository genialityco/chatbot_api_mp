"""
Servicio de chat.
Combina historial de conversación + contexto RAG + datos reales de MongoDB + LLM.
Para Gemini usa el SDK nativo (google-genai) para evitar problemas de serialización.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from app.core.config import get_settings
from app.rag.pipeline import RAGRetriever
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
- Nunca muestres IDs de MongoDB en la respuesta fuera de una URL.
- Cuando muestres un curso de `events`, el ID para la URL es el campo `_id` del evento (NO el userId).
- Cuando muestres un curso de `courseattendees`, el ID para la URL es el campo `event_id`.
- Cuando muestres una actividad, el ID para la URL es el `_id` de la actividad.
- Formato de URLs de GenCampus:
  - Perfil/cursos del usuario: {base_url}/organization/{{org_id}}/profile?tab=courses
  - Curso: {base_url}/organization/{{org_id}}/course/{{_id_del_evento}}
  - Actividad/video: {base_url}/organization/{{org_id}}/activitydetail/{{_id_actividad}}

## Corrección ortográfica
- Si el usuario escribe con faltas de ortografía, errores de tildes o términos médicos/educativos mal escritos, CORRÍGELOS AUTOMÁTICAMENTE en tu respuesta.
- Ejemplos comunes: "cancer" → "cáncer", "mama" → "mamá", "informacion" → "información", "evaluacion" → "evaluación", "hipertension" → "hipertensión", "sintomas" → "síntomas".
- Mantén la corrección natural y no menciones que estás corrigiendo errores.
"""

NO_CONTEXT_TEMPLATE = """{system_prompt}

## Corrección ortográfica
- Si el usuario escribe con faltas de ortografía, errores de tildes o términos médicos/educativos mal escritos, CORRÍGELOS AUTOMÁTICAMENTE en tu respuesta.
- Ejemplos comunes: "cancer" → "cáncer", "mama" → "mamá", "informacion" → "información", "evaluacion" → "evaluación", "hipertension" → "hipertensión", "sintomas" → "síntomas".
- Mantén la corrección natural y no menciones que estás corrigiendo errores.


No se encontró información relevante en la base de datos para esta consulta.
Responde indicando que no tienes datos disponibles sobre este tema.
"""


def build_prompt(
    system_prompt: str,
    data_context: str,
    user_id: str,
    user_name: str | None,
    org_id: str | None,
    history: list[dict],
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
        )
    else:
        system_content = NO_CONTEXT_TEMPLATE.format(system_prompt=system_prompt)

    print(f"[build_prompt] using={'SYSTEM_TEMPLATE' if has_data else 'NO_CONTEXT_TEMPLATE'}")

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
        self.retriever = RAGRetriever(platform_id, org_id)
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

        user_id_field = "user_id" if self.db_connections and self._uses_snake_case() else "userId"

        # Detectar si la pregunta es personal o una consulta general del catálogo
        _personal_kw = [
            "mis ", "mi ", "mío", "mía", "tengo", "he tomado", "he completado",
            "mi progreso", "mis certificados", "mis inscripciones", "estoy inscrito",
            "my ", "i have", "i am enrolled",
        ]
        is_personal = any(kw in message.lower() for kw in _personal_kw)

        if is_personal:
            ctx_parts = [f"{user_id_field}: {user_id}"]
            if user_name:
                ctx_parts.append(f"nombre: {user_name}")
            # NOTA: no incluir org_id en el enriquecimiento — se usa solo para URLs, no para filtros MongoDB
            enriched_message = f"{message} [{', '.join(ctx_parts)}]"
        else:
            # Consulta general — sin contexto de usuario
            enriched_message = message

        # ── Paso 1: RAG + historial en paralelo ──────────────────────────────
        (schema_context, sources), history = await asyncio.gather(
            asyncio.to_thread(self._rag_retrieve, message),
            load_history(self.platform_id, user_id, session_id),
        )

        # ── Paso 2: generar filtro + cargar schema cache en paralelo ─────────
        data_parts: list[str] = []
        if sources and self.db_connections:
            conn = self.db_connections[0]
            primary = sources[0].get("collection", "")

            # Detectar si la pregunta es sobre actividades
            _activity_kw = ["actividad", "actividades", "clase", "clases", "lección", "lecciones", "curso", "cursos"]
            is_activity_question = any(kw in message.lower() for kw in _activity_kw)

            # Forzar transcript_segments si la pregunta es sobre videos
            _video_kw = ["video", "videos", "grabación", "grabacion",
                         "transcripción", "transcripcion", "conferencia", "conferencias"]
            if any(kw in message.lower() for kw in _video_kw):
                primary = "transcript_segments"
            elif is_activity_question and primary == "activities":
                # Si pregunta por actividades y RAG detectó activities, buscar también en transcripts
                pass  # Mantener primary como activities, pero agregar búsqueda en transcripts después

            if primary:
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
                    mongo_filter = await generate_filter_async(enriched_message, primary, schema_fields, enum_values)
                    print(f"[chat] schema_fields={schema_fields} mongo_filter_before={json.dumps(mongo_filter, default=str)}")
                    # Expandir búsquedas de texto a palabras clave individuales
                    mongo_filter = _expand_text_filter_to_keywords(mongo_filter, schema_fields)
                    print(f"[chat] mongo_filter_after_keyword_expansion={json.dumps(mongo_filter, default=str)}")
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
                    docs = await asyncio.to_thread(_run_query, conn, primary, mongo_filter, sort, limit)
                print(f"[chat] primary={primary} docs={len(docs)}")

                if docs:
                    if primary == "transcript_segments":
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
                        data_parts.append(_transcripts_to_context(transcript_docs))
                        print(f"[chat] found {len(transcript_docs)} transcript segments related to activities")

        data_context = "\n\n".join(data_parts) if data_parts else "No se encontraron datos relevantes."
        print(f"[chat] data_context=\n{data_context[:500]}")

        if not data_parts:
            answer_text = (
                "Lo siento, no encontré información en la base de datos sobre ese curso en este momento. "
                "Por favor, intenta con otra búsqueda o consulta algo relacionado."
            )
            print(f"[chat] no data fallback answer_text={answer_text}")
        else:
            # ── Paso 4: invocar LLM ───────────────────────────────────────────────
            messages = build_prompt(self.system_prompt, data_context, user_id, user_name, org_id, history)
            print(f"[chat] prompt_messages={len(messages)}, data_has_content={'No se encontraron' not in data_context}")
            messages.append({"role": "user", "content": message})
            answer_text = await _invoke_llm(messages)
            print(f"[chat] answer_text={answer_text[:200] if answer_text else 'EMPTY'}")

        # ── Paso 5: guardar historial Redis + persistir en MongoDB ───────────
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer_text})
        collection_used = sources[0].get("collection") if sources else None
        asyncio.create_task(save_history(self.platform_id, user_id, session_id, history))
        asyncio.create_task(persist_turn(
            platform_id=self.platform_id,
            user_id=user_id,
            session_id=session_id,
            user_message=message,
            assistant_message=answer_text,
            user_name=user_name,
            org_id=org_id,
            collection_used=collection_used,
            sources=sources,
        ))

        return {
            "answer": answer_text,
            "session_id": session_id,
            "sources": [sources[0]] if sources else [],
            "platform_id": self.platform_id,
            "org_id": self.org_id,
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


async def _extract_topic(query: str) -> str:
    """Extrae el tema principal de búsqueda de la pregunta del usuario."""
    from google import genai
    from google.genai import types
    import warnings

    prompt = (
        f"Extrae el tema o concepto principal que el usuario quiere buscar en videos/clases. "
        f"Responde SOLO con el término o frase de búsqueda, sin explicaciones.\n\n"
        f"Pregunta: \"{query}\"\n\n"
        f"Tema de búsqueda:"
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

def _transcripts_to_context(activities: list[dict]) -> str:
    if not activities:
        return "No se encontraron videos relacionados con ese tema."
    lines = [f"### Videos encontrados ({len(activities)} actividades)\n"]
    for i, act in enumerate(activities, 1):
        lines.append(f"[{i}] Actividad: {act['name_activity']}")
        lines.append(f"    ID: {act['activity_id']}")
        for seg in act["segments"]:
            lines.append(f"    [{seg['startTime']}s] {seg['text'].strip()}")
        lines.append("")
    return "\n".join(lines)


def _run_query(conn: dict, collection: str, mongo_filter: dict, sort: list | None = None, limit: int | None = None) -> list[dict]:
    """Ejecuta el find en MongoDB con el filtro ya generado."""
    from app.db.mongo_query import _serialize, _REQUIRE_FILTER, MAX_DOCS, ReadOnlyCollection
    from pymongo import MongoClient

    if not mongo_filter and collection in _REQUIRE_FILTER:
        print(f"[chat] skipping {collection} — requires filter")
        return []

    client = MongoClient(conn["uri"], serverSelectionTimeoutMS=8000)
    try:
        col = ReadOnlyCollection(client[conn["database"]][collection])
        n = limit or MAX_DOCS
        cursor = col._col.find(mongo_filter, limit=n)
        if sort:
            cursor = cursor.sort(sort)
        docs = [_serialize(doc) for doc in cursor]
        print(f"[mongo_query] collection={collection} filter={mongo_filter} docs={len(docs)}")
        return docs
    except Exception as e:
        print(f"[mongo_query] query error: {e}")
        return []
    finally:
        client.close()


def _expand_text_filter_to_keywords(mongo_filter: dict, valid_fields: list[str]) -> dict:
    """
    Convierte una búsqueda de texto completo en búsqueda por palabras clave individuales.
    Ej: {"description": {"$regex": "cancer de mama"}} 
    -> {"$or": [{"description": {"$regex": "cancer"}}, {"description": {"$regex": "mama"}}]}
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
            # Campo con búsqueda de regex - expandir a palabras clave
            regex_pattern = value["$regex"]
            options = value.get("$options", "")
            
            # Extraer palabras clave (palabras de 4+ caracteres, excluyendo stopwords)
            keywords = _extract_search_keywords(regex_pattern)
            
            if len(keywords) > 1:
                # Múltiples palabras: buscar cada una en el campo actual
                or_conditions = []
                for keyword in keywords:
                    or_conditions.append({key: {"$regex": keyword, "$options": options}})
                result["$or"] = or_conditions
            else:
                # Una sola palabra o frase corta: mantener como está
                result[key] = value
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
    Para GenCampus: si events viene filtrado por `description` con regex,
    ampliar a búsqueda en `description` o `name`.
    """
    if not isinstance(mongo_filter, dict):
        return mongo_filter

    description_filter = mongo_filter.get("description")
    if not (isinstance(description_filter, dict) and "$regex" in description_filter):
        return mongo_filter

    remaining = {k: v for k, v in mongo_filter.items() if k != "description"}
    or_clause = [
        {"description": description_filter},
        {"name": description_filter},
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
    from pymongo import MongoClient
    from bson import ObjectId

    def _query():
        client = MongoClient(conn["uri"], serverSelectionTimeoutMS=8000)
        try:
            col = client[conn["database"]][collection]
            # Intentar con ObjectId
            try:
                oids = [ObjectId(i) for i in ids]
                docs = list(col.find({"_id": {"$in": oids}}, limit=MAX_DOCS))
                if docs:
                    return [_serialize(d) for d in docs]
            except Exception:
                pass
            # Intentar con string _id
            docs = list(col.find({"_id": {"$in": ids}}, limit=MAX_DOCS))
            if docs:
                return [_serialize(d) for d in docs]
            # Intentar con event_id como string (gencampus)
            if "event_id" in schema_fields:
                docs = list(col.find({"event_id": {"$in": ids}}, limit=MAX_DOCS))
                return [_serialize(d) for d in docs]
            return []
        except Exception as e:
            print(f"[chat] _fetch_related_col_multi error {collection}: {e}")
            return []
        finally:
            client.close()

    return await asyncio.to_thread(_query)


async def _fetch_related_col(
    conn: dict, collection: str, raw_id: str, schema_fields: list[str]
) -> list[dict]:
    return await _fetch_related_col_multi(conn, collection, [raw_id], schema_fields)
