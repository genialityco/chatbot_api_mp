"""
Servicio de chat.
Combina historial de conversación + contexto RAG + datos reales de MongoDB + LLM.
Para Gemini usa el SDK nativo (google-genai) para evitar problemas de serialización.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from app.core.config import get_settings
from app.rag.pipeline import RAGRetriever
from app.db.mongo_query import fetch_collection_data, docs_to_context, generate_filter_async
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
"""

NO_CONTEXT_TEMPLATE = """{system_prompt}

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
            if org_id:
                ctx_parts.append(f"org_id: {org_id}")
            enriched_message = f"{message} [{', '.join(ctx_parts)}]"
        else:
            # Consulta general — sin contexto de usuario
            enriched_message = message

        # ── Paso 1: RAG + historial en paralelo ──────────────────────────────
        (schema_context, sources), history = await asyncio.gather(
            asyncio.to_thread(self._rag_retrieve, message),
            load_history(self.platform_id, user_id),
        )

        # ── Paso 2: generar filtro + cargar schema cache en paralelo ─────────
        data_parts: list[str] = []
        if sources and self.db_connections:
            conn = self.db_connections[0]
            primary = sources[0].get("collection", "")

            if primary:
                cached = load_schema_cache(conn["uri"], conn["database"], conn.get("collections"))
                schema_fields = list(
                    (cached or {}).get("collections", {})
                    .get(primary, {}).get("schema", {}).keys()
                )

                # Generar filtro async (llama al LLM en paralelo con nada más por ahora)
                mongo_filter = await generate_filter_async(enriched_message, primary, schema_fields)

                # ── Paso 3: query MongoDB + enriquecimiento en paralelo ───────
                docs = await asyncio.to_thread(
                    _run_query, conn, primary, mongo_filter
                )
                print(f"[chat] primary={primary} docs={len(docs)}")

                if docs:
                    data_parts.append(docs_to_context(primary, docs))
                    related = await _fetch_related_async(docs, conn, cached, enriched_message)
                    if related:
                        data_parts.append(related)

        data_context = "\n\n".join(data_parts) if data_parts else "No se encontraron datos relevantes."
        print(f"[chat] data_context=\n{data_context[:500]}")

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
        asyncio.create_task(save_history(self.platform_id, user_id, history))
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


def _run_query(conn: dict, collection: str, mongo_filter: dict) -> list[dict]:
    """Ejecuta el find en MongoDB con el filtro ya generado."""
    from app.db.mongo_query import _serialize, _REQUIRE_FILTER, MAX_DOCS, ReadOnlyCollection
    from pymongo import MongoClient

    if not mongo_filter and collection in _REQUIRE_FILTER:
        print(f"[chat] skipping {collection} — requires filter")
        return []

    client = MongoClient(conn["uri"], serverSelectionTimeoutMS=8000)
    try:
        col = ReadOnlyCollection(client[conn["database"]][collection])
        docs = [_serialize(doc) for doc in col._col.find(mongo_filter, limit=MAX_DOCS)]
        print(f"[mongo_query] collection={collection} filter={mongo_filter} docs={len(docs)}")
        return docs
    except Exception as e:
        print(f"[mongo_query] query error: {e}")
        return []
    finally:
        client.close()


async def _fetch_related_async(
    docs: list[dict],
    conn: dict,
    cached: dict | None,
    query_hint: str,
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
            parts.append(docs_to_context(label, result))
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
