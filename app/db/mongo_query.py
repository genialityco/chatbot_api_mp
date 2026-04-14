"""
Ejecuta consultas a MongoDB usando filtros generados por el LLM.
El LLM recibe el esquema de la colección y la pregunta, y devuelve
un filtro MongoDB válido en JSON.

SEGURIDAD: Solo lectura. Escritura/borrado bloqueados.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection

from app.core.config import get_settings

settings = get_settings()

MAX_DOCS = 20

# Colecciones que NO deben consultarse sin un filtro específico
_REQUIRE_FILTER = frozenset([
    # ACHO / GenLive
    "users", "members", "attendees", "posters",
    # GenCampus
    "courseattendees", "organizationusers", "quizattempts",
    "transcript_segments",  # 114k docs — siempre requiere filtro de texto
])

# ─── Bloqueo de escritura ────────────────────────────────────────────────────

_BLOCKED_OPS = frozenset([
    "insert_one", "insert_many", "update_one", "update_many",
    "replace_one", "delete_one", "delete_many", "drop",
    "bulk_write", "find_one_and_delete", "find_one_and_replace",
    "find_one_and_update", "create_index", "drop_index", "drop_indexes",
    "rename",
])


class ReadOnlyCollection:
    def __init__(self, collection: Collection):
        self._col = collection

    def __getattr__(self, name: str):
        if name in _BLOCKED_OPS:
            raise PermissionError(f"Operación '{name}' bloqueada. Conexión de solo lectura.")
        return getattr(self._col, name)


# ─── Serialización ───────────────────────────────────────────────────────────

def _serialize(doc: dict) -> dict:
    result = {}
    for k, v in doc.items():
        if k == "_id":
            # Incluir _id serializado como string para que el LLM pueda usarlo en URLs
            result["_id"] = str(v)
            continue
        if isinstance(v, datetime):
            result[k] = v.isoformat()
        elif type(v).__name__ in ("ObjectId", "Decimal128"):
            result[k] = str(v)
        elif isinstance(v, dict):
            result[k] = _serialize(v)
        elif isinstance(v, list):
            result[k] = [
                _serialize(i) if isinstance(i, dict)
                else str(i) if type(i).__name__ == "ObjectId"
                else i
                for i in v
            ]
        else:
            result[k] = v
    return result


# ─── Conversión de filtro LLM a tipos MongoDB ────────────────────────────────

_OBJECTID_RE = re.compile(r"^[a-f0-9]{24}$", re.IGNORECASE)
_DATE_FIELDS = re.compile(r"(date|Date|fecha|Fecha|at|At|time|Time|createdAt|updatedAt)", re.IGNORECASE)

# Campos que son strings aunque parezcan ObjectId (por plataforma)
_STRING_ID_FIELDS = frozenset(["user_id", "event_id", "type_id", "rol_id", "position_id"])


def _try_parse_date(value: str) -> datetime | None:
    """Intenta parsear un string como fecha, devuelve None si falla."""
    # Normalizar Z → +00:00 para fromisoformat
    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    # Formatos adicionales
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(value.rstrip("Z"), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _cast_filter(value: Any, key: str = "") -> Any:
    """Convierte strings que parecen ObjectId o fechas ISO a sus tipos nativos."""
    if isinstance(value, str):
        is_id_field = key.endswith("Id") or key.endswith("_id") or key == "_id"
        is_string_id = key in _STRING_ID_FIELDS  # campos que son string aunque parezcan ObjectId

        # Solo convertir a ObjectId si es campo de ID y NO está en la lista de strings
        if is_id_field and not is_string_id and _OBJECTID_RE.match(value):
            return ObjectId(value)
        # Fechas
        is_date_field = bool(_DATE_FIELDS.search(key))
        if is_date_field or "T" in value or (len(value) == 10 and value[4] == "-"):
            parsed = _try_parse_date(value)
            if parsed:
                return parsed
    if isinstance(value, dict):
        return {k: _cast_filter(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_cast_filter(i, key) for i in value]
    return value


def _prepare_filter(raw: dict) -> dict:
    """Aplica _cast_filter recursivamente al filtro generado por el LLM."""
    return {k: _cast_filter(v, k) for k, v in raw.items()}


# ─── Generación de filtro con LLM ────────────────────────────────────────────

_FILTER_PROMPT = """Eres un experto en MongoDB. Tu tarea es generar un filtro de consulta MongoDB en JSON.

Fecha y hora actual: {now}

Colección: `{collection}`
Campos disponibles (usa EXACTAMENTE estos nombres, solo estos): {fields}

Pregunta del usuario: "{question}"

INSTRUCCIONES DE CORRECCIÓN ORTOGRÁFICA:
- Si la pregunta del usuario contiene faltas de ortografía, errores de tildes o términos médicos/educativos mal escritos, CORRÍGELOS AUTOMÁTICAMENTE antes de generar el filtro.
- Ejemplos comunes: "cancer" → "cáncer", "mama" → "mamá", "informacion" → "información", "evaluacion" → "evaluación", "hipertension" → "hipertensión", "sintomas" → "síntomas", "endocrinologia" → "endocrinología".
- Mantén el significado original pero corrige errores comunes de escritura.
- Aplica la corrección antes de procesar la pregunta para generar el filtro.

Responde ÚNICAMENTE con un objeto JSON válido que represente el filtro MongoDB.
RESTRICCIONES CRÍTICAS:
- SOLO usa campos de la lista "Campos disponibles". NO inventes ni uses campos que no estén listados.
- Si la pregunta menciona un campo que no está disponible, ignóralo.
- IMPORTANTE: usa los nombres de campo EXACTAMENTE como aparecen en "Campos disponibles".
- Si el campo es user_id (con guión bajo), usa "user_id" — NO "userId".
- Si el campo es event_id, usa "event_id" — NO "eventId".
- Usa operadores como $gte, $lte, $in, $regex cuando sea necesario.
- Para fechas usa formato ISO 8601.
- Los campos que terminan en "Id" (camelCase) son ObjectId — usa el string hex de 24 chars.
- Los campos con guión bajo como user_id, event_id son strings — NO los conviertas a ObjectId.
- Si la pregunta contiene "_id: <valor>", genera {{"_id": "<valor>"}}.
- NO inventes valores para campos enum — usa solo los valores que aparecen en la descripción del campo.
- Si la pregunta pide "último", "más reciente", "más nuevo" o "último disponible", responde con {{}} (sin filtro).
- Si no se necesita filtro, responde con {{}}.
- NO incluyas explicaciones, solo el JSON.

Ejemplos:
- campos: [startDate, name] + "eventos próximos" → {{"startDate": {{"$gte": "{now_date}"}}}}
- campos: [user_id, event_id] + "mis cursos user_id 6625bf2f" → {{"user_id": "6625bf2f8315f2e5d60ab7a2"}}
- campos: [userId, eventId] + "userId 672545c7" → {{"userId": "672545c7778fcbf45a1f2c83"}}
- campos: [user_id, event_id, status] + "mis cursos user_id xxx nombre ACE" → {{"user_id": "xxx"}} (ignora "nombre" porque no existe)
- "último curso disponible" → {{}}
- "curso más reciente" → {{}}
"""


def _generate_filter_with_llm(question: str, collection: str, fields: list[str]) -> dict:
    """Usa el LLM configurado para generar el filtro MongoDB (síncrono)."""
    from app.services.chat_service import get_llm

    now = datetime.now(timezone.utc)

    # Si no hay campos del schema, usar campos comunes por colección
    if not fields:
        if collection == "events":
            fields = ["name", "description", "startDate", "endDate", "location", "category", "tags"]
        elif collection == "activities":
            fields = ["name", "description", "content", "type", "duration", "order"]
        elif collection == "courseattendees":
            fields = ["user_id", "event_id", "status", "progress", "enrolledAt", "completedAt"]
        elif collection == "transcript_segments":
            fields = ["video_id", "text", "start_time", "end_time", "speaker"]
        elif collection == "users":
            fields = ["_id", "name", "email", "role", "organizationId"]
        else:
            # Campos genéricos para cualquier colección
            fields = ["name", "title", "description", "text", "content", "status", "createdAt", "updatedAt"]

    prompt = _FILTER_PROMPT.format(
        now=now.isoformat(),
        now_date=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        collection=collection,
        fields=", ".join(fields),
        question=question,
    )

    llm = get_llm(temperature=0)
    response = llm.invoke(prompt)
    raw_text = response.content.strip()

    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not json_match:
        return {}

    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return {}


async def generate_filter_async(question: str, collection: str, fields: list[str], enum_values: dict | None = None) -> dict:
    """Genera el filtro MongoDB usando el LLM configurado de forma async."""
    now = datetime.now(timezone.utc)

    # Si no hay campos del schema, usar campos comunes por colección
    if not fields:
        if collection == "events":
            fields = ["name", "description", "startDate", "endDate", "location", "category", "tags"]
        elif collection == "activities":
            fields = ["name", "description", "content", "type", "duration", "order"]
        elif collection == "courseattendees":
            fields = ["user_id", "event_id", "status", "progress", "enrolledAt", "completedAt"]
        elif collection == "transcript_segments":
            fields = ["video_id", "text", "start_time", "end_time", "speaker"]
        elif collection == "users":
            fields = ["_id", "name", "email", "role", "organizationId"]
        else:
            # Campos genéricos para cualquier colección
            fields = ["name", "title", "description", "text", "content", "status", "createdAt", "updatedAt"]

    # Construir descripción de campos con valores enum si existen
    fields_desc = []
    for f in fields:
        if enum_values and f in enum_values:
            fields_desc.append(f"{f} (valores: {', '.join(repr(v) for v in enum_values[f])})")
        else:
            fields_desc.append(f)

    prompt = _FILTER_PROMPT.format(
        now=now.isoformat(),
        now_date=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        collection=collection,
        fields=", ".join(fields_desc),
        question=question,
    )

    try:
        if settings.llm_provider == "gemini":
            import warnings
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=settings.gemini_api_key)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
                response = await client.aio.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0),
                )
            raw_text = response.text.strip()
        else:
            from app.services.chat_service import get_llm
            llm = get_llm(temperature=0)
            response = await llm.ainvoke(prompt)
            raw_text = response.content.strip()

        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            return {}
        raw = json.loads(json_match.group())
        result = _prepare_filter(raw)
        print(f"[mongo_query] llm_filter={json.dumps(raw, default=str)}")
        return result
    except Exception as e:
        print(f"[mongo_query] llm_filter_error={e}")
        return {}


# ─── API pública ─────────────────────────────────────────────────────────────

def fetch_collection_data(
    uri: str,
    database: str,
    collection: str,
    query_hint: str = "",
    schema_fields: list[str] | None = None,
) -> list[dict]:
    """
    Trae documentos reales de una colección.
    El LLM genera el filtro MongoDB basándose en la pregunta y el esquema.
    """
    allowed = settings.schema_allowed_collections
    if collection not in allowed:
        raise PermissionError(f"Colección '{collection}' no permitida.")

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    db = client[database]
    col = ReadOnlyCollection(db[collection])

    mongo_filter: dict = {}
    sort = None

    # Generar filtro con LLM si hay pregunta y campos disponibles
    if query_hint and schema_fields:
        try:
            raw_filter = _generate_filter_with_llm(query_hint, collection, schema_fields)
            print(f"[mongo_query] llm_filter={json.dumps(raw_filter, default=str)}")
            mongo_filter = _prepare_filter(raw_filter)
        except Exception as e:
            print(f"[mongo_query] llm_filter_error={e}")
            mongo_filter = {}

    print(f"[mongo_query] collection={collection} filter={json.dumps(mongo_filter, default=str)}")
    print(f"[mongo_query] filter_types={ {k: type(v).__name__ for k, v in mongo_filter.items()} }")

    # Colecciones sensibles: no consultar sin filtro
    if not mongo_filter and collection in _REQUIRE_FILTER:
        print(f"[mongo_query] skipping {collection} — requires filter")
        client.close()
        return []

    try:
        cursor = col._col.find(mongo_filter, limit=MAX_DOCS)
        docs = [_serialize(doc) for doc in cursor]
        print(f"[mongo_query] docs_found={len(docs)}")
    except Exception as e:
        print(f"[mongo_query] filter failed ({e}), retrying without filter")
        try:
            docs = [_serialize(doc) for doc in col._col.find({}, limit=MAX_DOCS)]
            print(f"[mongo_query] docs_found={len(docs)} (no filter)")
        except Exception:
            docs = []
    finally:
        client.close()

    return docs


def docs_to_context(collection: str, docs: list[dict], org_id: str = "") -> str:
    if not docs:
        return f"No se encontraron documentos en `{collection}`."
    lines = [f"### Datos reales de `{collection}` ({len(docs)} documentos)\n"]
    base = settings.gencampus_base_url
    for i, doc in enumerate(docs, 1):
        clean = {k: v for k, v in doc.items()
                 if not (isinstance(v, str) and len(v) == 24 and v.isalnum() and v != doc.get("_id"))}
        # Construir URL directamente para evitar que el LLM confunda IDs
        url = None
        if collection == "events" and doc.get("_id") and org_id:
            url = f"{base}/organization/{org_id}/course/{doc['_id']}"
        elif collection == "activities" and doc.get("_id") and org_id:
            url = f"{base}/organization/{org_id}/activitydetail/{doc['_id']}"

        entry = {"name": doc.get("name", ""), **{k: v for k, v in doc.items() if k not in ("_id",)}}
        if url:
            entry["url"] = url
        lines.append(f"[{i}] {json.dumps(entry, ensure_ascii=False, default=str)}")
    return "\n".join(lines)


def search_transcript_segments(
    uri: str,
    database: str,
    topic: str,
    max_activities: int = 5,
    segments_per_activity: int = 3,
) -> list[dict]:
    """
    Busca segmentos de transcripción que contengan el tema indicado.
    Agrupa por activity_id y devuelve los videos más relevantes con fragmentos de contexto.
    """
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    try:
        col = client[database]["transcript_segments"]
        # Buscar segmentos que mencionen el tema
        cursor = col.find(
            {"text": {"$regex": topic, "$options": "i"}},
            {"text": 1, "activity_id": 1, "name_activity": 1, "startTime": 1},
            limit=200,
        )

        # Agrupar por actividad
        activities: dict[str, dict] = {}
        for doc in cursor:
            aid = str(doc.get("activity_id", ""))
            name = doc.get("name_activity", "Sin nombre")
            if aid not in activities:
                activities[aid] = {
                    "activity_id": aid,
                    "name_activity": name,
                    "segments": [],
                }
            if len(activities[aid]["segments"]) < segments_per_activity:
                activities[aid]["segments"].append({
                    "text": doc.get("text", ""),
                    "startTime": doc.get("startTime"),
                })

        # Devolver las primeras N actividades
        result = list(activities.values())[:max_activities]
        print(f"[transcript] topic='{topic}' activities_found={len(result)}")
        return result
    except Exception as e:
        print(f"[transcript] search error: {e}")
        return []
    finally:
        client.close()
