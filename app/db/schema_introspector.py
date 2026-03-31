"""
Schema introspector para MongoDB.
Extrae estructura, tipos, relaciones y datos de ejemplo
para alimentar el pipeline RAG.

El resultado se cachea en disco (./data/schemas/<hash>.json).
Si el archivo ya existe, se carga directamente sin reconectar a MongoDB.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any
from collections import defaultdict

from pymongo import MongoClient
from pymongo.collection import Collection

from app.core.config import get_settings

settings = get_settings()

SCHEMA_CACHE_DIR = "./data/schemas"


# ─── Type inference ──────────────────────────────────────────────────────────

def _infer_type(value: Any, depth: int = 0) -> str:
    if depth > settings.schema_max_depth:
        return "..."
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        if not value:
            return "array<unknown>"
        inner = _infer_type(value[0], depth + 1)
        return f"array<{inner}>"
    if isinstance(value, dict):
        fields = {k: _infer_type(v, depth + 1) for k, v in value.items()}
        return f"object{json.dumps(fields)}"
    return type(value).__name__


# ─── Per-collection introspection ────────────────────────────────────────────

def _merge_schemas(base: dict, incoming: dict) -> dict:
    """Fusiona dos schemas tomando la unión de campos."""
    merged = dict(base)
    for key, val in incoming.items():
        if key not in merged:
            merged[key] = val
        elif merged[key] != val:
            merged[key] = f"{merged[key]} | {val}"
    return merged


def introspect_collection(
    collection: Collection,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """
    Devuelve un dict con:
      - name: nombre de la colección
      - doc_count: total de documentos
      - schema: mapa campo->tipo inferido
      - sample_documents: lista de docs de ejemplo (sin _id)
      - indexes: lista de índices
      - field_frequency: frecuencia relativa de cada campo
    """
    sample_size = sample_size or settings.schema_sample_docs
    total = collection.estimated_document_count()
    samples = list(collection.aggregate([{"$sample": {"size": sample_size}}]))

    # Construir schema y frecuencia de campos
    field_counts: dict[str, int] = defaultdict(int)
    merged_schema: dict[str, str] = {}

    for doc in samples:
        doc.pop("_id", None)
        doc_schema = {k: _infer_type(v) for k, v in doc.items()}
        merged_schema = _merge_schemas(merged_schema, doc_schema)
        for k in doc:
            field_counts[k] += 1

    field_frequency = {
        k: round(v / len(samples), 2) if samples else 0
        for k, v in field_counts.items()
    }

    # Índices
    indexes = []
    try:
        for idx in collection.list_indexes():
            indexes.append({
                "name": idx.get("name"),
                "keys": dict(idx.get("key", {})),
                "unique": idx.get("unique", False),
            })
    except Exception:
        pass

    # Limpiar samples para texto
    clean_samples = []
    for doc in samples[:3]:
        doc.pop("_id", None)
        # Truncar strings largos
        truncated = {
            k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
            for k, v in doc.items()
        }
        clean_samples.append(truncated)

    return {
        "name": collection.name,
        "doc_count": total,
        "schema": merged_schema,
        "sample_documents": clean_samples,
        "indexes": indexes,
        "field_frequency": field_frequency,
    }


# ─── Schema cache (disco) ────────────────────────────────────────────────────

def _cache_key(uri: str, database: str, collections: list[str] | None) -> str:
    """Hash estable basado en uri+db+colecciones para nombrar el archivo."""
    raw = f"{uri}:{database}:{sorted(collections) if collections else '__all__'}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _cache_path(uri: str, database: str, collections: list[str] | None) -> str:
    key = _cache_key(uri, database, collections)
    safe_db = database.replace("/", "_").replace("\\", "_")
    return os.path.join(SCHEMA_CACHE_DIR, f"{safe_db}__{key}.json")


def load_schema_cache(
    uri: str,
    database: str,
    collections: list[str] | None = None,
) -> dict[str, Any] | None:
    """Devuelve el schema cacheado si existe, None si no."""
    path = _cache_path(uri, database, collections)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_schema_cache(
    db_info: dict[str, Any],
    uri: str,
    collections: list[str] | None = None,
) -> str:
    """Guarda el schema en disco y devuelve la ruta del archivo."""
    os.makedirs(SCHEMA_CACHE_DIR, exist_ok=True)
    path = _cache_path(uri, db_info["database"], collections)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db_info, f, ensure_ascii=False, indent=2, default=str)
    return path


# ─── Full DB introspection ────────────────────────────────────────────────────

def introspect_database(
    uri: str,
    database: str,
    collections: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Introspecciona una base de datos MongoDB completa.

    Si ya existe un schema cacheado en disco y `force=False`, lo devuelve
    directamente sin conectar a MongoDB.

    Devuelve un dict listo para convertirse en documentos RAG.
    """
    if not force:
        cached = load_schema_cache(uri, database, collections)
        if cached:
            return cached

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[database]

    # Usar la lista de colecciones pasada como parámetro, o la global como fallback
    if collections:
        col_names = [c for c in collections if c in db.list_collection_names()]
    else:
        allowed = settings.schema_allowed_collections
        col_names = [c for c in db.list_collection_names() if c in allowed]
    result: dict[str, Any] = {
        "database": database,
        "uri_host": _safe_host(uri),
        "collections": {},
    }

    for col_name in col_names:
        try:
            col_info = introspect_collection(db[col_name])
            result["collections"][col_name] = col_info
        except Exception as exc:
            result["collections"][col_name] = {"error": str(exc)}

    client.close()

    saved_path = save_schema_cache(result, uri, collections)
    print(f"[schema] Schema guardado en: {saved_path}")

    return result


def _safe_host(uri: str) -> str:
    """Extrae solo el host sin credenciales."""
    try:
        from urllib.parse import urlparse
        p = urlparse(uri)
        return p.hostname or uri
    except Exception:
        return "unknown"


# ─── Render as human-readable text ───────────────────────────────────────────

def schema_to_text(db_info: dict[str, Any]) -> list[dict[str, str]]:
    """
    Convierte el dict de introspección en una lista de chunks de texto,
    uno por colección, listos para embeddings.

    Devuelve: [{"text": "...", "source": "db.collection", "doc_type": "schema"}]
    """
    # Descripciones semánticas por colección para mejorar el RAG
    COLLECTION_DESCRIPTIONS = {
        # ACHO / GenLive — eventos presenciales
        "events":     "Catálogo de cursos, eventos y programas educativos disponibles en la plataforma. Usar para buscar cursos por tema, nombre, especialidad o área de conocimiento. En ACHO/GenLive representa eventos presenciales o virtuales.",
        "attendees":  "Asistentes o participantes registrados en los eventos. Contiene datos de registro y asistencia de personas.",
        "speakers":   "Ponentes o conferencistas de los eventos. Contiene perfil, biografía y datos de contacto de los speakers.",
        "agendas":    "Agenda o programa de sesiones dentro de un evento. Contiene el listado de sesiones, horarios y salas de un evento específico.",
        "users":      "Usuarios del sistema con sus credenciales y roles de acceso.",
        "members":    "Miembros o integrantes de organizaciones o grupos dentro de la plataforma.",
        "highlights": "Destacados o contenido relevante que se muestra de forma prominente en la plataforma.",
        "news":       "Noticias o artículos publicados en la plataforma.",
        "posters":    "Pósters o materiales visuales presentados en los eventos.",
        # GenCampus — plataforma educativa
        "activities":        "Actividades o lecciones individuales dentro de un curso. Contiene nombre, descripción, duración y contenido de cada lección o sesión.",
        "courseattendees":   "Inscripciones y progreso de un usuario específico en cursos. Usar SOLO cuando se pregunta por los cursos de UN usuario en particular (mis cursos, mi progreso, mis inscripciones).",
        "modules":           "Módulos o unidades de contenido dentro de un curso. Contiene lecciones, videos y materiales de aprendizaje.",
        "organizations":     "Organizaciones o instituciones registradas en la plataforma educativa.",
        "organizationusers": "Relación entre usuarios y organizaciones. Contiene roles y permisos dentro de cada organización.",
        "quizzes":           "Evaluaciones o cuestionarios asociados a cursos o módulos.",
        "quizattempts":      "Intentos de evaluación realizados por los usuarios. Contiene puntajes, respuestas y resultados.",
    }

    docs = []
    db_name = db_info["database"]
    host = db_info.get("uri_host", "")

    for col_name, col in db_info["collections"].items():
        if "error" in col:
            continue

        description = COLLECTION_DESCRIPTIONS.get(col_name, f"Colección {col_name}.")

        lines = [
            f"## Colección: {col_name}",
            f"Descripción: {description}",
            f"Base de datos: {db_name} | Host: {host}",
            f"Total de documentos: {col['doc_count']:,}",
            "",
            "### Campos disponibles",
        ]

        for field in col["schema"].keys():
            lines.append(f"  - {field}")

        if col.get("indexes"):
            lines += ["", "### Índices"]
            for idx in col["indexes"]:
                keys = ", ".join(f"{k}:{v}" for k, v in idx["keys"].items())
                unique = " (único)" if idx["unique"] else ""
                lines.append(f"  - {idx['name']}: [{keys}]{unique}")

        docs.append({
            "text": "\n".join(lines),
            "source": f"{db_name}.{col_name}",
            "doc_type": "schema",
            "collection": col_name,
            "database": db_name,
        })

    return docs
