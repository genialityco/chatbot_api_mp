"""
Ejecuta consultas a Firestore usando filtros generados por el LLM o heurísticas.
Esto permite consultar datos en tiempo real (como disponibilidad de agenda, estado de reuniones)
que podrían estar desactualizados en el RAG.
"""
from __future__ import annotations

import json
from typing import Any
from google.cloud.firestore_v1.base_query import FieldFilter

from app.core.config import get_settings

settings = get_settings()

def _get_firestore_client():
    from app.rag.content_indexer import _init_firestore_client
    return _init_firestore_client()

async def generate_firestore_filter_async(
    query: str,
    collection_name: str,
) -> list[tuple[str, str, Any]]:
    """
    Usa el LLM para generar una lista de filtros simples de Firestore
    en base a la pregunta del usuario.
    Devuelve lista de tuplas (campo, operador, valor).
    Operadores soportados: '==', '<', '<=', '>', '>=', 'in', 'array_contains'
    """
    from google import genai
    from google.genai import types
    import warnings

    prompt = (
        f"Eres un experto en Firestore. Debes generar filtros de consulta para la colección '{collection_name}' "
        f"en base a la pregunta del usuario.\n"
        f"Firestore soporta operadores: ==, <, <=, >, >=, in, array_contains.\n"
        f"Devuelve SOLO un array JSON válido con listas de 3 elementos: [campo, operador, valor].\n"
        f"Si no hay filtros obvios, devuelve [].\n\n"
        f"Pregunta: {query}\n\n"
        f"Filtros JSON:"
    )

    client = genai.Client(api_key=settings.gemini_api_key)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            response = await client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
        
        text = response.text.strip()
        # Limpiar markdown
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        filters = json.loads(text.strip())
        if isinstance(filters, list):
            return [tuple(f) for f in filters if len(f) == 3]
        return []
    except Exception as e:
        print(f"[firestore_query] Error generating filter: {e}")
        return []

async def fetch_firestore_data(
    event_id: str,
    collection: str,
    filters: list[tuple[str, str, Any]],
    user_id: str | None = None
) -> list[dict[str, Any]]:
    """
    Obtiene datos de Firestore aplicando los filtros.
    Maneja las subcolecciones de eventos.
    """
    db = _get_firestore_client()
    
    # Determinar ruta de la colección
    if collection in ("companies", "products", "meetings", "agenda"):
        if not event_id:
            raise ValueError(f"La colección {collection} requiere event_id")
        col_ref = db.collection(f"events/{event_id}/{collection}")
    elif collection in ("users", "events"):
        col_ref = db.collection(collection)
        if collection == "users" and event_id:
            # Forzar filtro de evento si consultamos usuarios
            col_ref = col_ref.where(filter=FieldFilter("eventId", "==", event_id))
    else:
        return []

    docs = []
    
    # Caso especial: Si consultamos 'events' y tenemos un event_id específico, 
    # solo obtenemos ese documento, no todos.
    if collection == "events" and event_id:
        try:
            doc_snap = db.collection("events").document(event_id).get()
            if doc_snap.exists:
                d = doc_snap.to_dict() or {}
                d["id"] = doc_snap.id
                docs.append(d)
        except Exception as e:
            print(f"[firestore_query] Error obteniendo evento {event_id}: {e}")
        return docs
    
    # Heurística: si consultan reuniones y tenemos user_id y no hay filtros, 
    # es una consulta personal (is_personal). Extraemos SOLO sus reuniones.
    if collection == "meetings" and user_id and not filters:
        try:
            my_reqs = col_ref.where(filter=FieldFilter("requesterId", "==", user_id)).stream()
            my_recs = col_ref.where(filter=FieldFilter("receiverId", "==", user_id)).stream()
            
            for doc in my_reqs:
                d = doc.to_dict() or {}
                d["id"] = doc.id
                docs.append(d)
                    
            for doc in my_recs:
                d = doc.to_dict() or {}
                d["id"] = doc.id
                if d["id"] not in [x.get("id") for x in docs]:
                    docs.append(d)
        except Exception as e:
            print(f"[firestore_query] Error en heurística de meetings: {e}")
        return docs

    # Aplicar filtros generados (con cuidado, Firestore limita múltiples campos de desigualdad)
    query = col_ref
    for field, op, value in filters:
        try:
            query = query.where(filter=FieldFilter(field, op, value))
        except Exception as e:
            print(f"[firestore_query] Filtro ignorado ({field} {op} {value}): {e}")

    try:
        stream = query.stream()
        for doc in stream:
            d = doc.to_dict() or {}
            d["id"] = doc.id
            docs.append(d)
    except Exception as e:
        print(f"[firestore_query] Error ejecutando query en {collection}: {e}")

    return docs

async def get_related_firestore_context(event_id: str, collection: str, docs: list[dict[str, Any]]) -> str:
    """Busca en vivo la información de los usuarios/compañías referenciados en los documentos."""
    if not docs:
        return ""
        
    db = _get_firestore_client()
    related_lines = []
    
    if collection == "meetings":
        user_ids = set()
        company_ids = set()
        for d in docs:
            if d.get("requesterId"): user_ids.add(d["requesterId"])
            if d.get("receiverId"): user_ids.add(d["receiverId"])
            if d.get("companyId"): company_ids.add(d["companyId"])
            
        if user_ids:
            related_lines.append("### INFORMACIÓN DE LOS USUARIOS EN ESTAS REUNIONES:")
            for uid in list(user_ids)[:15]: # límite para no saturar firebase
                u_data = None
                try:
                    # Intenta por document ID
                    u_doc = db.collection("users").document(uid).get()
                    if u_doc.exists:
                        u_data = u_doc.to_dict()
                    else:
                        # Si no, por el campo ID o userId
                        snaps = db.collection("users").where(filter=FieldFilter("id", "==", uid)).limit(1).stream()
                        for s in snaps: 
                            u_data = s.to_dict()
                        if not u_data:
                            snaps2 = db.collection("users").where(filter=FieldFilter("userId", "==", uid)).limit(1).stream()
                            for s in snaps2:
                                u_data = s.to_dict()
                except Exception:
                    pass
                
                if u_data:
                    related_lines.append(f"- Usuario ID: {uid} | Nombre: {u_data.get('nombre', u_data.get('name', 'No disponible'))} | Empresa: {u_data.get('empresa', 'No disponible')} | Cargo: {u_data.get('cargo', u_data.get('role', 'No disponible'))} | Tel: {u_data.get('telefono', u_data.get('phone', ''))}")
        
        if company_ids:
            related_lines.append("\n### INFORMACIÓN DE LAS EMPRESAS/COMPAÑÍAS:")
            for cid in list(company_ids)[:10]:
                try:
                    c_doc = db.collection(f"events/{event_id}/companies").document(cid).get()
                    if c_doc.exists:
                        c_data = c_doc.to_dict()
                        related_lines.append(f"- Compañía ID: {cid} | Nombre: {c_data.get('razonSocial', c_data.get('nombre', 'No disponible'))} | Mesa Asignada: {c_data.get('fixedTable', 'No tiene')} | Info: {c_data.get('descripcion', '')[:150]}")
                except Exception:
                    pass

    elif collection == "agenda":
        meeting_ids = set()
        for d in docs:
            if d.get("meetingId"): meeting_ids.add(d["meetingId"])
            
        if meeting_ids:
            related_lines.append("### REUNIONES EN ESTA AGENDA:")
            for mid in list(meeting_ids)[:10]:
                try:
                    m_doc = db.collection(f"events/{event_id}/meetings").document(mid).get()
                    if m_doc.exists:
                        m_data = m_doc.to_dict()
                        related_lines.append(f"- Reunión ID: {mid} | Solicitante: {m_data.get('requesterId')} | Recibe: {m_data.get('receiverId')} | Estado: {m_data.get('status')}")
                except Exception:
                    pass

    return "\n".join(related_lines)

def firestore_docs_to_context(collection: str, docs: list[dict[str, Any]]) -> str:
    """Convierte los documentos a texto para el contexto."""
    if not docs:
        return ""
    
    lines = [f"### RESULTADOS EN TIEMPO REAL ({collection.upper()}):"]
    for i, doc in enumerate(docs[:20], 1): # Limitar a 20 docs
        lines.append(f"\n--- Documento {i} ---")
        for k, v in doc.items():
            if k not in ("id",) and v is not None:
                lines.append(f"{k}: {v}")
    
    return "\n".join(lines)
