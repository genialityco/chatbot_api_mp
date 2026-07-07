"""
Extractor de contenido para GenCampus.
Extrae cursos, módulos, actividades y transcripciones, genera resúmenes
por medio del LLM y los prepara como documentos para indexar en el RAG.
"""
import asyncio
import os
from typing import Any
from pymongo import MongoClient
from langchain_core.documents import Document

from app.core.config import get_settings
from app.rag.pipeline import _build_vector_store, _prepare_documents, get_embeddings

settings = get_settings()


def _init_firestore_client():
    """Inicializa el cliente de Firestore usando credenciales de service account."""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError:
        raise ImportError("firebase-admin no está instalado. Instala: pip install firebase-admin")

    creds_path = settings.networking_firestore_credentials_path
    if not creds_path or not os.path.exists(creds_path):
        raise FileNotFoundError(
            f"Archivo de credenciales no encontrado: {creds_path}. "
            f"Asegúrate de que NETWORKING_FIRESTORE_CREDENTIALS_PATH está configurado."
        )

    # Verificar si ya existe una app de Firebase inicializada
    if not firebase_admin._apps:
        cred = credentials.Certificate(creds_path)
        firebase_admin.initialize_app(cred)
        print("[_init_firestore_client] Firebase inicializado correctamente.")
    else:
        print("[_init_firestore_client] Firebase ya estaba inicializado.")
        
    return firestore.client()


async def _fetch_firestore_collection(db, collection_path: str) -> dict[str, Any]:
    """Obtiene todos los documentos de una colección o subcolección de Firestore."""
    docs = {}
    for doc in db.collection(collection_path).stream():
        docs[doc.id] = doc.to_dict() or {}
    return docs


def _format_policy_text(policies: dict[str, Any]) -> str:
    if not isinstance(policies, dict) or not policies:
        return ""
    lines = []
    for key, value in policies.items():
        if isinstance(value, dict):
            lines.append(f"- {key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"    - {sub_key}: {sub_value}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _format_dict_lines(item: dict[str, Any]) -> str:
    lines = []
    for key, value in item.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  - {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _networking_event_to_documents(
    event_id: str,
    event_data: dict[str, Any],
    users: list[dict[str, Any]],
) -> list[Document]:
    print(f"[_networking_event_to_documents] Extrayendo datos del evento {event_id}...")
    documents: list[Document] = []
    
    # Extraer datos principales del evento
    event_name = event_data.get("eventName") or event_data.get("name") or "Evento sin nombre"
    
    # Extraer campos de config
    config = event_data.get("config", {})
    event_dates = config.get("eventDates") or event_data.get("eventDates") or []
    event_start_time = config.get("eventStartTime") or event_data.get("eventStartTime") or ""
    event_end_time = config.get("eventEndTime") or event_data.get("eventEndTime") or ""
    event_location = config.get("eventLocation") or event_data.get("eventLocation") or ""
    
    policies = event_data.get("policies") or {}
    companies = event_data.get("companies") or {}
    products = event_data.get("products") or {}
    meetings = event_data.get("meetings") or {}
    agenda = event_data.get("agenda") or {}

    summary_lines = [
        f"EVENTO: {event_name}",
        f"ID DEL EVENTO: {event_id}",
        f"UBICACIÓN: {event_location if event_location else 'No disponible'}",
        f"FECHAS: {', '.join(event_dates) if event_dates else 'No disponible'}",
        f"HORA DE INICIO: {event_start_time if event_start_time else 'No disponible'}",
        f"HORA DE FIN: {event_end_time if event_end_time else 'No disponible'}",
        "",
       
    ]

    if companies:
        summary_lines.append("COMPAÑÍAS:")
        for company_id, company in companies.items():
            summary_lines.append(
                f"- {company.get('razonSocial', company.get('nombre', company_id))} "
                f"(NIT: {company.get('nitNorm', company.get('company_nit', company_id))})"
            )
    else:
        summary_lines.append("No hay compañías registradas en este evento.")

    if products:
        summary_lines.append("\nPRODUCTOS:")
        for product_id, product in products.items():
            summary_lines.append(
                f"- {product.get('title', product.get('name', product_id))}: "
                f"{product.get('description', '')}"
            )

    if meetings:
        summary_lines.append("\nREUNIONES:")
        for meeting_id, meeting in meetings.items():
            summary_lines.append(
                f"- {meeting.get('meetingDate', '')} {meeting.get('timeSlot', '')} | "
                f"{meeting.get('requesterId', '')} → {meeting.get('receiverId', '')} | "
                f"Estado: {meeting.get('status', '')} | Mesa: {meeting.get('tableAssigned', 'N/A')}"
            )

    if agenda:
        summary_lines.append("\nAGENDA:")
        for slot_id, slot in agenda.items():
            summary_lines.append(
                f"- {slot.get('date', '')} {slot.get('startTime', '')}-{slot.get('endTime', '')} | "
                f"Mesa: {slot.get('tableNumber', slot.get('tableAssigned', 'N/A'))} | "
                f"Disponible: {slot.get('available', False)}"
            )

    event_summary = Document(
        page_content="\n".join(summary_lines),
        metadata={
            "doc_type": "event_summary",
            "collection": "events",
            "event_id": event_id,
            "name": event_name,
        },
    )
    documents.append(event_summary)

    for company_id, company in companies.items():
        documents.append(Document(
            page_content=(
                f"COMPAÑÍA: {company.get('razonSocial', company.get('nombre', company_id))}\n"
                f"NIT: {company.get('nitNorm', company.get('company_nit', ''))}\n"
                f"DESCRIPCIÓN: {company.get('descripcion', '')}\n"
                f"LOGO: {company.get('logoUrl', '')}\n"
                f"MESA FIJA: {company.get('fixedTable', '')}\n"
            ),
            metadata={
                "doc_type": "company",
                "collection": "companies",
                "event_id": event_id,
                "company_id": company_id,
                "name": company.get('razonSocial', company.get('nombre', company_id)),
            },
        ))

    for product_id, product in products.items():
        documents.append(Document(
            page_content=(
                f"PRODUCTO: {product.get('title', product.get('name', product_id))}\n"
                f"DESCRIPCIÓN: {product.get('description', '')}\n"
                f"EMPRESA: {product.get('companyId', '')}\n"
                f"PROPIETARIO: {product.get('ownerUserId', '')}\n"
            ),
            metadata={
                "doc_type": "product",
                "collection": "products",
                "event_id": event_id,
                "product_id": product_id,
                "company_id": product.get('companyId', ''),
                "name": product.get('title', product.get('name', product_id)),
            },
        ))

    for meeting_id, meeting in meetings.items():
        documents.append(Document(
            page_content=(
                f"REUNIÓN: {meeting_id}\n"
                f"SOLICITANTE: {meeting.get('requesterId', '')}\n"
                f"RECEPCIONISTA: {meeting.get('receiverId', '')}\n"
                f"FECHA: {meeting.get('meetingDate', '')}\n"
                f"HORA: {meeting.get('timeSlot', '')}\n"
                f"ESTADO: {meeting.get('status', '')}\n"
                f"MESA: {meeting.get('tableAssigned', '')}\n"
                f"PRODUCTO: {meeting.get('productId', '')}\n"
                f"COMPAÑÍA: {meeting.get('companyId', '')}\n"
                f"NOTA: {meeting.get('contextNote', '')}\n"
            ),
            metadata={
                "doc_type": "meeting",
                "collection": "meetings",
                "event_id": event_id,
                "meeting_id": meeting_id,
                "name": meeting.get('contextNote', meeting_id),
            },
        ))

    for slot_id, slot in agenda.items():
        documents.append(Document(
            page_content=(
                f"FRANJA: {slot.get('date', '')} {slot.get('startTime', '')}-{slot.get('endTime', '')}\n"
                f"MESA: {slot.get('tableNumber', slot.get('tableAssigned', ''))}\n"
                f"DISPONIBLE: {slot.get('available', False)}\n"
                f"REUNIÓN: {slot.get('meetingId', '')}\n"
                f"BREAK: {slot.get('isBreak', False)}\n"
            ),
            metadata={
                "doc_type": "agenda_slot",
                "collection": "agenda",
                "event_id": event_id,
                "slot_id": slot_id,
                "name": f"{slot.get('date', '')} {slot.get('startTime', '')}-{slot.get('endTime', '')}",
            },
        ))

    for user in users:
        user_id = user.get("id") or user.get("userId") or ""
        documents.append(Document(
            page_content=(
                f"USUARIO: {user.get('nombre', user.get('name', ''))}\n"
                f"CORREO: {user.get('correo', user.get('email', ''))}\n"
                f"TELÉFONO: {user.get('telefono', user.get('phone', ''))}\n"
                f"CARGO: {user.get('cargo', user.get('role', ''))}\n"
                f"DESCRIPCIÓN: {user.get('descripcion', '')}\n"
                f"TIPO: {user.get('tipoAsistente', '')}\n"
                f"EMPRESA: {user.get('empresa', '')}\n"
                f"NIT: {user.get('company_nit', '')}\n"
                f"COMPANY_ID: {user.get('companyId', '')}\n"
                f"EVENT_ID: {user.get('eventId', '')}\n"
            ),
            metadata={
                "doc_type": "user",
                "collection": "users",
                "event_id": event_id,
                "user_id": user_id,
                "name": user.get('nombre', user.get('name', user_id)),
            },
        ))

    print(f"[_networking_event_to_documents] Evento {event_id}: procesadas {len(companies)} compañías, {len(products)} productos, {len(meetings)} reuniones, {len(agenda)} slots de agenda, {len(users)} usuarios.")

    return documents


async def build_networking_rag(
    platform_id: str,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Indexa eventos y datos de Firestore para networking."""
    print(f"[build_networking_rag] Iniciando indexación de networking. platform_id={platform_id}, event_id={event_id}")
    try:
        db = _init_firestore_client()
    except Exception as e:
        print(f"[build_networking_rag] Error inicializando Firestore: {str(e)}")
        return {"status": "error", "message": f"Error inicializando Firestore: {str(e)}"}

    try:
        if event_id:
            print(f"[build_networking_rag] Buscando evento específico: {event_id}")
            event_ref = db.collection("events").document(event_id)
            event_doc = event_ref.get()
            if not event_doc.exists:
                print(f"[build_networking_rag] Evento {event_id} no encontrado en Firestore.")
                return {"status": "error", "message": f"Evento {event_id} no encontrado en Firestore."}
            event_data = event_doc.to_dict() or {}

            # Fetch subcollections
            event_data["companies"] = await _fetch_firestore_collection(db, f"events/{event_id}/companies")
            event_data["products"] = await _fetch_firestore_collection(db, f"events/{event_id}/products")
            event_data["meetings"] = await _fetch_firestore_collection(db, f"events/{event_id}/meetings")
            event_data["agenda"] = await _fetch_firestore_collection(db, f"events/{event_id}/agenda")

            print(f"[build_networking_rag] Buscando usuarios para el evento {event_id}...")
            users_snap = db.collection("users").where("eventId", "==", event_id).stream()
            users_for_event = [doc.to_dict() or {} for doc in users_snap]
            print(f"[build_networking_rag] Se encontraron {len(users_for_event)} usuarios.")

            print(f"[build_networking_rag] Generando documentos...")
            documents = _networking_event_to_documents(event_id, event_data, users_for_event)
            raw_docs = [{"text": doc.page_content, **doc.metadata} for doc in documents]
            print(f"[build_networking_rag] Se generaron {len(documents)} documentos. Preparando chunks...")
            
            chunks = _prepare_documents(raw_docs)
            embeddings = get_embeddings()
            print(f"[build_networking_rag] Se obtuvieron {len(chunks)} chunks. Construyendo vector store...")
            
            _build_vector_store(chunks, embeddings, platform_id, event_id, force=True)
            print(f"[build_networking_rag] Vector store construido exitosamente para el evento {event_id}.")
            
            from app.rag.pipeline import invalidate_retriever
            invalidate_retriever(platform_id, event_id)

            return {
                "status": "ready",
                "event_id": event_id,
                "documents_indexed": len(documents),
                "chunks_indexed": len(chunks),
            }

        print(f"[build_networking_rag] Obteniendo todos los eventos y usuarios...")
        events_snap = db.collection("events").stream()
        users_by_event: dict[str, list[dict[str, Any]]] = {}

        all_users_snap = db.collection("users").stream()
        for user_doc in all_users_snap:
            user = user_doc.to_dict() or {}
            eid = str(user.get("eventId") or user.get("event_id") or "")
            if eid:
                users_by_event.setdefault(eid, []).append(user)

        print(f"[build_networking_rag] Usuarios agrupados por evento. Procesando eventos...")

        total_documents = 0
        total_chunks = 0
        event_count = 0
        
        embeddings = get_embeddings()

        for event_doc in events_snap:
            event_data = event_doc.to_dict() or {}
            eid = event_doc.id
            
            # Fetch subcollections for each event
            event_data["companies"] = await _fetch_firestore_collection(db, f"events/{eid}/companies")
            event_data["products"] = await _fetch_firestore_collection(db, f"events/{eid}/products")
            event_data["meetings"] = await _fetch_firestore_collection(db, f"events/{eid}/meetings")
            event_data["agenda"] = await _fetch_firestore_collection(db, f"events/{eid}/agenda")
            
            print(f"[build_networking_rag] Procesando evento {eid}...")
            docs = _networking_event_to_documents(eid, event_data, users_by_event.get(eid, []))
            
            if not docs:
                continue

            raw_docs = [{"text": doc.page_content, **doc.metadata} for doc in docs]
            chunks = _prepare_documents(raw_docs)
            
            print(f"[build_networking_rag] Guardando en vector store del evento {eid} ({len(chunks)} chunks)...")
            _build_vector_store(chunks, embeddings, platform_id, eid, force=True)
            
            from app.rag.pipeline import invalidate_retriever
            invalidate_retriever(platform_id, eid)
            
            total_documents += len(docs)
            total_chunks += len(chunks)
            event_count += 1

        if event_count == 0:
            print(f"[build_networking_rag] No se encontraron documentos para indexar.")
            return {"status": "error", "message": "No se encontraron eventos para indexar en Firestore."}

        print(f"[build_networking_rag] Indexación masiva completada.")

        return {
            "status": "ready",
            "documents_indexed": total_documents,
            "chunks_indexed": total_chunks,
            "events_indexed": event_count,
        }
    except Exception as e:
        print(f"[build_networking_rag] Excepción durante la indexación: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Error indexando Firestore: {str(e)}"}


async def _summarize_course_content(course_name: str, modules_data: list[str]) -> str:
    """Usa el LLM para resumir el contenido completo de un curso."""
    from google import genai
    from google.genai import types
    import warnings

    prompt = (
        f"Eres un experto en currículo educativo. Crea un resumen detallado y muy bien estructurado "
        f"sobre el curso '{course_name}'. A continuación se listan sus módulos y actividades, así como transcripciones. "
        f"Destaca de qué trata el curso, los temas que cubre y el objetivo de aprendizaje.\n\n"
        f"Contenido del curso:\n" + "\n".join(modules_data)
    )

    client = genai.Client(api_key=settings.gemini_api_key)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
    return response.text.strip()


async def build_content_rag(uri: str, database: str, platform_id: str, org_id: str | None = None) -> dict[str, Any]:
    """Extrae los datos reales, los resume e indexa en el RAG. Si se provee org_id, filtra por esa organización."""
    print(f"[build_content_rag] Iniciando indexación de GenCampus. platform_id={platform_id}, org_id={org_id}")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    db = client[database]

    event_query = {}
    if org_id:
        # En GenCampus la organización suele llamarse organizer_id y es un string
        event_query = {"organizer_id": org_id}
            
    print(f"[build_content_rag] Buscando eventos con query: {event_query}...")
    events = list(db["events"].find(event_query))
    
    if not events:
        print(f"[build_content_rag] No se encontraron eventos para org_id {org_id}. Cerrando conexión.")
        client.close()
        return {"status": "ok", "message": f"No se encontraron eventos para org_id {org_id}"}
        
    event_ids_str = [str(ev.get("_id")) for ev in events]
    print(f"[build_content_rag] Se encontraron {len(events)} eventos. Buscando módulos y actividades...")
    
    modules = list(db["modules"].find({"$or": [{"eventId": {"$in": event_ids_str}}, {"event_id": {"$in": event_ids_str}}]}))
    activities = list(db["activities"].find({"$or": [{"eventId": {"$in": event_ids_str}}, {"event_id": {"$in": event_ids_str}}]}))
    
    activity_ids_str = [str(act.get("_id")) for act in activities]
    print(f"[build_content_rag] Se encontraron {len(modules)} módulos y {len(activities)} actividades. Buscando transcripciones...")
    transcripts = list(db["transcript_segments"].find({"activity_id": {"$in": activity_ids_str}}))
    
    doc_query = {"active": True}
    if org_id:
        doc_query["organizer_id"] = org_id
            
    print(f"[build_content_rag] Buscando documentos subidos con query: {doc_query}...")
    raw_documents = list(db["documents"].find(doc_query))

    print(f"[build_content_rag] Resumen de extracción para org_id={org_id}: {len(events)} eventos, {len(modules)} modulos, {len(activities)} actividades, {len(transcripts)} fragmentos transcript, {len(raw_documents)} documentos.")

    # Agrupar por curso
    courses_data = {}
    for ev in events:
        eid = str(ev.get("_id"))
        courses_data[eid] = {
            "name": ev.get("name", ev.get("title", "Curso sin nombre")),
            "description": ev.get("description", ev.get("summary", "")),
            "modules": {}
        }
    
    # Agregar módulos
    for mod in modules:
        eid = str(mod.get("eventId") or mod.get("event_id") or "")
        mid = str(mod.get("_id"))
        if eid and eid in courses_data:
            courses_data[eid]["modules"][mid] = {
                "name": mod.get("name", mod.get("title", "Módulo sin nombre")),
                "activities": []
            }

    # Agregar actividades y fragmentos
    for act in activities:
        eid = str(act.get("eventId") or act.get("event_id") or "")
        raw_mid = act.get("moduleId") or act.get("module_id")
        mid = str(raw_mid) if raw_mid else None

        if not eid or eid not in courses_data:
            continue

        # Si la actividad no tiene módulo, usar un contenedor genérico directo al curso
        if not mid or mid not in courses_data[eid]["modules"]:
            mid = "unassigned"
            if mid not in courses_data[eid]["modules"]:
                courses_data[eid]["modules"][mid] = {"name": "Actividades", "activities": []}

        aid = str(act.get("_id"))
        act_name = act.get("name", act.get("title", "Actividad"))

        # Buscar transcripciones de esta actividad (activity_id puede ser string o ObjectId)
        act_transcripts = [
            t.get("text", "") for t in transcripts
            if str(t.get("activity_id")) == aid
        ]
        transcripts_text = " ".join(act_transcripts)
        if len(transcripts_text) > 2000:
            transcripts_text = transcripts_text[:1997] + "..."

        courses_data[eid]["modules"][mid]["activities"].append({
            "name": act_name,
            "content": act.get("description", act.get("content", "")),
            "transcripts": transcripts_text
        })

    client.close()

    documents = []

    # Procesar cada curso para RAG
    from app.rag.pipeline import _prepare_documents

    # Indexar documentos subidos (PDF, PPT, Word, etc.)
    for raw_doc in raw_documents:
        content = raw_doc.get("content", "").strip()
        if not content:
            continue

        event_id = str(raw_doc["eventId"]) if raw_doc.get("eventId") else ""
        org_id_doc = str(raw_doc.get("organizer_id", ""))

        # Determinar nivel de asociación para el título de contexto
        if event_id:
            scope = f"evento:{event_id}"
        else:
            scope = f"organización:{org_id_doc}"

        documents.append(Document(
            page_content=f"DOCUMENTO: {raw_doc.get('name', '')}\nASSOCIACIÓN: {scope}\n\n{content}",
            metadata={
                "doc_type": raw_doc.get("mimetype", "document"),
                "title": raw_doc.get("name", ""),
                "event_id": event_id,
                "org_id": org_id_doc,
                "source": raw_doc.get("url", ""),
                "collection": "documents",
            }
        ))

    print(f"[build_content_rag] {len(raw_documents)} documentos sueltos preparados para indexar.")
    
    for eid, cdata in courses_data.items():
        course_name = cdata["name"]
        print(f"[build_content_rag] Procesando y resumiendo curso: '{course_name}' ({eid})...")
        
        modules_lines = []
        for mid, mdata in cdata["modules"].items():
            modules_lines.append(f"Módulo: {mdata['name']}")
            for a in mdata["activities"]:
                modules_lines.append(f" - Actividad: {a['name']}")
                if a['transcripts']:
                    modules_lines.append(f"   (Habla sobre: {a['transcripts']})")
        
        if modules_lines:
            try:
                summary = await _summarize_course_content(course_name, modules_lines)
                content_to_index = f"CURSO: {course_name}\nDESCRIPCIÓN: {cdata['description']}\n\nRESUMEN GLOBAL:\n{summary}\n\nESTRUCTURA DE MÓDULOS Y ACTIVIDADES:\n" + "\n".join(modules_lines)
            except Exception as e:
                print(f"[build_content_rag] Error resumiendo curso '{course_name}': {e}")
                content_to_index = f"CURSO: {course_name}\nDESCRIPCIÓN: {cdata['description']}\n\n" + "\n".join(modules_lines)
        else:
            # Curso sin actividades — indexar solo con nombre y descripción
            print(f"[build_content_rag] Curso '{course_name}' no tiene actividades, indexando solo descripción.")
            content_to_index = f"CURSO: {course_name}\nDESCRIPCIÓN: {cdata['description']}"
            if not cdata['description']:
                print(f"[build_content_rag] Curso '{course_name}' vacío, ignorado.")
                continue  # nada útil que indexar

        doc = Document(
            page_content=content_to_index,
            metadata={
                "doc_type": "course_summary",
                "collection": "events",
                "event_id": eid,
                "name": course_name
            }
        )
        documents.append(doc)

    final_documents = []
    if documents:
        print(f"[build_content_rag] Preparando un total de {len(documents)} bloques finales de texto para el chunker...")
        raw_docs_list = [{"text": doc.page_content, **doc.metadata} for doc in documents]
        final_documents = _prepare_documents(raw_docs_list)
        print(f"[build_content_rag] Documentos fragmentados en {len(final_documents)} chunks. Obteniendo embeddings y guardando en Chroma/FAISS...")
        embeddings = get_embeddings()
        _build_vector_store(final_documents, embeddings, platform_id, org_id, force=True)
        print(f"[build_content_rag] ¡Indexación completada exitosamente! Se han guardado {len(final_documents)} chunks en la Base de Datos Vectorial.")
        
        from app.rag.pipeline import invalidate_retriever
        invalidate_retriever(platform_id, org_id)

    return {
        "status": "ready",
        "courses_indexed": len(courses_data),
        "documents_indexed": len(raw_documents),
        "total_chunks": len(final_documents),
    }

async def reindex_documents(
    uri: str,
    database: str,
    platform_id: str,
    org_id: str | None = None,
) -> dict[str, Any]:
    """
    Re-indexa solo la colección 'documents' en el RAG existente.
    No toca cursos ni llama al LLM — es rápido y apto para llamarse
    cada vez que GenCampus sube un documento nuevo.
    """
    from app.rag.pipeline import (
        _load_vector_store, _build_vector_store, get_embeddings,
        _prepare_documents, _chroma_dir, _namespace,
    )

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    db = client[database]
    raw_documents = list(db["documents"].find({"active": True}))
    client.close()

    print(f"[reindex_documents] {len(raw_documents)} documentos activos encontrados.")

    lc_docs: list[Document] = []
    for raw_doc in raw_documents:
        content = raw_doc.get("content", "").strip()
        if not content:
            continue
        event_id = str(raw_doc["eventId"]) if raw_doc.get("eventId") else ""
        org_id_doc = str(raw_doc.get("organizer_id", ""))
        lc_docs.append(Document(
            page_content=f"DOCUMENTO: {raw_doc.get('name', '')}\n\n{content}",
            metadata={
                "doc_type": "document",
                "collection": "documents",
                "title": raw_doc.get("name", ""),
                "event_id": event_id,
                "org_id": org_id_doc,
                "source": raw_doc.get("url", ""),
                "mongo_id": str(raw_doc["_id"]),
            },
        ))

    embeddings = get_embeddings()
    persist_dir = _chroma_dir(platform_id, org_id)

    # Eliminar chunks anteriores de 'documents' del índice existente
    if os.path.exists(persist_dir):
        try:
            import chromadb as _chromadb
            chroma_client = _chromadb.PersistentClient(path=persist_dir)
            col = chroma_client.get_collection(_namespace(platform_id, org_id))
            existing = col.get(where={"collection": "documents"})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
                print(f"[reindex_documents] {len(existing['ids'])} chunks anteriores eliminados.")
        except Exception as e:
            print(f"[reindex_documents] No se pudo limpiar chunks anteriores: {e}")

    if not lc_docs:
        return {"status": "ok", "documents_indexed": 0, "chunks_indexed": 0}

    raw_docs_list = [{"text": doc.page_content, **doc.metadata} for doc in lc_docs]
    chunks = _prepare_documents(raw_docs_list)

    vs = _load_vector_store(embeddings, platform_id, org_id)
    if vs is not None:
        vs.add_documents(chunks)
    else:
        _build_vector_store(chunks, embeddings, platform_id, org_id, force=False)

    print(f"[reindex_documents] {len(chunks)} chunks indexados.")

    from app.rag.pipeline import invalidate_retriever
    invalidate_retriever(platform_id, org_id)

    return {
        "status": "ok",
        "documents_indexed": len(lc_docs),
        "chunks_indexed": len(chunks),
    }
async def build_single_course_rag(
    uri: str, database: str, platform_id: str, event_id: str, org_id: str | None = None
) -> dict[str, Any]:
    """Indexa un solo curso de forma incremental sin tocar el resto del índice."""
    print(f"[build_single_course_rag] Iniciando indexación incremental para GenCampus. event_id={event_id}, org_id={org_id}")
    from bson import ObjectId
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    db = client[database]

    # Buscar el evento
    try:
        eid_query = ObjectId(event_id)
    except Exception:
        eid_query = event_id
        
    print(f"[build_single_course_rag] Buscando curso con _id={eid_query}...")
    event = db["events"].find_one({"_id": eid_query})
    if not event:
        print(f"[build_single_course_rag] Curso {event_id} no encontrado en MongoDB.")
        client.close()
        return {"status": "error", "message": f"Evento {event_id} no encontrado."}

    eid_str = str(event["_id"])
    course_name = event.get("name", event.get("title", "Curso sin nombre"))
    print(f"[build_single_course_rag] Curso encontrado: '{course_name}'. Buscando actividades...")

    # Actividades del curso
    activities = list(db["activities"].find(
        {"$or": [{"event_id": eid_str}, {"eventId": eid_str}]}
    ))
    
    print(f"[build_single_course_rag] Se encontraron {len(activities)} actividades para '{course_name}'. Buscando transcripciones...")
    transcripts = list(db["transcript_segments"].find({})) if activities else []

    modules_lines = []
    for act in activities:
        aid = str(act.get("_id"))
        act_name = act.get("name", act.get("title", "Actividad"))
        act_transcripts = [t.get("text", "") for t in transcripts if str(t.get("activity_id")) == aid]
        transcripts_text = " ".join(act_transcripts)[:2000]
        modules_lines.append(f" - Actividad: {act_name}")
        if transcripts_text:
            modules_lines.append(f"   (Habla sobre: {transcripts_text})")

    client.close()

    print(f"[build_single_course_rag] Procesando contenido de '{course_name}'...")
    if modules_lines:
        print(f"[build_single_course_rag] Generando resumen LLM de la estructura del curso...")
        try:
            summary = await _summarize_course_content(course_name, modules_lines)
            content = f"CURSO: {course_name}\nDESCRIPCIÓN: {event.get('description', '')}\n\nRESUMEN GLOBAL:\n{summary}\n\nESTRUCTURA:\n" + "\n".join(modules_lines)
        except Exception as e:
            print(f"[build_single_course_rag] Error resumiendo el curso: {e}")
            content = f"CURSO: {course_name}\nDESCRIPCIÓN: {event.get('description', '')}\n\n" + "\n".join(modules_lines)
    else:
        print(f"[build_single_course_rag] El curso no tiene actividades, indexando solo descripción...")
        content = f"CURSO: {course_name}\nDESCRIPCIÓN: {event.get('description', '')}"

    print(f"[build_single_course_rag] Fragmentando texto del curso en chunks...")
    from app.rag.pipeline import _prepare_documents
    raw = [{"text": content, "doc_type": "course_summary", "collection": "events", "event_id": eid_str, "name": course_name}]
    docs = _prepare_documents(raw)

    print(f"[build_single_course_rag] Obteniendo embeddings y guardando {len(docs)} chunks en la Base de Datos Vectorial...")
    embeddings = get_embeddings()
    
    # Prefix org_id with course_ so pipeline.py saves it in courses/event_id
    course_namespace = f"course_{eid_str}"
    
    # force=True → queremos que reemplace completamente la carpeta del curso si ya existía
    _build_vector_store(docs, embeddings, platform_id, course_namespace, force=True)
    
    from app.rag.pipeline import invalidate_retriever
    invalidate_retriever(platform_id, course_namespace)
    
    print(f"[build_single_course_rag] Indexación completada para '{course_name}'.")

    return {"status": "ready", "documents_indexed": len(docs), "course": course_name}


if __name__ == "__main__":
    from app.core.config import get_settings
    settings = get_settings()
    asyncio.run(build_content_rag(settings.gencampus_mongo_uri, settings.gencampus_mongo_db, "gencampus", "63f552d916065937427b3b02"))
