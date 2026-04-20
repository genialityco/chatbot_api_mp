"""
Extractor de contenido para GenCampus.
Extrae cursos, módulos, actividades y transcripciones, genera resúmenes
por medio del LLM y los prepara como documentos para indexar en el RAG.
"""
import asyncio
from typing import Any
from pymongo import MongoClient
from langchain_core.documents import Document

from app.core.config import get_settings
from app.rag.pipeline import _build_vector_store, get_embeddings

settings = get_settings()

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
    """Extrae los datos reales, los resume e indexa en el RAG."""
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    db = client[database]
    
    events = list(db["events"].find({}))
    modules = list(db["modules"].find({}))
    activities = list(db["activities"].find({}))
    transcripts = list(db["transcript_segments"].find({}))

    print(f"[content_indexer] Encontrados: {len(events)} eventos, {len(modules)} modulos, {len(activities)} actividades, {len(transcripts)} fragmentos transcript.")

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
        eid = str(mod.get("eventId") or mod.get("event_id"))
        mid = str(mod.get("_id"))
        if eid in courses_data:
            courses_data[eid]["modules"][mid] = {
                "name": mod.get("name", mod.get("title", "Módulo sin nombre")),
                "activities": []
            }
            
    # Agregar actividades y fragmentos
    for act in activities:
        eid = str(act.get("eventId") or act.get("event_id"))
        mid = str(act.get("moduleId") or act.get("module_id"))
        
        # En caso de que la actividad no tenga moduleId pero sí eventId, crear un módulo genérico
        if not mid and eid in courses_data:
            mid = "unassigned"
            if mid not in courses_data[eid]["modules"]:
                courses_data[eid]["modules"][mid] = {"name": "Módulo General", "activities": []}
                
        if eid in courses_data and mid in courses_data[eid]["modules"]:
            aid = str(act.get("_id"))
            act_name = act.get("name", act.get("title", "Actividad"))
            
            # Buscar transcripciones de esta actividad
            act_transcripts = [t.get("text", "") for t in transcripts if str(t.get("activity_id")) == aid]
            transcripts_text = " ".join(act_transcripts)
            if len(transcripts_text) > 2000:
                transcripts_text = transcripts_text[:1997] + "..." # Limitar para no explotar tokens en resumen
                
            courses_data[eid]["modules"][mid]["activities"].append({
                "name": act_name,
                "content": act.get("description", act.get("content", "")),
                "transcripts": transcripts_text
            })

    client.close()

    documents = []
    
    # Procesar cada curso para RAG
    from app.rag.pipeline import _prepare_documents
    
    for eid, cdata in courses_data.items():
        course_name = cdata["name"]
        print(f"[content_indexer] Generando resumen para curso: {course_name}")
        
        modules_lines = []
        for mid, mdata in cdata["modules"].items():
            modules_lines.append(f"Módulo: {mdata['name']}")
            for a in mdata["activities"]:
                modules_lines.append(f" - Actividad: {a['name']}")
                if a['transcripts']:
                    modules_lines.append(f"   (Habla sobre: {a['transcripts']})")
        
        if modules_lines:
            try:
                # 1. Resumen con IA
                summary = await _summarize_course_content(course_name, modules_lines)
                
                # 2. Agregar al documento principal del curso
                content_to_index = f"CURSO: {course_name}\nDESCRIPCIÓN: {cdata['description']}\n\nRESUMEN GLOBAL:\n{summary}\n\nESTRUCTURA DE MÓDULOS Y ACTIVIDADES:\n" + "\n".join(modules_lines)
                
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
            except Exception as e:
                print(f"[content_indexer] Error resumiendo {course_name}: {e}")

    # Dividir y empaquetar los documentos grandes usando _prepare_documents para evitar el error de límite de embeddings
    if documents:
        # Convertiremos los Documents construidos arriba al formato que espera `_prepare_documents` 
        # (lista de diccionarios con 'text' y metadata)
        raw_docs_list = []
        for doc in documents:
            raw_docs_list.append({
                "text": doc.page_content,
                **doc.metadata
            })
            
        final_documents = _prepare_documents(raw_docs_list)
        
        embeddings = get_embeddings()
        _build_vector_store(final_documents, embeddings, platform_id, org_id, force=True)
        print(f"[content_indexer] Se han indexado {len(final_documents)} chunks de contenido en RAG.")
    
    return {
        "status": "ready",
        "documents_indexed": len(documents)
    }

if __name__ == "__main__":
    from app.core.config import get_settings
    settings = get_settings()
    asyncio.run(build_content_rag(settings.gencampus_mongo_uri, settings.gencampus_mongo_db, "gencampus", "63f552d916065937427b3b02"))
