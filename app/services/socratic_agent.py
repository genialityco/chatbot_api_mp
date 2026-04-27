"""
Agente Socrático — enseña mediante preguntas guiadas en lugar de dar respuestas directas.

Flujo ReAct por turno:
  1. Evalúa la respuesta anterior del usuario contra el RAG (¿entendió el concepto?).
  2. Decide la siguiente pregunta socrática (más profunda, aclaratoria o de refuerzo).
  3. Genera y devuelve la pregunta — nunca la respuesta directa.

Compatible con WhatsApp y el chat web sin cambios en las capas de transporte.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from app.core.config import get_settings
from app.rag.pipeline import RAGRetriever
from app.services.history_service import load_history, save_history, persist_turn
from app.db.mongo_query import fetch_collection_data, docs_to_context, generate_filter_async
from app.db.schema_introspector import load_schema_cache

settings = get_settings()

# Palabras clave que indican consulta personal (requiere DB con filtro por usuario)
_PERSONAL_KW = [
    "mis", "mi", "mío", "mía", "tengo", "he tomado", "he completado",
    "mi progreso", "mis certificados", "mis inscripciones", "estoy inscrito",
    "my", "i have", "i am enrolled", "mis cursos", "mis clases",
]
_PATTERN_PERSONAL = r'\b(?:' + '|'.join(_PERSONAL_KW) + r')\b'

# Palabras clave que indican búsqueda de videos/transcripts
_VIDEO_KW = ["video", "videos", "grabación", "grabacion", "transcripción", "transcripcion",
             "conferencia", "conferencias", "clase grabada", "clases grabadas"]
_PATTERN_VIDEO = r'\b(?:' + '|'.join(_VIDEO_KW) + r')\b'

# Palabras clave que indican solicitud de detalle de curso/actividad (link, actividades, módulos)
_DETAIL_KW = ["link", "enlace", "url", "actividad", "actividades", "módulo", "modulo",
              "módulos", "modulos", "primera actividad", "ver el curso", "acceder",
              "más información", "mas información", "más info", "mas info", "detalles"]
_PATTERN_DETAIL = r'\b(?:' + '|'.join(_DETAIL_KW) + r')\b'

# Palabras clave que indican solicitud de evaluación/quiz
_EVAL_KW = ["evalúame", "evaluame", "evalúa", "evalua", "examen", "quiz", "prueba",
            "quiero que me evalúes", "quiero que me evalues", "hazme una prueba",
            "hazme un examen", "quiero ser evaluado", "ponme a prueba", "test"]
_PATTERN_EVAL = r'\b(?:' + '|'.join(_EVAL_KW) + r')\b'

# ─── Clasificadores de intención ─────────────────────────────────────────────

async def _is_specific_catalog_query(query: str, last_agent_msg: str = "") -> bool:
    """
    Determina si la consulta (o la respuesta a una pregunta de refinamiento)
    menciona un tema concreto que permita buscar en la DB.
    """
    from google import genai
    from google.genai import types
    import warnings

    context_hint = ""
    if last_agent_msg:
        context_hint = f"El asistente preguntó antes: \"{last_agent_msg[:120]}\"\n"

    prompt = (
        f"{context_hint}"
        "Analiza el siguiente mensaje y determina si menciona un tema, enfermedad, área o curso ESPECÍFICO "
        "que permita hacer una búsqueda concreta en una base de datos educativa.\n"
        "Responde SOLO 'SI' o 'NO'.\n\n"
        "Ejemplos ESPECÍFICOS (SI): diabetes, obesidad, cardiología, nutrición, oncología, endocrinología\n"
        "Ejemplos GENÉRICOS (NO): algo nuevo, cualquier cosa, lo que sea, no sé\n\n"
        f"Mensaje: \"{query}\"\n\nRespuesta:"
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
        ans = response.text.strip().lower()
        return "si" in ans or "sí" in ans
    except Exception as e:
        print(f"[socratic] _is_specific_catalog_query error: {e}")
        return True


# ─── Prompts ─────────────────────────────────────────────────────────────────

_SOCRATIC_SYSTEM = """{base_prompt}

## Modo Socrático — Instrucciones de comportamiento
Eres un tutor que enseña EXCLUSIVAMENTE mediante el método socrático.
NUNCA des la respuesta directa. Tu único objetivo es guiar al estudiante
para que DESCUBRA el conocimiento por sí mismo.

### Ciclo de razonamiento (ReAct interno)
Para cada mensaje del estudiante debes:
1. EVALUAR: ¿La respuesta del estudiante demuestra comprensión del concepto?
   - Compara con el CONOCIMIENTO DE REFERENCIA (RAG) provisto abajo.
   - Identifica conceptos correctos, incorrectos o ausentes.
2. ESTRATEGIZAR: Elige una acción:
   - Si el estudiante está en lo correcto → profundiza con una pregunta de nivel superior.
   - Si está parcialmente correcto → haz una pregunta que lo lleve a completar la idea.
   - Si está equivocado → haz una pregunta más simple que lo lleve a reconsiderar.
   - Si es la primera interacción → comienza con una pregunta abierta sobre el tema.
3. ACTUAR: Formula UNA sola pregunta clara y concisa.

### Reglas estrictas
- Responde SIEMPRE con una pregunta, nunca con una afirmación directa.
- Usa el CONOCIMIENTO DE REFERENCIA para validar respuestas, pero no lo cites textualmente.
- Si el estudiante pide la respuesta directamente, responde con una pista en forma de pregunta.
- Adapta el idioma al del estudiante.
- Máximo 3 oraciones por respuesta (pista + pregunta).
- Cuando el estudiante demuestre comprensión completa, felicítalo brevemente y propón el siguiente concepto con una nueva pregunta.
- EXCEPCIÓN: Si el usuario pregunta por sus datos personales (cursos, progreso, inscripciones), responde con esa información directamente usando los DATOS DE LA BASE DE DATOS, luego formula una pregunta socrática sobre el contenido de esos cursos.
- NUNCA muestres IDs de MongoDB (cadenas hexadecimales de 24 caracteres) en la respuesta. Usa solo nombres, títulos o descripciones.
- NUNCA muestres URLs con "undefined" o IDs en ellas. Si no tienes una URL válida, omítela.
- NUNCA inventes cursos, eventos o datos que no estén explícitamente en los DATOS DE LA BASE DE DATOS. Si no hay datos, dilo y pregunta socráticamente.

## Contexto del estudiante
- ID: {user_id}
- Nombre: {user_name}
- Sesión: {session_id}

## DATOS DE LA BASE DE DATOS
{db_context}

## CONOCIMIENTO DE REFERENCIA (RAG)
{rag_context}

Nota sobre el RAG: cada fuente está etiquetada con su tipo. "CURSO: nombre" indica un curso completo
con sus actividades internas. Cuando menciones contenido, distingue claramente si es el curso en sí
o una actividad/tema específico dentro de ese curso, respetando la jerarquía: Curso → Módulo (si existe) → Actividad.
"""

_FIRST_TURN_PROMPT = """{base_prompt}

## Modo Socrático — Primera interacción
El estudiante acaba de enviar: "{topic}"

## DATOS DE LA BASE DE DATOS
{db_context}

## CONOCIMIENTO DE REFERENCIA (RAG)
{rag_context}

Instrucciones según el tipo de mensaje:
- Si el RAG dice "No aplica" y los datos dicen "No aplica", la pregunta es GENÉRICA.
  En ese caso responde con UNA sola pregunta socrática para descubrir qué tema o área específica le interesa al estudiante. NO menciones cursos ni datos concretos.
- Si hay DATOS DE LA BASE DE DATOS reales, muéstralos directamente (sin IDs) y luego formula una pregunta socrática sobre ese contenido.
- Si hay CONOCIMIENTO DE REFERENCIA real, formula una pregunta socrática abierta sobre ese tema.
- NUNCA muestres IDs de MongoDB ni URLs con "undefined".
La pregunta debe ser abierta, no respondible con sí/no.
"""


# ─── Agente ──────────────────────────────────────────────────────────────────

def _resolve_org_id(conn: dict, user_id: str) -> str | None:
    """Busca el org_id del usuario en organizationusers."""
    try:
        from bson import ObjectId
        from app.db.mongo_pool import get_db
        db = get_db(conn["uri"], conn["database"])
        try:
            query_id = ObjectId(user_id)
        except Exception:
            query_id = user_id
        doc = db["organizationusers"].find_one(
            {"user_id": query_id},
            {"organization_id": 1}
        )
        if doc and doc.get("organization_id"):
            return str(doc["organization_id"])
        return None
    except Exception as e:
        print(f"[socratic] _resolve_org_id error: {e}")
        return None

class SocraticAgent:
    """
    Agente socrático que envuelve el RAG y el LLM para enseñar mediante preguntas.
    Comparte la misma interfaz que ChatService.chat() para ser intercambiable.
    """

    def __init__(
        self,
        platform_id: str,
        org_id: str | None,
        base_prompt: str,
        db_connections: list[dict] | None = None,
    ):
        self.platform_id = platform_id
        self.org_id = org_id
        self.base_prompt = base_prompt
        self.db_connections = db_connections or []
        self.retriever = RAGRetriever(platform_id, org_id)

    async def chat(
        self,
        message: str,
        user_id: str,
        user_name: str | None = None,
        org_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        import uuid
        session_id = session_id or str(uuid.uuid4())
        resolved_org_id = org_id or self.org_id

        # Si aún no tenemos org_id, resolverlo desde organizationusers del usuario
        if not resolved_org_id and self.db_connections:
            resolved_org_id = await asyncio.to_thread(
                _resolve_org_id, self.db_connections[0], user_id
            )
            print(f"[socratic] resolved org_id from db: {resolved_org_id}")

        # ── 1. RAG + historial + clasificación de intención en paralelo ───────
        from app.services.chat_service import _requires_db_query, _search_transcripts_async, _build_gencampus_cards_template_async, _transcripts_to_context, _invoke_llm, _save_quiz_result

        is_personal = bool(re.search(_PATTERN_PERSONAL, message.lower()))
        is_video = bool(re.search(_PATTERN_VIDEO, message.lower()))
        is_eval_request = bool(re.search(_PATTERN_EVAL, message.lower()))
        is_detail_request = bool(re.search(_PATTERN_DETAIL, message.lower()))

        (rag_context, sources), history, needs_db = await asyncio.gather(
            asyncio.to_thread(self._rag_retrieve, message),
            load_history(self.platform_id, user_id, session_id),
            _requires_db_query(message),
        )

        # Si hay historial, revisar si el agente estaba refinando el tema del usuario.
        # En ese caso la respuesta del usuario ES el tema → forzar consulta específica a DB.
        forced_specific = False
        if not is_personal and not is_video and history:
            last_assistant = next(
                (m["content"] for m in reversed(history) if m.get("role") == "assistant"),
                ""
            )
            # Señales de que el agente hizo una pregunta de refinamiento de tema
            _refine_signals = [
                "qué tema", "qué área", "qué aspecto", "qué tipo de curso",
                "qué te interesa", "cuál es el tema", "sobre qué tema",
                "qué campo", "qué especialidad", "qué temas", "qué áreas",
                "interesa aprender", "te gustaría aprender", "te interesa",
                "específicamente", "en particular", "cuál sería", "cuál tema",
            ]
            # No activar si el último mensaje era un saludo/bienvenida genérica
            _greeting_signals = [
                "hola", "bienvenido", "en qué puedo ayudarte", "cómo puedo ayudarte",
                "en qué te puedo", "¿en qué", "¿cómo puedo",
            ]
            is_greeting = any(s in last_assistant.lower() for s in _greeting_signals)
            if not is_greeting and any(s in last_assistant.lower() for s in _refine_signals):
                needs_db = True
                forced_specific = True
                print(f"[socratic] forced_specific=True, last_agent_msg={last_assistant[:80]}")

        print(f"[socratic] rag_context_len={len(rag_context)} sources={len(sources)} needs_db={needs_db} is_personal={is_personal} is_video={is_video} is_eval={is_eval_request} is_detail={is_detail_request}")
        rag_context = rag_context or "No se encontró conocimiento de referencia específico."

        # ── 2a. Flujo de evaluación: buscar cursos y preguntar cuál evaluar ──
        if is_eval_request and self.db_connections:
            # Verificar si el agente ya preguntó qué curso evaluar (evitar loop)
            last_assistant = next(
                (m["content"] for m in reversed(history) if m.get("role") == "assistant"), ""
            )
            already_asked_course = any(s in last_assistant.lower() for s in [
                "qué curso", "cual curso", "cuál curso", "sobre qué curso", "qué tema quieres"
            ])

            if not already_asked_course:
                courses_ctx = await self._fetch_user_data(message, user_id, resolved_org_id)
                system_content = f"""{self.base_prompt}

El usuario quiere ser evaluado. Tienes la lista de sus cursos inscritos abajo.
Pregúntale EXACTAMENTE cuál curso desea que lo evalúes. Lista los cursos disponibles de forma clara y numerada.
NO empieces la evaluación todavía. Solo pregunta cuál curso.
NUNCA muestres IDs de MongoDB.

## CURSOS DEL USUARIO
{courses_ctx}
"""
                messages = [{"role": "system", "content": system_content}]
                for msg in history:
                    role = "user" if msg.get("role") == "user" else "model"
                    content = msg.get("content", "")
                    if not isinstance(content, str):
                        content = str(content)
                    messages.append({"role": role, "content": content})
                messages.append({"role": "user", "content": message})

                answer_text = await _invoke_llm(messages, temperature=0.2)
                print(f"[socratic] eval course selection answer={answer_text[:80]}")

                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": answer_text})
                await asyncio.gather(
                    save_history(self.platform_id, user_id, session_id, history),
                    persist_turn(
                        platform_id=self.platform_id, user_id=user_id, session_id=session_id,
                        user_message=message, assistant_message=answer_text,
                        user_name=user_name, org_id=org_id, collection_used=None, sources=[],
                    ),
                )
                return {
                    "answer": answer_text,
                    "session_id": session_id,
                    "sources": [],
                    "platform_id": self.platform_id,
                    "org_id": self.org_id,
                }

        # ── 2. Consultar DB según intención ───────────────────────────────────
        db_context = ""
        templates_context = ""
        conn = self.db_connections[0] if self.db_connections else None

        if is_detail_request and conn and history:
            # El usuario pide detalles/link de un curso mencionado en el historial
            # Extraer el nombre del curso del historial reciente
            last_assistant = next(
                (m["content"] for m in reversed(history) if m.get("role") == "assistant"), ""
            )
            detail_ctx = await self._fetch_course_detail(message, last_assistant, resolved_org_id)
            if detail_ctx:
                db_context = detail_ctx
                print(f"[socratic] detail_context_len={len(db_context)}")

        elif is_video and conn:
            # Búsqueda de transcripts — igual que ChatService
            transcript_docs = await _search_transcripts_async(conn, message)
            if transcript_docs:
                db_context = _transcripts_to_context(transcript_docs)
                if resolved_org_id:
                    cards_html = await _build_gencampus_cards_template_async(conn, transcript_docs, "transcript_segments", resolved_org_id)
                    if cards_html:
                        templates_context = (
                            "\n\n## PLANTILLAS DE TARJETAS HTML DISPONIBLES\n"
                            "Para cada video encontrado, IMPRIME EXACTAMENTE su bloque HTML y reemplaza "
                            "{{RESUMEN_AQUI}} con tu descripción de qué se habla en el video.\n\n"
                            + cards_html + "\n\n"
                        )
                print(f"[socratic] transcript_docs={len(transcript_docs)}")
        elif is_personal and conn and needs_db:
            db_context = await self._fetch_user_data(message, user_id, resolved_org_id)
            print(f"[socratic] db_context_len={len(db_context)}")
        elif needs_db and conn and not is_personal:
            last_assistant = next(
                (m["content"] for m in reversed(history) if m.get("role") == "assistant"),
                ""
            ) if history else ""
            is_specific = forced_specific or await _is_specific_catalog_query(message, last_assistant)
            print(f"[socratic] is_specific={is_specific} forced={forced_specific}")
            if is_specific:
                db_context = await self._fetch_catalog_data(message, resolved_org_id)
                print(f"[socratic] catalog_context_len={len(db_context)}")

        # ── 3. Construir prompt ───────────────────────────────────────────────
        is_first_turn = len(history) == 0

        # Si la consulta es genérica (no específica, no personal, no video),
        # no inyectar RAG ni DB — el agente debe preguntar socráticamente para refinar.
        is_generic = needs_db and not is_personal and not is_video and not db_context
        effective_rag = rag_context if not is_generic else "No aplica — pregunta al usuario qué tema específico le interesa."
        full_db_context = (db_context + templates_context) if (db_context or templates_context) else "No aplica para esta consulta."

        if is_first_turn:
            system_content = _FIRST_TURN_PROMPT.format(
                base_prompt=self.base_prompt,
                topic=message,
                db_context=full_db_context,
                rag_context=effective_rag,
            )
        else:
            system_content = _SOCRATIC_SYSTEM.format(
                base_prompt=self.base_prompt,
                user_id=user_id,
                user_name=user_name or "estudiante",
                session_id=session_id,
                db_context=full_db_context,
                rag_context=effective_rag,
            )

        # ── 4. Construir mensajes con historial ───────────────────────────────
        from app.services.history_service import compress_history_for_prompt
        history_summary, recent_history = await compress_history_for_prompt(history)

        if history_summary:
            # Inyectar resumen en el system prompt
            system_content += f"\n\n## CONTEXTO DE CONVERSACIÓN ANTERIOR\n{history_summary}"

        messages: list[dict] = [{"role": "system", "content": system_content}]
        for msg in recent_history:
            role = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        print(f"[socratic] prompt_messages={len(messages)} history_turns={len(recent_history)//2} summary={'yes' if history_summary else 'no'}")

        # ── 5. Invocar LLM ────────────────────────────────────────────────────
        # Para videos/catálogo usar temperatura baja; para socrático más alta
        temperature = 0.2 if (is_video or (needs_db and not is_personal)) else 0.5
        answer_text = await _invoke_llm(messages, temperature=temperature)

        print(f"[socratic] answer={answer_text[:120] if answer_text else 'EMPTY'}")

        # ── 6. Detectar bloque JSON de quiz y guardarlo ───────────────────────
        quiz_save_coro = None
        quiz_match = re.search(r"```json\s*(\{.*?\"quiz_result\".*?\})\s*```", answer_text, re.DOTALL)
        if quiz_match:
            try:
                quiz_data = json.loads(quiz_match.group(1))
                uri = self.db_connections[0]["uri"] if self.db_connections else settings.meta_mongodb_uri
                database = self.db_connections[0]["database"] if self.db_connections else settings.meta_mongodb_db
                quiz_save_coro = _save_quiz_result(uri, database, self.platform_id, user_id, org_id, quiz_data)
                answer_text = (answer_text[:quiz_match.start()] + answer_text[quiz_match.end():]).strip()
                print(f"[socratic] quiz_result detected and queued for save")
            except Exception as e:
                print(f"[socratic] error parsing quiz json: {e}")

        # ── 7. Persistir historial ────────────────────────────────────────────
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer_text})

        tasks = [
            save_history(self.platform_id, user_id, session_id, history),
            persist_turn(
                platform_id=self.platform_id,
                user_id=user_id,
                session_id=session_id,
                user_message=message,
                assistant_message=answer_text,
                user_name=user_name,
                org_id=org_id,
                collection_used=sources[0].get("collection") if sources else None,
                sources=sources,
            ),
        ]
        if quiz_save_coro:
            tasks.append(quiz_save_coro)

        await asyncio.gather(*tasks)

        return {
            "answer": answer_text,
            "session_id": session_id,
            "sources": [sources[0]] if sources else [],
            "platform_id": self.platform_id,
            "org_id": self.org_id,
        }

    async def _fetch_catalog_data(self, message: str, org_id: str | None) -> str:
        """Consulta el catálogo general de eventos/cursos disponibles."""
        try:
            conn = self.db_connections[0]
            from app.services.chat_service import _run_query, _validate_filter_fields, _expand_events_text_filter_for_gencampus, _expand_text_filter_to_keywords, _extract_topic

            cached = load_schema_cache(conn["uri"], conn["database"], conn.get("collections"))
            col_info = (cached or {}).get("collections", {}).get("events", {})
            schema_fields = list(col_info.get("schema", {}).keys())
            enum_values = col_info.get("enum_values", {})

            # Extraer solo el tema clave para evitar frases largas en el regex
            topic = await _extract_topic(message)
            print(f"[socratic] catalog topic='{topic}'")

            # Si el topic es una palabra genérica, no tiene sentido buscar en DB
            _generic_topics = {"cursos", "curso", "eventos", "evento", "clases", "clase",
                               "actividades", "actividad", "algo", "temas", "tema", "información"}
            if topic.lower().strip() in _generic_topics:
                print(f"[socratic] topic '{topic}' is generic, skipping DB query")
                return ""

            # Construir filtro fuzzy directamente con el tema extraído
            keywords = [kw.strip() for kw in topic.replace(",", " ").split() if len(kw.strip()) > 3]
            if not keywords:
                keywords = [topic]

            # Un $or por cada keyword en name y description
            or_clauses = []
            for kw in keywords:
                or_clauses.append({"name": {"$regex": kw, "$options": "i"}})
                or_clauses.append({"description": {"$regex": kw, "$options": "i"}})
            mongo_filter = {"$or": or_clauses}

            print(f"[socratic] catalog filter={mongo_filter}")
            docs = await asyncio.to_thread(_run_query, conn, "events", mongo_filter, None, None)
            if docs:
                docs = docs[:5]
            if not docs:
                return ""
            return docs_to_context("events", docs, org_id or "")
        except Exception as e:
            print(f"[socratic] _fetch_catalog_data error: {e}")
            return ""

    async def _fetch_course_detail(self, message: str, last_assistant_msg: str, org_id: str | None) -> str:
        """Busca actividades de un curso mencionado en el historial reciente."""
        try:
            conn = self.db_connections[0]
            from app.services.chat_service import _run_query, _extract_topic

            # Extraer nombre del curso del mensaje del agente anterior o del mensaje actual
            combined = f"{last_assistant_msg} {message}"
            course_name = await _extract_topic(combined, "el nombre exacto del curso o simposio mencionado")
            print(f"[socratic] detail course_name='{course_name}'")

            if not course_name or len(course_name) < 4:
                return ""

            # 1. Buscar el evento por nombre
            event_filter = {"name": {"$regex": course_name, "$options": "i"}}
            events = await asyncio.to_thread(_run_query, conn, "events", event_filter, None, 1)
            if not events:
                # Intentar con palabras clave individuales
                keywords = [w for w in course_name.split() if len(w) > 4]
                if keywords:
                    event_filter = {"$or": [{"name": {"$regex": kw, "$options": "i"}} for kw in keywords]}
                    events = await asyncio.to_thread(_run_query, conn, "events", event_filter, None, 1)

            if not events:
                return ""

            event = events[0]
            event_id = str(event.get("_id", ""))
            event_name = event.get("name") or event.get("title") or course_name
            base_url = settings.gencampus_base_url

            lines = [f"### Curso: {event_name}"]
            if org_id:
                lines.append(f"URL del curso: {base_url}/organization/{org_id}/course/{event_id}")

            # 2. Buscar actividades del evento (event_id es string en activities)
            act_filter = {"$or": [{"event_id": event_id}, {"eventId": event_id}]}
            activities = await asyncio.to_thread(_run_query, conn, "activities", act_filter, None, None)

            if activities:
                lines.append(f"\nActividades ({len(activities)}):")
                for i, act in enumerate(activities[:10], 1):
                    act_id = str(act.get("_id", ""))
                    act_name = act.get("name") or act.get("title") or f"Actividad {i}"
                    act_url = f"{base_url}/organization/{org_id}/activitydetail/{act_id}" if org_id else ""
                    entry = f"  [{i}] {act_name}"
                    if act_url:
                        entry += f" → {act_url}"
                    lines.append(entry)
            else:
                # Sin actividades, buscar módulos
                mod_filter = {"$or": [{"event_id": event_id}, {"eventId": event_id}]}
                modules = await asyncio.to_thread(_run_query, conn, "modules", mod_filter, None, None)
                if modules:
                    lines.append(f"\nMódulos ({len(modules)}):")
                    for i, mod in enumerate(modules[:5], 1):
                        lines.append(f"  [{i}] {mod.get('name') or mod.get('title') or f'Módulo {i}'}")

            return "\n".join(lines)
        except Exception as e:
            print(f"[socratic] _fetch_course_detail error: {e}")
            return ""

    def _rag_retrieve(self, message: str) -> tuple[str, list[dict]]:
        try:
            # Log scores para diagnóstico
            store = self.retriever._get_store()
            raw = store.similarity_search_with_relevance_scores(message, k=settings.rag_top_k)
            print(f"[socratic] rag raw scores: {[(round(s, 3), d.metadata.get('name', d.page_content[:40])) for d, s in raw]}")

            context, sources = self.retriever.retrieve_as_context(message)
            print(f"[socratic] rag retrieved {len(sources)} sources (threshold={settings.rag_score_threshold})")
            return context, sources
        except ValueError as e:
            print(f"[socratic] rag not available: {e}")
            return "", []
        except Exception as e:
            print(f"[socratic] rag error: {e}")
            return "", []

    async def _fetch_user_data(self, message: str, user_id: str, org_id: str | None) -> str:
        """Consulta MongoDB para obtener datos personales del usuario (cursos, progreso, etc.)."""
        try:
            conn = self.db_connections[0]

            # Siempre resolver el org_id real del usuario desde la DB
            real_org_id = await asyncio.to_thread(_resolve_org_id, conn, user_id)
            resolved_org_id = real_org_id or org_id or self.org_id
            print(f"[socratic] _fetch_user_data org_id={resolved_org_id}")

            cached = load_schema_cache(conn["uri"], conn["database"], conn.get("collections"))
            if not cached and conn.get("collections"):
                cached = load_schema_cache(conn["uri"], conn["database"], None)

            primary = "courseattendees"
            col_info = (cached or {}).get("collections", {}).get(primary, {})
            schema_fields = list(col_info.get("schema", {}).keys())
            enum_values = col_info.get("enum_values", {})

            user_id_field = "userId" if "userId" in schema_fields else "user_id"
            enriched = f"{message} [{user_id_field}: {user_id}]"
            mongo_filter = await generate_filter_async(enriched, primary, schema_fields, enum_values)

            # Garantizar filtro por usuario
            if not mongo_filter:
                mongo_filter = {user_id_field: user_id}
            elif user_id_field not in str(mongo_filter):
                mongo_filter = {"$and": [{user_id_field: user_id}, mongo_filter]}

            print(f"[socratic] db query primary={primary} filter={mongo_filter}")

            from app.db.mongo_pool import get_db
            db = get_db(conn["uri"], conn["database"], readPreference="secondaryPreferred")
            attendee_docs = list(db[primary].find(mongo_filter).sort("updatedAt", -1).limit(10))

            # Resolver nombres de cursos desde la colección events
            if attendee_docs:
                from bson import ObjectId
                event_id_field = "eventId" if attendee_docs[0].get("eventId") else "event_id"
                event_ids = []
                for d in attendee_docs:
                    eid = d.get(event_id_field) or d.get("eventId") or d.get("event_id")
                    if eid:
                        try:
                            event_ids.append(ObjectId(str(eid)))
                        except Exception:
                            pass

                event_names: dict[str, str] = {}
                if event_ids:
                    events = list(db["events"].find({"_id": {"$in": event_ids}}, {"_id": 1, "name": 1, "title": 1}))
                    for ev in events:
                        name = ev.get("name") or ev.get("title") or "Sin nombre"
                        event_names[str(ev["_id"])] = name

                # Enriquecer los docs con el nombre del curso
                for d in attendee_docs:
                    eid = d.get(event_id_field) or d.get("eventId") or d.get("event_id")
                    if eid and str(eid) in event_names:
                        d["course_name"] = event_names[str(eid)]

            if not attendee_docs:
                return "No se encontraron inscripciones o cursos para este usuario."

            # Formatear manualmente para evitar IDs en el contexto
            base_url = settings.gencampus_base_url
            lines = [f"### Cursos inscritos del usuario ({len(attendee_docs)} registros)\n"]
            for i, doc in enumerate(attendee_docs, 1):
                name = doc.get("course_name") or doc.get("name") or "Curso sin nombre"
                progress = doc.get("progress", 0)
                status = doc.get("status", "")
                cert_hours = doc.get("certificationHours", "")
                event_id = doc.get("event_id") or doc.get("eventId", "")
                entry = f"[{i}] Curso: {name} | Progreso: {progress}%"
                if status:
                    entry += f" | Estado: {status}"
                if cert_hours:
                    entry += f" | Horas certificación: {cert_hours}"
                if resolved_org_id and event_id:
                    entry += f" | URL: {base_url}/organization/{resolved_org_id}/course/{event_id}"
                lines.append(entry)

            return "\n".join(lines)
        except Exception as e:
            print(f"[socratic] _fetch_user_data error: {e}")
            return "No se pudo acceder a los datos del usuario."
