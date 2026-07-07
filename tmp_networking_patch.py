async def build_networking_rag(
    platform_id: str,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Indexa eventos y datos de Firestore para networking."""
    try:
        db = _init_firestore_client()
    except Exception as e:
        return {"status": "error", "message": f"Error inicializando Firestore: {str(e)}"}

    try:
        if event_id:
            event_doc = db.collection("events").document(event_id).get()
            if not event_doc.exists:
                return {"status": "error", "message": f"Evento {event_id} no encontrado en Firestore."}
            event_data = event_doc.to_dict() or {}

            users_snap = db.collection("users").where("eventId", "==", event_id).stream()
            users_for_event = [doc.to_dict() or {} for doc in users_snap]

            documents = _networking_event_to_documents(event_id, event_data, users_for_event)
            raw_docs = [{"text": doc.page_content, **doc.metadata} for doc in documents]
            chunks = _prepare_documents(raw_docs)
            embeddings = get_embeddings()
            _build_vector_store(chunks, embeddings, platform_id, event_id, force=True)

            return {
                "status": "ready",
                "event_id": event_id,
                "documents_indexed": len(documents),
                "chunks_indexed": len(chunks),
            }

        events_snap = db.collection("events").stream()
        users_by_event: dict[str, list[dict[str, Any]]] = {}

        all_users_snap = db.collection("users").stream()
        for user_doc in all_users_snap:
            user = user_doc.to_dict() or {}
            eid = str(user.get("eventId") or user.get("event_id") or "")
            if eid:
                users_by_event.setdefault(eid, []).append(user)

        documents: list[Document] = []
        event_count = 0
        for event_doc in events_snap:
            event_data = event_doc.to_dict() or {}
            eid = event_doc.id
            docs = _networking_event_to_documents(eid, event_data, users_by_event.get(eid, []))
            documents.extend(docs)
            event_count += 1

        if not documents:
            return {"status": "error", "message": "No se encontraron eventos para indexar en Firestore."}

        raw_docs = [{"text": doc.page_content, **doc.metadata} for doc in documents]
        chunks = _prepare_documents(raw_docs)
        embeddings = get_embeddings()
        _build_vector_store(chunks, embeddings, platform_id, None, force=True)

        return {
            "status": "ready",
            "documents_indexed": len(documents),
            "chunks_indexed": len(chunks),
            "events_indexed": event_count,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Error indexando Firestore: {str(e)}"}
