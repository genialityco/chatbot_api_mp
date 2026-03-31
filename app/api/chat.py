from fastapi import APIRouter, Depends, HTTPException
from app.core.auth import get_platform_context
from app.models.platform import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.services.history_service import clear_history, get_user_history, get_platform_history

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    ctx: dict = Depends(get_platform_context),
):
    """
    Endpoint principal de chat.
    
    Headers requeridos:
    - X-Platform-Id: ID de la plataforma
    - X-API-Key: API key de la plataforma
    - X-Org-Id: (opcional) ID de la organización
    """
    platform = ctx["platform"]
    org_id = ctx["org_id"]

    system_prompt = platform.get_system_prompt(org_id)

    service = ChatService(
        platform_id=ctx["platform_id"],
        org_id=org_id,
        system_prompt=system_prompt,
        db_connections=[c.model_dump() for c in platform.db_connections],
    )

    try:
        result = await service.chat(
            message=request.message,
            user_id=request.user_id,
            user_name=request.user_name,
            org_id=request.org_id,
            session_id=request.session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(**result)


@router.delete("/history/{user_id}")
async def delete_history(
    user_id: str,
    ctx: dict = Depends(get_platform_context),
):
    """Borra el historial de chat de un usuario en la plataforma."""
    await clear_history(ctx["platform_id"], user_id)
    return {"message": "Historial eliminado.", "user_id": user_id}


@router.get("/history/{user_id}")
async def user_history(
    user_id: str,
    limit: int = 50,
    ctx: dict = Depends(get_platform_context),
):
    """Historial completo de conversaciones de un usuario."""
    return await get_user_history(ctx["platform_id"], user_id, limit)


@router.get("/history")
async def platform_history(
    limit: int = 100,
    ctx: dict = Depends(get_platform_context),
):
    """Conversaciones recientes de toda la plataforma (útil para recomendaciones)."""
    return await get_platform_history(ctx["platform_id"], limit)


@router.get("/sessions/{user_id}")
async def user_sessions(
    user_id: str,
    ctx: dict = Depends(get_platform_context),
):
    """Devuelve las sesiones de conversación de un usuario, agrupadas por session_id."""
    from app.models.conversation import ChatTurn
    turns = await ChatTurn.find(
        ChatTurn.platform_id == ctx["platform_id"],
        ChatTurn.user_id == user_id,
    ).sort(-ChatTurn.created_at).limit(200).to_list()

    # Agrupar por session_id manteniendo orden cronológico
    sessions: dict[str, dict] = {}
    for t in reversed(turns):
        sid = t.session_id
        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "title": t.user_message[:60] + ("..." if len(t.user_message) > 60 else ""),
                "created_at": t.created_at.isoformat(),
                "turns": [],
            }
        sessions[sid]["turns"].append({
            "user": t.user_message,
            "assistant": t.assistant_message,
            "created_at": t.created_at.isoformat(),
        })

    # Devolver más recientes primero
    return list(reversed(list(sessions.values())))
