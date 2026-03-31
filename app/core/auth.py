"""
Middleware de autenticación por API Key.
La plataforma y organización se pasan por headers HTTP:
  X-Platform-Id: <platform_id>
  X-Org-Id: <org_id>          (opcional)
  X-API-Key: <api_key>
"""
from __future__ import annotations

import hashlib
from fastapi import Header, HTTPException, status, Depends
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import get_settings
from app.models.platform import Platform

settings = get_settings()


def _hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


async def get_platform_context(
    x_platform_id: str = Header(..., alias="X-Platform-Id"),
    x_api_key: str = Header(..., alias="X-API-Key"),
    x_org_id: str | None = Header(None, alias="X-Org-Id"),
) -> dict:
    """
    Dependency que valida la API Key y carga la plataforma desde MongoDB.
    Inyectable en cualquier endpoint con `Depends(get_platform_context)`.
    """
    platform = await Platform.find_one(
        Platform.platform_id == x_platform_id,
        Platform.active == True,  # noqa: E712
    )

    if not platform:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plataforma '{x_platform_id}' no encontrada o inactiva.",
        )

    if platform.api_key_hash != _hash_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida.",
        )

    # Validar org si se pasa
    org = None
    if x_org_id:
        org = platform.get_org(x_org_id)
        if not org:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Organización '{x_org_id}' no encontrada en la plataforma.",
            )

    return {
        "platform": platform,
        "org": org,
        "platform_id": x_platform_id,
        "org_id": x_org_id,
    }


# ─── Admin dependency (usa el SECRET_KEY del .env) ───────────────────────────

async def require_admin(
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
) -> None:
    if x_admin_key != settings.secret_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acceso de administrador requerido.",
        )
