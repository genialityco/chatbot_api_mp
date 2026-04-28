"""
Pool de conexiones MongoDB centralizado.
PyMongo gestiona internamente el pool de conexiones por cliente.
Reutilizar la misma instancia evita el overhead de handshake TLS en cada request.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from pymongo import MongoClient

# Lock para thread-safety en la creación del cliente
_lock = threading.Lock()
_clients: dict[str, MongoClient] = {}

_DEFAULT_OPTS = dict(
    serverSelectionTimeoutMS=8000,
    connectTimeoutMS=8000,
    socketTimeoutMS=15000,
    retryWrites=True,
    maxPoolSize=10,
    minPoolSize=1,
)


def get_client(uri: str, **kwargs) -> MongoClient:
    """
    Retorna un MongoClient reutilizable para la URI dada.
    El cliente se crea una sola vez y se reutiliza en todos los requests.
    Thread-safe.
    """
    if uri not in _clients:
        with _lock:
            if uri not in _clients:
                opts = {**_DEFAULT_OPTS, **kwargs}
                _clients[uri] = MongoClient(uri, **opts)
                print(f"[mongo_pool] new client for {uri[:40]}...")
    return _clients[uri]


def get_db(uri: str, database: str, **kwargs):
    """Shortcut para obtener una DB directamente."""
    return get_client(uri, **kwargs)[database]


def close_all() -> None:
    """Cierra todos los clientes. Llamar al shutdown de la app."""
    with _lock:
        for client in _clients.values():
            try:
                client.close()
            except Exception:
                pass
        _clients.clear()
