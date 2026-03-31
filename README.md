# Chatbot API

API de chatbot multi-plataforma con RAG dinámico sobre MongoDB. Cada plataforma tiene su propio índice de conocimiento construido a partir de sus colecciones de datos.

## Stack

- **FastAPI** — API REST
- **LangChain + ChromaDB** — RAG y búsqueda semántica
- **Google Gemini** — LLM y embeddings (también soporta OpenAI y Anthropic)
- **MongoDB** — datos de plataformas y persistencia de historial
- **Redis** — caché de sesiones activas

## Plataformas soportadas

| ID | Nombre | Colecciones |
|----|--------|-------------|
| `acho` | ACHO | events, attendees, speakers, agendas, users, members, highlights, news, posters |
| `gencampus` | GenCampus | activities, courseattendees, modules, organizations, organizationusers, quizzes, quizattempts, users |
| `genlive` | GenLive | events, attendees, speakers, agendas, users, members, highlights, news, posters |

## Instalación

```bash
# 1. Clonar y crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# 4. Levantar infraestructura (MongoDB + Redis)
docker-compose up -d

# 5. Registrar plataformas en la base de datos
python -m scripts.seed_platforms

# 6. Iniciar servidor
uvicorn app.main:app --reload --port 8000
```

## Flujo de indexación

Antes de usar el chat, cada plataforma debe ser indexada:

```bash
# Introspecciona MongoDB, genera embeddings y construye el índice vectorial
POST /platforms/{platform_id}/index
X-Admin-Key: your-secret-key-here
```

Verificar estado:
```bash
GET /platforms/{platform_id}/index/status
X-Admin-Key: your-secret-key-here
```

Para re-indexar con schema actualizado:
```bash
POST /platforms/{platform_id}/index?force_schema=true
```

## Uso del chat

```bash
POST /chat
X-Platform-Id: acho
X-API-Key: <api_key_de_la_plataforma>
Content-Type: application/json

{
  "message": "¿Cuáles son los próximos eventos?",
  "user_id": "672545c7778fcbf45a1f2c83",
  "user_name": "Juan Pérez",
  "org_id": "67604688eb1d8802a0483514"
}
```

## Widget embebible

El chat puede incrustarse en cualquier web via iframe:

```html
<iframe
  src="https://tu-api.com/widget/acho?api_key=xxx&user_id=yyy&user_name=Juan"
  width="420"
  height="620"
  frameborder="0"
  style="border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,.15);">
</iframe>
```

El widget incluye renderizado de markdown y menú lateral con historial de conversaciones.

## Endpoints principales

### Chat
| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/chat` | Enviar mensaje |
| `GET` | `/chat/sessions/{user_id}` | Sesiones del usuario |
| `GET` | `/chat/history/{user_id}` | Historial completo |
| `DELETE` | `/chat/history/{user_id}` | Borrar sesión activa |

### Plataformas (admin)
| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/platforms` | Crear plataforma |
| `GET` | `/platforms` | Listar plataformas |
| `POST` | `/platforms/{id}/index` | Indexar RAG |
| `GET` | `/platforms/{id}/index/status` | Estado del índice |
| `POST` | `/platforms/{id}/regenerate-key` | Regenerar API key |
| `POST` | `/platforms/{id}/db-connections` | Agregar conexión DB |

### Widget
| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/widget/{platform_id}` | HTML del chat widget |

## Scripts

```bash
# Registrar plataformas iniciales
python -m scripts.seed_platforms

# Actualizar colecciones y system prompts
python -m scripts.update_collections
```

## Estructura del proyecto

```
app/
├── api/
│   ├── chat.py          # Endpoints de chat e historial
│   ├── platforms.py     # Gestión de plataformas (admin)
│   └── widget.py        # Chat widget embebible
├── core/
│   ├── auth.py          # Autenticación por API key
│   └── config.py        # Configuración via .env
├── db/
│   ├── mongo_query.py   # Queries dinámicas con filtros generados por LLM
│   └── schema_introspector.py  # Introspección de esquemas MongoDB
├── models/
│   ├── conversation.py  # Modelo de historial (MongoDB)
│   └── platform.py      # Modelo de plataforma
├── rag/
│   └── pipeline.py      # Indexación y recuperación RAG
└── services/
    ├── chat_service.py  # Lógica principal del chat
    └── history_service.py  # Persistencia Redis + MongoDB
data/
├── chroma/              # Índices vectoriales por plataforma
└── schemas/             # Cache de esquemas MongoDB (JSON)
scripts/
├── seed_platforms.py    # Registro inicial de plataformas
└── update_collections.py  # Actualizar config de plataformas
```
