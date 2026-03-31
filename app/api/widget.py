"""
Widget de chat embebible via iframe.

Uso en el cliente:
    <iframe src="https://tu-api.com/widget/acho?api_key=xxx&user_id=yyy&user_name=Juan"
            width="400" height="600" frameborder="0">
    </iframe>
"""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse

from app.models.platform import Platform
from app.core.auth import _hash_api_key

router = APIRouter(prefix="/widget", tags=["Widget"])


@router.get("/{platform_id}", response_class=HTMLResponse)
async def chat_widget(
    platform_id: str,
    request: Request,
    api_key: str = "",
    user_id: str = "",
    user_name: str = "",
    org_id: str = "",
):
    platform = await Platform.find_one(
        Platform.platform_id == platform_id,
        Platform.active == True,
    )
    if not platform:
        raise HTTPException(404, "Plataforma no encontrada.")
    if not api_key or platform.api_key_hash != _hash_api_key(api_key):
        raise HTTPException(401, "API Key inválida.")

    base_url = str(request.base_url).rstrip("/")
    html = _build_widget_html(
        platform_id=platform_id,
        platform_name=platform.name,
        api_key=api_key,
        user_id=user_id,
        user_name=user_name,
        org_id=org_id,
        api_base_url=base_url,
    )
    return HTMLResponse(content=html)


def _build_widget_html(
    platform_id: str,
    platform_name: str,
    api_key: str,
    user_id: str,
    user_name: str,
    org_id: str,
    api_base_url: str,
) -> str:
    greeting = f"Hola{' ' + user_name if user_name else ''}! ¿En qué puedo ayudarte?"
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chat {platform_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f5; height: 100vh; display: flex; overflow: hidden; }}

  /* ── Sidebar ── */
  #sidebar {{ position: absolute; top: 0; left: 0; height: 100%; width: 220px;
              background: #12122a; color: #ccc; display: flex; z-index: 100;
              flex-direction: column; flex-shrink: 0;
              transform: translateX(-100%); transition: transform .25s; }}
  #sidebar.open {{ transform: translateX(0); }}
  #sidebar-overlay {{ position: absolute; inset: 0; background: rgba(0,0,0,.35);
                      z-index: 99; display: none; }}
  #sidebar-overlay.visible {{ display: block; }}
  #sidebar-header {{ padding: 14px 12px; font-size: 12px; font-weight: 600;
                     text-transform: uppercase; letter-spacing: .5px; color: #888;
                     border-bottom: 1px solid #ffffff18; display: flex;
                     justify-content: space-between; align-items: center; flex-shrink: 0; }}
  #new-chat {{ background: #ffffff18; border: none; color: #ccc; border-radius: 6px;
               padding: 4px 8px; font-size: 11px; cursor: pointer; white-space: nowrap; }}
  #new-chat:hover {{ background: #ffffff30; }}
  #sessions {{ flex: 1; overflow-y: auto; padding: 8px 0; }}
  .session-item {{ padding: 10px 12px; cursor: pointer; font-size: 13px;
                   border-left: 3px solid transparent; transition: background .15s;
                   white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .session-item:hover {{ background: #ffffff12; }}
  .session-item.active {{ background: #ffffff18; border-left-color: #4ade80; color: white; }}
  .session-date {{ font-size: 10px; color: #666; margin-top: 2px; }}

  /* ── Main ── */
  #main {{ flex: 1; display: flex; flex-direction: column; min-width: 0;
           position: relative; width: 100%; }}
  #header {{ background: #1a1a2e; color: white; padding: 12px 14px;
             display: flex; align-items: center; gap: 10px; flex-shrink: 0; }}
  #toggle-sidebar {{ background: none; border: none; color: white; cursor: pointer;
                     font-size: 18px; padding: 2px 6px; border-radius: 4px; }}
  #toggle-sidebar:hover {{ background: #ffffff20; }}
  #header .dot {{ width: 9px; height: 9px; background: #4ade80; border-radius: 50%; }}
  #header-title {{ font-weight: 600; font-size: 15px; flex: 1; }}

  #messages {{ flex: 1; overflow-y: auto; padding: 16px; display: flex;
               flex-direction: column; gap: 12px; }}
  .msg {{ max-width: 82%; padding: 10px 14px; border-radius: 16px;
          font-size: 14px; line-height: 1.55; word-wrap: break-word; }}
  .msg.bot {{ background: white; color: #1a1a2e; border-bottom-left-radius: 4px;
              box-shadow: 0 1px 3px rgba(0,0,0,.1); align-self: flex-start; }}
  .msg.user {{ background: #1a1a2e; color: white; border-bottom-right-radius: 4px;
               align-self: flex-end; }}
  .msg.bot p {{ margin-bottom: 6px; }}
  .msg.bot p:last-child {{ margin-bottom: 0; }}
  .msg.bot ul, .msg.bot ol {{ padding-left: 18px; margin: 6px 0; }}
  .msg.bot li {{ margin-bottom: 3px; }}
  .msg.bot strong {{ font-weight: 600; }}
  .msg.bot code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 4px;
                   font-family: monospace; font-size: 13px; }}
  .msg.bot pre {{ background: #f0f0f0; padding: 10px; border-radius: 8px;
                  overflow-x: auto; margin: 6px 0; }}
  .msg.bot pre code {{ background: none; padding: 0; }}
  .msg.bot h1, .msg.bot h2, .msg.bot h3 {{ font-weight: 600; margin: 8px 0 4px; }}
  .msg.typing {{ color: #888; font-style: italic; }}

  #form {{ display: flex; gap: 8px; padding: 12px; background: white;
           border-top: 1px solid #e5e5e5; flex-shrink: 0; }}
  #input {{ flex: 1; border: 1px solid #ddd; border-radius: 20px;
            padding: 10px 16px; font-size: 14px; outline: none; transition: border-color .2s; }}
  #input:focus {{ border-color: #1a1a2e; }}
  #send {{ background: #1a1a2e; color: white; border: none; border-radius: 50%;
           width: 40px; height: 40px; cursor: pointer; font-size: 18px;
           display: flex; align-items: center; justify-content: center;
           flex-shrink: 0; transition: opacity .2s; }}
  #send:disabled {{ opacity: .4; cursor: not-allowed; }}
  #send:hover:not(:disabled) {{ opacity: .85; }}
</style>
</head>
<body>

<div id="sidebar-overlay"></div>
<div id="sidebar">
  <div id="sidebar-header">
    <span>Conversaciones</span>
    <button id="new-chat">+ Nueva</button>
  </div>
  <div id="sessions"></div>
</div>

<div id="main">
  <div id="header">
    <button id="toggle-sidebar">&#9776;</button>
    <div class="dot"></div>
    <span id="header-title">{platform_name}</span>
  </div>
  <div id="messages">
    <div class="msg bot">{greeting}</div>
  </div>
  <form id="form">
    <input id="input" type="text" placeholder="Escribe tu mensaje..." autocomplete="off" />
    <button id="send" type="submit">&#9658;</button>
  </form>
</div>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
const API_BASE   = "{api_base_url}";
const PLATFORM   = "{platform_id}";
const API_KEY    = "{api_key}";
const USER_ID    = "{user_id}";
const USER_NAME  = "{user_name}";
const ORG_ID     = "{org_id}";

marked.setOptions({{ breaks: true, gfm: true }});

let SESSION_ID = crypto.randomUUID();
let activeSid  = null;

const messagesEl = document.getElementById("messages");
const form       = document.getElementById("form");
const input      = document.getElementById("input");
const sendBtn    = document.getElementById("send");
const sessionsEl = document.getElementById("sessions");
const sidebar    = document.getElementById("sidebar");

// ── Helpers ──────────────────────────────────────────────────────────────────

function addMessage(text, role, isTyping = false) {{
  const div = document.createElement("div");
  div.className = "msg " + role;
  if (role === "bot" && !isTyping) {{
    div.innerHTML = marked.parse(text);
  }} else {{
    div.textContent = text;
  }}
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}}

function setLoading(v) {{
  sendBtn.disabled = v;
  input.disabled   = v;
}}

function clearMessages() {{
  messagesEl.innerHTML = "";
  addMessage("{greeting}", "bot");
}}

function formatDate(iso) {{
  const d = new Date(iso);
  return d.toLocaleDateString("es", {{ day: "2-digit", month: "short" }}) +
         " " + d.toLocaleTimeString("es", {{ hour: "2-digit", minute: "2-digit" }});
}}

// ── Sidebar ───────────────────────────────────────────────────────────────────

async function loadSessions() {{
  if (!USER_ID) return;
  try {{
    const res = await fetch(API_BASE + "/chat/sessions/" + USER_ID, {{
      headers: {{ "X-Platform-Id": PLATFORM, "X-API-Key": API_KEY }}
    }});
    const sessions = await res.json();
    renderSessions(sessions);
  }} catch (e) {{ console.warn("No se pudo cargar historial", e); }}
}}

function renderSessions(sessions) {{
  sessionsEl.innerHTML = "";
  sessions.forEach(s => {{
    const div = document.createElement("div");
    div.className = "session-item" + (s.session_id === activeSid ? " active" : "");
    div.dataset.sid = s.session_id;
    div.innerHTML = `<div>${{s.title}}</div><div class="session-date">${{formatDate(s.created_at)}}</div>`;
    div.addEventListener("click", () => openSession(s));
    sessionsEl.appendChild(div);
  }});
}}

function openSession(session) {{
  activeSid  = session.session_id;
  SESSION_ID = session.session_id;
  messagesEl.innerHTML = "";
  session.turns.forEach(t => {{
    addMessage(t.user, "user");
    addMessage(t.assistant, "bot");
  }});
  // Marcar activo
  document.querySelectorAll(".session-item").forEach(el => {{
    el.classList.toggle("active", el.dataset.sid === activeSid);
  }});
  closeSidebar();
}}

// ── Toggle sidebar ────────────────────────────────────────────────────────────

const overlay  = document.getElementById("sidebar-overlay");

function openSidebar() {{
  sidebar.classList.add("open");
  overlay.classList.add("visible");
}}
function closeSidebar() {{
  sidebar.classList.remove("open");
  overlay.classList.remove("visible");
}}

document.getElementById("toggle-sidebar").addEventListener("click", () => {{
  sidebar.classList.contains("open") ? closeSidebar() : openSidebar();
}});
overlay.addEventListener("click", closeSidebar);

document.getElementById("new-chat").addEventListener("click", () => {{
  SESSION_ID = crypto.randomUUID();
  activeSid  = null;
  clearMessages();
  document.querySelectorAll(".session-item").forEach(el => el.classList.remove("active"));
  closeSidebar();
  input.focus();
}});

// ── Send message ──────────────────────────────────────────────────────────────

form.addEventListener("submit", async (e) => {{
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  input.value = "";
  addMessage(text, "user");
  setLoading(true);
  const typing = addMessage("Escribiendo...", "bot", true);

  try {{
    const res = await fetch(API_BASE + "/chat", {{
      method: "POST",
      headers: {{
        "Content-Type": "application/json",
        "X-Platform-Id": PLATFORM,
        "X-API-Key": API_KEY,
      }},
      body: JSON.stringify({{
        message: text,
        user_id: USER_ID || "anonymous",
        user_name: USER_NAME || null,
        org_id: ORG_ID || null,
        session_id: SESSION_ID,
      }}),
    }});
    const data = await res.json();
    typing.remove();
    addMessage(data.answer || "Sin respuesta.", "bot");
    // Recargar sesiones después de cada mensaje
    await loadSessions();
  }} catch (err) {{
    typing.remove();
    addMessage("Error al conectar con el servidor.", "bot");
  }} finally {{
    setLoading(false);
    input.focus();
  }}
}});

// ── Init ──────────────────────────────────────────────────────────────────────
loadSessions();
input.focus();
</script>
</body>
</html>"""
