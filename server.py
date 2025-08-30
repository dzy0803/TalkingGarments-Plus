# server.py
import uuid, time, asyncio
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Talosense Central")

# --- 内存态存储（PoC 够用；后续可换 SQLite） ---
AGENTS: Dict[str, dict] = {}                 # agent_id -> info
CONVERSATIONS: Dict[str, dict] = {}          # conv_id -> {started_at, agents:[...], meta}
MESSAGES: List[dict] = []                    # append-only
WS_DASHBOARD: List[WebSocket] = []           # 正在观看仪表盘的连接

def now_ts(): return int(time.time() * 1000)

async def broadcast(event: dict):
    dead = []
    for ws in WS_DASHBOARD:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try: WS_DASHBOARD.remove(ws)
        except: pass

# -------- 数据模型 --------
class AgentRegister(BaseModel):
    agent_id: str
    name: str
    type: str            # 'greeter'/'concierge'/'product' ...
    host: Optional[str] = None
    location: Optional[str] = None
    meta: Optional[dict] = None

class Heartbeat(BaseModel):
    agent_id: str
    status: str = "ok"
    meta: Optional[dict] = None

class StartConversation(BaseModel):
    agent_id: str
    meta: Optional[dict] = None

class MessageIn(BaseModel):
    conversation_id: str
    agent_id: str
    role: str           # 'user' | 'assistant' | 'event'
    text: str
    meta: Optional[dict] = None

class Handover(BaseModel):
    conversation_id: str
    from_agent: str
    to_agent: str
    reason: Optional[str] = None
    meta: Optional[dict] = None

# -------- API --------
@app.post("/register")
async def register(a: AgentRegister):
    AGENTS[a.agent_id] = {
        "agent_id": a.agent_id, "name": a.name, "type": a.type,
        "host": a.host, "location": a.location, "meta": a.meta,
        "last_seen": now_ts()
    }
    await broadcast({"type":"agent_registered","data":AGENTS[a.agent_id]})
    return {"ok": True}

@app.post("/heartbeat")
async def heartbeat(hb: Heartbeat):
    if hb.agent_id in AGENTS:
        AGENTS[hb.agent_id]["last_seen"] = now_ts()
        AGENTS[hb.agent_id]["status"] = hb.status
        AGENTS[hb.agent_id]["meta"] = hb.meta or AGENTS[hb.agent_id].get("meta")
    await broadcast({"type":"agent_heartbeat","data":{"agent_id":hb.agent_id,"ts":now_ts(),"status":hb.status}})
    return {"ok": True}

@app.post("/conversation/start")
async def start_conv(req: StartConversation):
    conv_id = str(uuid.uuid4())
    CONVERSATIONS[conv_id] = {"id":conv_id,"started_at":now_ts(),"agents":[req.agent_id], "meta": req.meta or {}}
    await broadcast({"type":"conversation_started","data":CONVERSATIONS[conv_id]})
    return {"ok": True, "conversation_id": conv_id}

@app.post("/message")
async def message(msg: MessageIn):
    if msg.conversation_id not in CONVERSATIONS:
        CONVERSATIONS[msg.conversation_id] = {"id":msg.conversation_id,"started_at":now_ts(),"agents":[msg.agent_id], "meta":{}}
    if msg.agent_id not in CONVERSATIONS[msg.conversation_id]["agents"]:
        CONVERSATIONS[msg.conversation_id]["agents"].append(msg.agent_id)

    row = {"ts": now_ts(), **msg.dict()}
    MESSAGES.append(row)
    await broadcast({"type":"message","data":row})
    return {"ok": True}

@app.post("/handover")
async def handover(h: Handover):
    if h.conversation_id in CONVERSATIONS and h.to_agent not in CONVERSATIONS[h.conversation_id]["agents"]:
        CONVERSATIONS[h.conversation_id]["agents"].append(h.to_agent)
    event = {"ts": now_ts(), **h.dict()}
    await broadcast({"type":"handover","data":event})
    return {"ok": True}

@app.get("/agents")
async def list_agents():
    return {"agents": list(AGENTS.values())}

@app.get("/conversations")
async def list_conversations():
    return {"conversations": list(CONVERSATIONS.values())}

@app.get("/messages")
async def list_messages(conversation_id: Optional[str] = None):
    if conversation_id:
        return {"messages": [m for m in MESSAGES if m["conversation_id"] == conversation_id]}
    return {"messages": MESSAGES}

# -------- 仪表盘（超轻量） --------
DASHBOARD_HTML = """
<!doctype html><meta charset="utf-8">
<title>Talosense Central Dashboard</title>
<style>
body{font:14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, "PingFang SC", "Microsoft Yahei";}
#wrap{display:grid; grid-template-columns: 320px 1fr; gap:16px; height:100vh; padding:12px; box-sizing:border-box;}
.card{border:1px solid #ddd; border-radius:12px; padding:12px; overflow:auto;}
.tag{display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid #aaa; margin-right:6px;}
.msg{border-bottom:1px dashed #eee; padding:6px 0;}
.role-user{color:#0b7;}
.role-assistant{color:#06c;}
.role-event{color:#a60;}
.small{color:#777; font-size:12px;}
</style>
<div id="wrap">
  <div class="card">
    <h3>Agents</h3>
    <div id="agents"></div>
    <hr>
    <h3>Conversations</h3>
    <div id="convs"></div>
  </div>
  <div class="card">
    <h3>Live Stream</h3>
    <div id="stream"></div>
  </div>
</div>
<script>
const $ = sel => document.querySelector(sel);
const agents = {};
const convs = {};

function ts2time(ts){ try{return new Date(ts).toLocaleTimeString();}catch(e){return "";} }

// ★ 新增：根据 agent_id 取职责（type）
function agentType(aid){
  const a = agents[aid];
  return a && a.type ? a.type : null;
}

// ★ 新增：把 assistant 显示成对应职责，其它角色保持原样
function displayRole(role, aid){
  if(role === "assistant"){
    return agentType(aid) || "assistant";
  }
  return role; // 'user' 和 'event' 原样
}

function renderAgents(){
  const el = $("#agents"); el.innerHTML="";
  Object.values(agents).sort((a,b)=>a.name.localeCompare(b.name)).forEach(a=>{
    const last = ts2time(a.last_seen||0);
    el.insertAdjacentHTML("beforeend",
      `<div class="msg"><b>${a.name}</b> <span class="tag">${a.type}</span><div class="small">${a.host||""} · ${a.location||""} · last_seen ${last}</div></div>`)
  });
}
function renderConvs(){
  const el=$("#convs"); el.innerHTML="";
  Object.values(convs).sort((a,b)=>b.started_at-a.started_at).slice(0,50).forEach(c=>{
    const t=ts2time(c.started_at);
    el.insertAdjacentHTML("beforeend",
      `<div class="msg"><b>${c.id.slice(0,8)}</b> <span class="small">${t}</span><div class="small">agents: ${c.agents.join(", ")}</div></div>`)
  });
}
function pushStream(line){
  const el=$("#stream");
  el.insertAdjacentHTML("afterbegin", line);
}

// ★ 修改：renderMessage 里用 displayRole(...) 代替原来的 role 文本
function renderMessage(data){
  const role = data.role;
  const who  = data.agent_id;
  const cid  = (data.conversation_id||"").slice(0,8);
  const text = (data.text||"").replaceAll("<","&lt;");
  const tstr = ts2time(data.ts);

  // 可选：显示说话时长（若客户端上报）
  let extra = "";
  if (role === "assistant" && data.meta && (data.meta.spoken_sec || data.meta.speak_sec)) {
    const sec = data.meta.spoken_sec || data.meta.speak_sec;
    extra = ` <span class="small">⏱ ${Number(sec).toFixed(1)}s</span>`;
  }
  if (role === "user" && data.meta && data.meta.utterance_sec) {
    const sec = data.meta.utterance_sec;
    extra = ` <span class="small">🗣 ${Number(sec).toFixed(1)}s</span>`;
  }

  // class 仍按原始 role（用于配色）；显示文本用 displayRole(role, who)
  const label = displayRole(role, who);

  pushStream(
    `<div class="msg role-${role}">
       <b>[${cid}] ${tstr} ${label}</b> <i>${who}</i>: ${text}${extra}
     </div>`
  );
}

async function boot(){
  const base = location.origin.replace("http","ws");
  const ws = new WebSocket(base+"/ws/dashboard");
  ws.onmessage = (ev)=>{
    const {type, data} = JSON.parse(ev.data);
    if(type==="agent_registered"||type==="agent_heartbeat"){
      agents[data.agent_id] = {...(agents[data.agent_id]||{}), ...data};
      renderAgents();
    }
    if(type==="conversation_started"){
      convs[data.id]=data; renderConvs();
      const tstr = ts2time(data.started_at);
      pushStream(`<div class="msg role-event">🆕 ${tstr} conversation <b>${data.id.slice(0,8)}</b> started by <b>${data.agents[0]}</b></div>`);
    }
    if(type==="handover"){
      const tstr = ts2time(data.ts);
      // ★ 可选：在交接里也显示职责
      const fromType = agentType(data.from_agent);
      const toType   = agentType(data.to_agent);
      const fromLabel = fromType ? `${data.from_agent} (${fromType})` : data.from_agent;
      const toLabel   = toType   ? `${data.to_agent} (${toType})`   : data.to_agent;
      pushStream(`<div class="msg role-event">🔀 ${tstr} handover ${fromLabel} → ${toLabel} <span class="small">${data.conversation_id.slice(0,8)}</span></div>`);
    }
    if(type==="message"){
      renderMessage(data);
    }
  };
  // 预加载历史
  const a = await (await fetch("/agents")).json(); a.agents.forEach(x=>agents[x.agent_id]=x); renderAgents();
  const c = await (await fetch("/conversations")).json(); c.conversations.forEach(x=>convs[x.id]=x); renderConvs();
  const m = await (await fetch("/messages")).json();
  m.messages.slice(-200).reverse().forEach(renderMessage);
}
boot();
</script>

"""
@app.get("/dashboard")
async def dashboard(): return HTMLResponse(DASHBOARD_HTML)

@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket):
    await ws.accept()
    WS_DASHBOARD.append(ws)
    try:
        while True:
            await ws.receive_text()   # 我们不接收消息，仅保持连接
    except WebSocketDisconnect:
        if ws in WS_DASHBOARD:
            WS_DASHBOARD.remove(ws)
