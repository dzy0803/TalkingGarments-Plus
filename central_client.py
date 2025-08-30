# central_client.py
import os, threading, time, requests, socket, json

CENTRAL_URL = os.getenv("CENTRAL_URL", "http://<192.168.1.121>:8000")  # 记得在 .env 里配
CONV_FILE = "/tmp/current_conversation_id.txt"

def _post(path, payload):
    try:
        r = requests.post(f"{CENTRAL_URL}{path}", json=payload, timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[central] POST {path} failed: {e}")
        return {"ok": False}

def register(agent_id, name, type, location=None, meta=None):
    host = socket.gethostname()
    return _post("/register", {"agent_id":agent_id,"name":name,"type":type,"host":host,"location":location,"meta":meta or {}})

def heartbeat_loop(agent_id, interval=10, meta_fn=None):
    def _loop():
        while True:
            meta = meta_fn() if meta_fn else None
            _post("/heartbeat", {"agent_id":agent_id,"status":"ok","meta":meta})
            time.sleep(interval)
    t = threading.Thread(target=_loop, daemon=True); t.start(); return t

def start_conversation(agent_id, meta=None):
    resp = _post("/conversation/start", {"agent_id":agent_id, "meta": meta or {}})
    if resp.get("ok"):
        conv_id = resp["conversation_id"]
        try: open(CONV_FILE,"w").write(conv_id)
        except: pass
        return conv_id
    return None

def load_conversation_id():
    try:
        return open(CONV_FILE).read().strip()
    except:
        return None

def save_conversation_id(cid: str):
    try: open(CONV_FILE,"w").write(cid)
    except: pass

def log_message(conversation_id, agent_id, role, text, meta=None):
    if not conversation_id: return {"ok": False}
    return _post("/message", {"conversation_id":conversation_id,"agent_id":agent_id,"role":role,"text":text,"meta":meta or {}})

def handover(conversation_id, from_agent, to_agent, reason=None, meta=None):
    if not conversation_id: return {"ok": False}
    return _post("/handover", {"conversation_id":conversation_id,"from_agent":from_agent,"to_agent":to_agent,"reason":reason,"meta":meta or {}})
