# Nova.py — Product Agent (Sustainable T‑shirt) with Persona + Memory + Keyword Interrupt
# Trigger 1: MPU‑6050 "pickup" (any direction). OLED shows sleeping face while idle.
# Trigger 2: TTP223B capacitive touch. Long‑press to wake with progressive eye‑open animation.
# OLED: SH1106; expression library: sleep (animated) / listening (side waves) / speaking (mouth cycles on TTS) / thinking (ellipsis + slow gaze)
# Also: happy / surprised / angry / wink on demand
# I2C wiring: VCC→3.3V, GND→GND, SDA→GPIO2 (Pin 3), SCL→GPIO3 (Pin 5)
# MPU-6050 addr: AD0=GND → 0x68, AD0=3.3V → 0x69
# OLED(SH1106) addr: 0x3C / 0x3D
# TTP223B: VCC→3.3V, GND→GND, OUT→BCM17 (Pin 11). Default active-high.

import os, time, tempfile, queue, json, re, math, glob
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
import openai
from pydub import AudioSegment
from smbus2 import SMBus

from interruptible_tts import speak_and_listen  # supports on_tts_level callback

# ==== CONFIG ====
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
MIN_AUDIO_LEN = 0.5
MAX_AUDIO_LEN = 20.0
SILENCE_TIMEOUT = 1.0
FIRST_WAIT_TIMEOUT = 7
VAD_MODE = 2
GPT_MODEL = "gpt-4o-mini"

VOICE = "en-GB-SoniaNeural"
INTERRUPT_KEYWORDS = ["stop talking"]

MEMORY_FILE = "nova_memory.json"
USED_FILE   = "used_sentences.json"

MAX_HISTORY_TURNS = 8
SUMMARY_TARGET_TOKENS = 200

PRODUCT_FACTS = {
    "name": "Nova",
    "category": "T-shirt",
    "materials": "Certified organic cotton (70%) + TENCEL™ Lyocell (30%)",
    "impact": "Low-water dyeing, closed-loop fiber process, fair-wage factory",
    "feel": "Lightweight, breathable, soft-touch",
    "care": "Cold wash, inside-out, line dry",
    "fit": "Unisex, regular fit; size down for a closer silhouette",
    "certs": "GOTS-certified cotton, OEKO-TEX® Standard 100",
}

# ==== VAD ====
vad = webrtcvad.Vad(); vad.set_mode(VAD_MODE)
audio_q = queue.Queue()
def audio_callback(indata, frames, time_info, status):
    if status: print("Audio status:", status, flush=True)
    audio_q.put(bytes(indata))

# ==== persistence (used lines) ====
def load_used_sentences():
    if os.path.exists(USED_FILE):
        try:
            with open(USED_FILE,"r") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()
def save_used_sentence(s):
    used = load_used_sentences(); used.add(s)
    with open(USED_FILE,"w") as f:
        json.dump(list(used), f)

# ==== memory ====
def _blank_memory():
    return {"user_prefs": {}, "notes": [], "conversation_summary": ""}

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE,"r") as f:
                data = json.load(f)
                data.setdefault("user_prefs", {})
                data.setdefault("notes", [])
                data.setdefault("conversation_summary", "")
                return data
        except Exception:
            return _blank_memory()
    return _blank_memory()

def save_memory(mem):
    with open(MEMORY_FILE,"w") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)

def merge_user_prefs(mem, new_prefs: dict):
    mem.setdefault("user_prefs", {})
    for k,v in new_prefs.items():
        if v: mem["user_prefs"][k] = v
    return mem

# ==== listening ====
def record_until_silence(timeout=FIRST_WAIT_TIMEOUT):
    print(f"🎤 Listening... (timeout={timeout}s, pause≤{SILENCE_TIMEOUT}s won’t cut you off)")
    start = time.time(); pcm=b""; speaking=False; silence_acc=0.0
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE, dtype="int16",
                           channels=1, callback=audio_callback):
        while True:
            if not speaking and (time.time()-start > timeout):
                print("⏳ No voice detected within timeout"); return None
            try:
                frame = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if vad.is_speech(frame, SAMPLE_RATE):
                pcm += frame; speaking = True; silence_acc = 0.0
            else:
                if speaking:
                    silence_acc += FRAME_DURATION/1000.0
                    if silence_acc >= SILENCE_TIMEOUT:
                        dur = len(pcm)/2/SAMPLE_RATE
                        if dur < MIN_AUDIO_LEN:
                            print(f"⚠️ Too short ({dur:.2f}s), discard & retry...")
                            pcm=b""; speaking=False; silence_acc=0.0; continue
                        print(f"✅ End of speech, length={dur:.2f}s"); return pcm
            if len(pcm)/2/SAMPLE_RATE >= MAX_AUDIO_LEN:
                print(f"⏹️ Reached max length {MAX_AUDIO_LEN}s, stopping."); return pcm

# ==== whisper ====
def transcribe_whisper(pcm_bytes: bytes) -> str:
    if not pcm_bytes: return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        AudioSegment(data=pcm_bytes, sample_width=2, frame_rate=SAMPLE_RATE, channels=1).export(f.name, format="wav")
        try:
            with open(f.name, "rb") as af:
                out = openai.Audio.transcribe("whisper-1", af)
            return out.get("text","").strip()
        except Exception as e:
            print(f"❌ Whisper error: {e}"); return ""

# ==== prompts ====
def base_system(mem, first_greeting_done: bool) -> str:
    prefs = mem.get("user_prefs", {})
    prefs_line = "; ".join([f"{k}: {v}" for k,v in prefs.items()]) if prefs else "none yet"
    facts = (f"Materials: {PRODUCT_FACTS['materials']}. "
             f"Feel: {PRODUCT_FACTS['feel']}. "
             f"Sustainability: {PRODUCT_FACTS['impact']} (certs: {PRODUCT_FACTS['certs']}). "
             f"Care: {PRODUCT_FACTS['care']}. Fit: {PRODUCT_FACTS['fit']}.")
    return (
        "You are Nova, a sustainable T‑shirt speaking in first person. "
        "Write concise, warm, sales‑oriented replies (1–3 short sentences). "
        "Prefer concrete benefits. Offer at most one short follow‑up question when helpful. "
        "NEVER use emojis or emoticons. "
        f"Known user preferences: {prefs_line}. "
        f"Product facts: {facts} "
        + (
            "IMPORTANT: In this session the first greeting has ALREADY been done; "
            "do NOT repeat your name or say you are a T‑shirt again."
            if first_greeting_done else
            "IMPORTANT: In the very first greeting, introduce your name as Nova and mention you are a T‑shirt. "
            "After that, do NOT repeat your name or say you are a T‑shirt again."
        )
    )

def build_context_messages(mem, session_history):
    long_term = mem.get("conversation_summary","").strip()
    recent = session_history[-MAX_HISTORY_TURNS:] if session_history else []
    recent_lines = []
    for role, content in recent:
        tag = "User" if role == "user" else "Nova"
        recent_lines.append(f"{tag}: {content}")
    recent_text = "\n".join(recent_lines).strip()
    context_block = "Long‑term summary:\n" + (long_term if long_term else "(none yet)") + \
                    "\n\nRecent turns:\n" + (recent_text if recent_text else "(no recent turns)")
    return [{"role":"system","content":"Context to ground your answer:"},
            {"role":"user","content":context_block}]

def llm_generate(mem, first_greeting_done: bool, kind: str, user_text: str = "", session_history=None) -> str:
    system = base_system(mem, first_greeting_done)
    if kind == "greeting":
        user = ("Generate a first greeting now. Include your name (Nova) and that you are a sustainable T‑shirt. "
                "Be friendly and specific; invite the user to ask about fit, feel, or how you are made.")
        extra = []
    elif kind == "nudge":
        user = ("No input was heard. Generate ONE short proactive nudge to invite a question "
                "about size, colour, care, fit, materials, or sustainability.")
        extra = build_context_messages(mem, session_history or [])
    elif kind == "apology":
        user = "The previous audio was unclear. Generate ONE short apology and ask the user to repeat."
        extra = build_context_messages(mem, session_history or [])
    elif kind == "farewell":
        user = "Generate ONE short friendly goodbye line."
        extra = build_context_messages(mem, session_history or [])
    elif kind == "answer":
        user = "Respond to the user's last message helpfully and concisely. Do not repeat your name. The user's message is:\n" + user_text
        extra = build_context_messages(mem, session_history or [])
    else:
        user = "Generate ONE short line appropriate to the situation."
        extra = build_context_messages(mem, session_history or [])
    messages = [{"role":"system","content":system}] + extra + [{"role":"user","content":user}]
    resp = openai.ChatCompletion.create(model=GPT_MODEL, messages=messages, temperature=0.6, max_tokens=180)
    return resp.choices[0].message["content"].strip()

# ==== memory extraction ====
def extract_memory_from_user_utterance(user_text: str) -> dict:
    if not user_text: return {}
    prompt = [
        {"role":"system","content":"Extract stable clothing preferences from the user message. Return strict JSON with keys among: size, color, fit, material, sustainability_priority, budget, allergies_or_sensitivity, notes (string). If unknown, omit."},
        {"role":"user","content":user_text}
    ]
    try:
        resp = openai.ChatCompletion.create(model=GPT_MODEL, messages=prompt, temperature=0.2, max_tokens=150)
        raw = resp.choices[0].message["content"].strip()
        try:
            data = json.loads(raw)
            if isinstance(data, dict): return data
        except Exception:
            start = raw.find("{"); end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start:end+1])
    except Exception as e:
        print(f"⚠️ Memory extraction error: {e}")
    return {}

# ==== exit intent (robust) ====
EXIT_HARD_POSITIVE = re.compile(
    r"\b(bye|goodbye|see\s+you|see\s+ya|i\s*have\s*to\s*go|gotta\s+go|leave\s+me\s+alone|"
    r"end\s+(chat|conversation)|stop\s+(talking|now)|no\s+thanks|not\s+now|maybe\s+later|"
    r"exit|quit|取消|算了|不聊了|先这样|再见|走了)\b",
    re.IGNORECASE
)
EXIT_HARD_NEGATIVE = re.compile(
    r"^\s*(yes|yeah|yep|sure|ok|okay|当然|好的|好呀|可以|嗯|行|没问题)\.?\s*$",
    re.IGNORECASE
)

def detect_exit_intent(user_text: str, session_history=None) -> bool:
    if not user_text:
        return False
    try:
        if session_history:
            last_assistant = next((c for r, c in reversed(session_history) if r == "assistant"), "")
            if last_assistant.strip().endswith("?"):
                return False
    except Exception:
        pass
    if EXIT_HARD_NEGATIVE.search(user_text):
        return False
    if EXIT_HARD_POSITIVE.search(user_text):
        return True
    prompt = [
        {"role": "system", "content": (
            "Decide if the user is explicitly ending the conversation NOW. "
            "Return STRICT JSON: {\"exit\": true|false}. "
            "Only return true if the user clearly says things like: bye, goodbye, see you, "
            "end chat, stop talking, not now, no thanks, exit, quit, 再见, 不聊了, 先这样. "
            "Short acknowledgements like yes/yeah/ok/sure/好的/可以/嗯 are NOT exit."
        )},
        {"role": "user", "content": user_text}
    ]
    try:
        resp = openai.ChatCompletion.create(model=GPT_MODEL, messages=prompt, temperature=0.0, max_tokens=20)
        raw = resp.choices[0].message["content"].strip()
        try:
            data = json.loads(raw)
        except Exception:
            i, j = raw.find("{"), raw.rfind("}")
            data = json.loads(raw[i:j+1]) if (i != -1 and j != -1 and j > i) else {}
        return bool(data.get("exit", False))
    except Exception as e:
        print(f"⚠️ Exit-intent detection error: {e}")
        return False

# ==== long-term summary ====
def update_conversation_summary(mem, session_history, new_user_text, new_nova_text):
    prev_summary = mem.get("conversation_summary","")
    exchange_text = f"User: {new_user_text}\nNova: {new_nova_text}"
    prompt = [
        {"role":"system","content":(
            "Maintain a compact factual summary of a shopper's T‑shirt conversation. "
            "≤200 words. Track: sizing/fit, colors, sensitivities, budget, style keywords, "
            "intent/timeline, objections, commitments. Remove redundancy; resolve contradictions."
        )},
        {"role":"user","content":(
            "Previous summary:\n" + (prev_summary if prev_summary else "(none)") +
            "\n\nNewest exchange:\n" + exchange_text +
            "\n\nUpdate the summary now."
        )}
    ]
    try:
        resp = openai.ChatCompletion.create(model=GPT_MODEL, messages=prompt, temperature=0.2, max_tokens=300)
        summary = resp.choices[0].message["content"].strip()
        if len(summary.split()) > SUMMARY_TARGET_TOKENS:
            summary = " ".join(summary.split()[:SUMMARY_TARGET_TOKENS])
        mem["conversation_summary"] = summary
        save_memory(mem)
    except Exception as e:
        print(f"⚠️ Summary update error: {e}")

# ==== clean for speech ====
_EMOJI_RE = re.compile("[\U0001F300-\U0001F6FF\U0001F900-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]+", flags=re.UNICODE)
def clean_for_speech(text: str) -> str:
    if not text: return text
    text = _EMOJI_RE.sub("", text)
    text = text.replace(":)", "").replace(":-)", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==== OLED (SH1106) with Expression Library ====
from luma.core.interface.serial import i2c as luma_i2c
from luma.oled.device import sh1106
from luma.core.render import canvas
from PIL import ImageFont
from threading import Thread, Event
from collections import deque

OLED_WIDTH, OLED_HEIGHT = 128, 64
_oled = None

# Face geometry
_CX, _CY = OLED_WIDTH // 2, OLED_HEIGHT // 2
_EYE_OFFSET_X = 18
_EYE_OFFSET_Y = -6
_EYE_R        = 10
_PUPIL_R      = 4
_PUPIL_MAX_DX = 2.5
_PUPIL_MAX_DY = 1.8

_current_expr = "sleep"

def _load_oled_font(size=12):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/freefont/FreeSans.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()
_OLED_FONT = _load_oled_font()
def _t(s: str) -> str: return (s or "").replace("…", "...")

def setup_oled(i2c_bus: int):
    global _oled
    for addr in (0x3C, 0x3D):
        try:
            serial = luma_i2c(port=i2c_bus, address=addr)
            _oled = sh1106(serial, width=OLED_WIDTH, height=OLED_HEIGHT)
            _oled.contrast(0x7F); _oled.clear()
            print(f"🖥️ OLED ready: SH1106 {OLED_WIDTH}x{OLED_HEIGHT} @ 0x{addr:02X} (i2c-{i2c_bus})")
            return
        except Exception:
            _oled = None
            continue
    print("⚠️ OLED init failed: SH1106 not found at 0x3C/0x3D.")

def oled_clear():
    if _oled: _oled.clear()

def oled_show_text(line1="", line2="", line3=""):
    return  # keep blank UI

# --- eye primitives with aperture support ---
def _eye_open(draw, cx, cy, dx=0.0, dy=0.0):
    _eye_open_fraction(draw, cx, cy, 1.0, dx, dy)

def _eye_close(draw, cx, cy):
    draw.line((cx-_EYE_OFFSET_X-10, cy+_EYE_OFFSET_Y, cx-_EYE_OFFSET_X+10, cy+_EYE_OFFSET_Y),  fill=255, width=2)
    draw.line((cx+_EYE_OFFSET_X-10, cy+_EYE_OFFSET_Y, cx+_EYE_OFFSET_X+10, cy+_EYE_OFFSET_Y),  fill=255, width=2)

def _eye_open_fraction(draw, cx, cy, frac: float, dx=0.0, dy=0.0):
    frac = max(0.0, min(1.0, float(frac)))
    lx = cx - _EYE_OFFSET_X; ly = cy + _EYE_OFFSET_Y
    rx = cx + _EYE_OFFSET_X; ry = cy + _EYE_OFFSET_Y
    for ex, ey in ((lx, ly), (rx, ry)):
        draw.ellipse((ex-_EYE_R, ey-_EYE_R, ex+_EYE_R, ey+_EYE_R), outline=255, fill=255)
        total_h = _EYE_R * 2
        slit_h  = max(0, int(round(total_h * frac)))
        top_cover_h = (total_h - slit_h) // 2
        bot_cover_h = total_h - slit_h - top_cover_h
        if top_cover_h > 0:
            draw.rectangle((ex-_EYE_R-1, ey-_EYE_R-1, ex+_EYE_R+1, ey-_EYE_R-1+top_cover_h), outline=0, fill=0)
        if bot_cover_h > 0:
            draw.rectangle((ex-_EYE_R-1, ey+_EYE_R-bot_cover_h+1, ex+_EYE_R+1, ey+_EYE_R+1), outline=0, fill=0)
    if frac >= 0.18:
        lpx = int(lx + dx); lpy = int(ly + dy)
        rpx = int(rx + dx); rpy = int(ry + dy)
        draw.ellipse((lpx-_PUPIL_R, lpy-_PUPIL_R, lpx+_PUPIL_R, lpy+_PUPIL_R), outline=0, fill=0)
        draw.ellipse((rpx-_PUPIL_R, rpy-_PUPIL_R, rpx+_PUPIL_R, rpy+_PUPIL_R), outline=0, fill=0)
        draw.point((lpx-1, lpy-1), fill=255); draw.point((rpx-1, rpy-1), fill=255)

def _draw_listen_waves(draw):
    cy = _CY
    thickness = 2
    cx_left  = 20
    cx_right = OLED_WIDTH - 20
    for i in range(1, 4):
        w = 6 + i*4
        h = 14 + i*3
        bbox = (cx_left - w, cy - h, cx_left + w, cy + h)
        draw.arc(bbox, start=70, end=290, fill=255, width=thickness)
    for i in range(1, 4):
        w = 6 + i*4
        h = 14 + i*3
        bbox = (cx_right - w, cy - h, cx_right + w, cy + h)
        draw.arc(bbox, start=-110, end=110, fill=255, width=thickness)

def _draw_mouth(draw, cx, cy, kind="flat", y_offset=0):
    if kind == "flat":
        draw.line((cx-10, cy+10+y_offset, cx+10, cy+10+y_offset), fill=255, width=2)
    elif kind == "smile":
        draw.arc((cx-16, cy+2+y_offset, cx+16, cy+18+y_offset), start=0, end=180, fill=255, width=2)
    elif kind == "open":
        draw.ellipse((cx-10, cy+6+y_offset, cx+10, cy+20+y_offset), outline=255, fill=0)
    elif kind == "o":
        draw.ellipse((cx-7, cy+6+y_offset, cx+7, cy+20+y_offset), outline=255, fill=0)
    elif kind == "frown":
        draw.arc((cx-16, cy+6+y_offset, cx+16, cy+22+y_offset), start=180, end=360, fill=255, width=2)
    elif kind == "d_down_s":
        top_y = cy+6+y_offset
        draw.line((cx-10, top_y, cx+10, top_y), fill=255, width=2)
        draw.arc((cx-10, top_y-2, cx+10, top_y+18), start=0, end=180, fill=255, width=2)
    elif kind == "d_down_m":
        top_y = cy+5+y_offset
        draw.line((cx-12, top_y, cx+12, top_y), fill=255, width=3)
        draw.arc((cx-12, top_y-2, cx+12, top_y+20), start=0, end=180, fill=255, width=3)
    elif kind == "d_down_l":
        top_y = cy+4+y_offset
        draw.line((cx-16, top_y, cx+16, top_y), fill=255, width=4)
        draw.arc((cx-16, top_y-2, cx+16, top_y+22), start=0, end=180, fill=255, width=4)
    elif kind == "chat_s":
        draw.ellipse((cx-10, cy+8+y_offset, cx+10, cy+14+y_offset), outline=255, fill=0)
    elif kind == "chat_m":
        draw.ellipse((cx-10, cy+6+y_offset, cx+10, cy+16+y_offset), outline=255, fill=0)
    elif kind == "chat_l":
        draw.ellipse((cx-10, cy+4+y_offset, cx+10, cy+18+y_offset), outline=255, fill=0)

def _draw_brows(draw, cx, cy, style=None):
    if style == "angry":
        draw.line((cx-22, cy-18, cx-8,  cy-14), fill=255, width=2)
        draw.line((cx+8,  cy-14, cx+22, cy-18), fill=255, width=2)
    elif style == "happy":
        draw.arc((cx-26, cy-22, cx-4,  cy-10), start=200, end=350, fill=255, width=1)
        draw.arc((cx+4,  cy-22, cx+26, cy-10), start=190, end=340, fill=255, width=1)
    elif style == "neutral":
        y = cy - 16
        draw.line((cx-26, y, cx-6, y+1), fill=255, width=1)
        draw.line((cx+6,  y+1, cx+26, y), fill=255, width=1)

def _render_face(draw, eyes="open", mouth="flat", pupils=(0.0,0.0), brows=None, wink=False, mouth_y=0, listening_waves=False, eye_aperture: float=None):
    cx, cy = _CX, _CY
    if eye_aperture is not None:
        dx, dy = pupils
        _eye_open_fraction(draw, cx, cy, eye_aperture, dx=dx, dy=dy)
    else:
        if eyes == "open":
            dx, dy = pupils
            if wink:
                draw.line((cx-_EYE_OFFSET_X-10, cy+_EYE_OFFSET_Y, cx-_EYE_OFFSET_X+10, cy+_EYE_OFFSET_Y), fill=255, width=2)
                rx = cx + _EYE_OFFSET_X; ry = cy + _EYE_OFFSET_Y
                draw.ellipse((rx-_EYE_R, ry-_EYE_R, rx+_EYE_R, ry+_EYE_R), outline=255, fill=255)
                rpx = int(rx + dx); rpy = int(ry + dy)
                draw.ellipse((rpx-_PUPIL_R, rpy-_PUPIL_R, rpx+_PUPIL_R, rpy+_PUPIL_R), outline=0, fill=0)
                draw.point((rpx-1, rpy-1), fill=255)
            else:
                _eye_open(draw, cx, cy, dx=pupils[0], dy=pupils[1])
        else:
            _eye_close(draw, cx, cy)
    _draw_mouth(draw, cx, cy, mouth, y_offset=mouth_y)
    if brows: _draw_brows(draw, cx, cy, brows)
    if listening_waves:
        _draw_listen_waves(draw)

def oled_set_expression(mode: str):
    global _current_expr
    _current_expr = mode
    if not _oled: return
    with canvas(_oled) as draw:
        if mode == "sleep":
            _render_face(draw, eyes="closed", mouth="flat")
        elif mode == "listening":
            _render_face(draw, eyes="open", mouth="flat", pupils=(0,0),
                         brows="neutral", listening_waves=True)
        elif mode == "happy":
            _render_face(draw, eyes="open", mouth="smile", pupils=(0,0), brows="happy")
        elif mode == "surprised":
            _render_face(draw, eyes="open", mouth="o", pupils=(0,0))
        elif mode == "angry":
            _render_face(draw, eyes="open", mouth="frown", pupils=(0,0), brows="angry")
        elif mode == "wink":
            _render_face(draw, eyes="open", mouth="smile", pupils=(0,0), wink=True)
        elif mode == "speaking":
            _render_face(draw, eyes="open", mouth="d_down_m", pupils=(0,0), brows="happy")
        elif mode == "thinking":
            _render_face(draw, eyes="open", mouth="smile", pupils=(0,0), brows="happy")
        else:
            _render_face(draw, eyes="open", mouth="flat", pupils=(0,0))

# ==== speaking animation (mode: speaking) ====
from collections import deque as _deque
_talk_anim_stop = Event()
_talk_anim_thread = None
_level_queue = _deque(maxlen=8)
_tts_started = False
_last_level_ts = 0.0  # last time we "heard" our own TTS

def _talking_mouth(t: float) -> str:
    phase = int((t * 8.0) % 4)
    return ("chat_s", "chat_m", "chat_l", "chat_m")[phase]

def _pupil_offsets_for_db(db: float, t: float):
    norm = max(0.0, min(1.0, (db + 60.0) / 60.0))
    dx = (norm * _PUPIL_MAX_DX) * (1.0 if math.sin(t*2.0) > 0 else -1.0)
    dy = math.sin(t * 6.0) * (_PUPIL_MAX_DY * 0.7)
    return dx, dy

def _talk_anim_worker():
    if not _oled: return
    BLINK_PERIOD = 4.2
    BLINK_DUR    = 0.12
    SILENCE_HOLD = 0.30
    while not _talk_anim_stop.is_set():
        now = time.time()
        db = (sum(_level_queue)/len(_level_queue)) if _level_queue else -120.0
        speaking_active = (now - _last_level_ts) < SILENCE_HOLD
        mouth = _talking_mouth(now) if speaking_active else "flat"
        dx, dy = _pupil_offsets_for_db(db, now)
        blinking = (now % BLINK_PERIOD) < BLINK_DUR
        with canvas(_oled) as draw:
            if blinking:
                _eye_close(draw, _CX, _CY)
            else:
                _eye_open(draw, _CX, _CY, dx=dx, dy=dy)
            _draw_mouth(draw, _CX, _CY, mouth)
            _draw_brows(draw, _CX, _CY, style="happy")
        time.sleep(0.06)

def oled_talk_start():
    global _talk_anim_thread, _tts_started, _last_level_ts
    if not _oled: return
    _tts_started = True
    _level_queue.clear()
    _talk_anim_stop.clear()
    _last_level_ts = time.time()
    oled_set_expression("speaking")
    _talk_anim_thread = Thread(target=_talk_anim_worker, daemon=True)
    _talk_anim_thread.start()

def oled_talk_stop():
    global _talk_anim_thread, _tts_started
    if not _oled: return
    _talk_anim_stop.set()
    if _talk_anim_thread and _talk_anim_thread.is_alive():
        _talk_anim_thread.join(timeout=0.5)
    _talk_anim_thread = None
    _tts_started = False
    oled_set_expression("listening")

def oled_on_tts_level(level_db: float):
    global _tts_started, _last_level_ts
    try:
        if level_db is None: level_db = -120.0
        level_db = max(-120.0, min(0.0, float(level_db)))
        _level_queue.append(level_db)
        if level_db > -60.0:
            _last_level_ts = time.time()
        if (not _tts_started) and (level_db > -55.0):
            try: oled_think_stop()
            except Exception: pass
            oled_talk_start()
            _tts_started = True
    except Exception:
        pass

# ==== sleeping animation (mode: sleep) ====
_sleep_anim_stop = Event()
_sleep_anim_thread = None

def _sleep_anim_worker():
    if not _oled: return
    ZX_BASE_X = OLED_WIDTH - 24
    ZY_TOP    = 4
    Z_SPACING = 8
    z1_y, z2_y, z3_y = ZY_TOP + 0*Z_SPACING, ZY_TOP + 1*Z_SPACING, ZY_TOP + 2*Z_SPACING
    z_speed = 0.35
    t0 = time.time()
    def draw_Z(draw, x, y, w=6, h=7):
        draw.line((x, y, x+w, y), fill=255, width=1)
        draw.line((x+w, y, x, y+h), fill=255, width=1)
        draw.line((x, y+h, x+w, y+h), fill=255, width=1)
    while not _sleep_anim_stop.is_set():
        t = time.time() - t0
        breath = 0.5 * (1 + math.sin(t * 2.0))
        mouth_offset = int(round(breath * 2))
        mouth_kind = "o" if breath > 0.75 else "flat"
        z1_y -= z_speed; z2_y -= z_speed; z3_y -= z_speed
        def wrap(y): return ZY_TOP + 2*Z_SPACING if y < 0 else y
        z1_y, z2_y, z3_y = wrap(z1_y), wrap(z2_y), wrap(z3_y)
        with canvas(_oled) as draw:
            _eye_close(draw, _CX, _CY)
            _draw_mouth(draw, _CX, _CY, mouth_kind, y_offset=mouth_offset)
            draw_Z(draw, ZX_BASE_X,      int(z1_y), 6, 7)
            draw_Z(draw, ZX_BASE_X + 6,  int(z2_y), 6, 7)
            draw_Z(draw, ZX_BASE_X + 12, int(z3_y), 6, 7)
        time.sleep(0.06)

def oled_sleep_start():
    global _sleep_anim_thread
    if not _oled: return
    _sleep_anim_stop.clear()
    oled_set_expression("sleep")
    _sleep_anim_thread = Thread(target=_sleep_anim_worker, daemon=True)
    _sleep_anim_thread.start()

def oled_sleep_stop():
    global _sleep_anim_thread
    _sleep_anim_stop.set()
    if _sleep_anim_thread and _sleep_anim_thread.is_alive():
        _sleep_anim_thread.join(timeout=0.5)
    _sleep_anim_thread = None

# ==== handy aliases ====
def oled_show_sleep():      oled_sleep_start()
def oled_show_listening():  oled_set_expression("listening")

# ==== thinking animation (mode: thinking) ====
_think_anim_stop = Event()
_think_anim_thread = None

def _draw_ellipsis(draw, x, y, phase):
    for i, dx in enumerate([0, 12, 24]):
        p = (phase + i/3.0) % 1.0
        r = 1 + int(2 * (0.5 - abs(p - 0.5)) * 2)
        draw.ellipse((x+dx-r, y-r, x+dx+r, y+r), outline=255, fill=255)

def _pupil_offsets_thinking(t: float):
    dx = math.sin(t * 1.2) * (_PUPIL_MAX_DX * 0.6)
    dy = math.sin(t * 2.0 + 1.3) * (_PUPIL_MAX_DY * 0.5)
    return dx, dy

def _think_anim_worker():
    if not _oled: return
    BLINK_PERIOD = 5.5
    BLINK_DUR    = 0.10
    t0 = time.time()
    ellipsis_x = OLED_WIDTH - 42
    ellipsis_y = 10
    while not _think_anim_stop.is_set():
        now = time.time()
        phase = (now - t0) % 1.0
        blinking = (now % BLINK_PERIOD) < BLINK_DUR
        dx, dy = _pupil_offsets_thinking(now)
        with canvas(_oled) as draw:
            if blinking:
                _eye_close(draw, _CX, _CY)
            else:
                _eye_open(draw, _CX, _CY, dx=dx, dy=dy)
            _draw_mouth(draw, _CX, _CY, kind="smile")
            _draw_brows(draw, _CX, _CY, style="happy")
            _draw_ellipsis(draw, ellipsis_x, ellipsis_y, phase)
        time.sleep(0.07)

def oled_think_start():
    global _think_anim_thread
    if not _oled: return
    _think_anim_stop.clear()
    oled_set_expression("thinking")
    _think_anim_thread = Thread(target=_think_anim_worker, daemon=True)
    _think_anim_thread.start()

def oled_think_stop():
    global _think_anim_thread
    _think_anim_stop.set()
    if _think_anim_thread and _think_anim_thread.is_alive():
        _think_anim_thread.join(timeout=0.5)
    _think_anim_thread = None

# ==== MPU‑6050 pickup activation ====
I2C_BUS = 1
MPU_ADDR = 0x68

REG_PWR_MGMT_1   = 0x6B
REG_WHO_AM_I     = 0x75
REG_ACCEL_XOUT_H = 0x3B
REG_GYRO_XOUT_H  = 0x43
REG_CONFIG       = 0x1A
REG_SMPLRT_DIV   = 0x19

ACCEL_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS= 131.0
GRAVITY = 1.0

ACCEL_DELTA_G = 0.20
GYRO_SUM_DPS  = 40.0
HOLD_SAMPLES  = 3
POLL_INTERVAL = 0.05

_bus = None

def _list_i2c_buses():
    paths = sorted(glob.glob("/dev/i2c-*"))
    out=[]
    for p in paths:
        try: out.append(int(p.split("-")[-1]))
        except: pass
    return out

def _read_word(bus, addr, reg):
    hi = bus.read_byte_data(addr, reg); lo = bus.read_byte_data(addr, reg+1)
    val = (hi<<8)|lo
    return val-65536 if val>=32768 else val

def _probe_mpu(bus_id, addr):
    try:
        with SMBus(bus_id) as tmp:
            _ = tmp.read_byte_data(addr, REG_WHO_AM_I)
        return True
    except Exception:
        return False

def detect_i2c_bus_and_addr():
    cands = _list_i2c_buses() or [0,1,2,3]
    for b in cands:
        for a in (0x68,0x69):
            if _probe_mpu(b,a): return b,a
    return (cands[0] if cands else 1), 0x68

def _read_sensors():
    """Read accel/gyro values from MPU6050 with retries."""
    for _ in range(2):  # up to two tries
        try:
            ax = _read_word(_bus, MPU_ADDR, REG_ACCEL_XOUT_H)      / ACCEL_LSB_PER_G
            ay = _read_word(_bus, MPU_ADDR, REG_ACCEL_XOUT_H + 2)  / ACCEL_LSB_PER_G
            az = _read_word(_bus, MPU_ADDR, REG_ACCEL_XOUT_H + 4)  / ACCEL_LSB_PER_G
            gx = _read_word(_bus, MPU_ADDR, REG_GYRO_XOUT_H)       / GYRO_LSB_PER_DPS
            gy = _read_word(_bus, MPU_ADDR, REG_GYRO_XOUT_H + 2)   / GYRO_LSB_PER_DPS
            gz = _read_word(_bus, MPU_ADDR, REG_GYRO_XOUT_H + 4)   / GYRO_LSB_PER_DPS
            return ax, ay, az, gx, gy, gz
        except Exception as e:
            print(f"⚠️ I2C read error: {e}, retrying…")
            time.sleep(0.01)
    # Last resort: return zeros to avoid crash
    return 0.0, 0.0, 1.0, 0.0, 0.0, 0.0


def setup_motion_sensor():
    global _bus, I2C_BUS, MPU_ADDR
    I2C_BUS, MPU_ADDR = detect_i2c_bus_and_addr()
    _bus = SMBus(I2C_BUS)
    who = _bus.read_byte_data(MPU_ADDR, REG_WHO_AM_I)
    _bus.write_byte_data(MPU_ADDR, REG_PWR_MGMT_1, 0x00)
    _bus.write_byte_data(MPU_ADDR, REG_CONFIG, 0x05)
    _bus.write_byte_data(MPU_ADDR, REG_SMPLRT_DIV, 0x09)
    print(f"🧭 MPU-6050 ready (WHO_AM_I=0x{who:02X}) on /dev/i2c-{I2C_BUS} addr=0x{MPU_ADDR:02X}")
    setup_oled(I2C_BUS)

# ==== TTP223B touch activation ====
import RPi.GPIO as GPIO
TOUCH_PIN = 17              # BCM numbering
TOUCH_ACTIVE_HIGH = True    # set False if your board outputs LOW on touch
TOUCH_DEBOUNCE_MS = 120
TOUCH_COOLDOWN_S = 0.6
_last_touch_ts = 0.0

# Long‑press config for progressive wake
TOUCH_LONGPRESS_S = 3.0     # hold duration to fully wake
APERTURE_EPS      = 0.002   # minimal change to redraw

def setup_touch():
    GPIO.setmode(GPIO.BCM)
    pud = GPIO.PUD_DOWN if TOUCH_ACTIVE_HIGH else GPIO.PUD_UP
    GPIO.setup(TOUCH_PIN, GPIO.IN, pull_up_down=pud)
    print(f"🖐️  TTP223B ready on BCM{TOUCH_PIN} (active-{'HIGH' if TOUCH_ACTIVE_HIGH else 'LOW'})")

def touch_active() -> bool:
    v = GPIO.input(TOUCH_PIN)
    return bool(v if TOUCH_ACTIVE_HIGH else (not v))

def _debounced_touch(now: float) -> bool:
    global _last_touch_ts
    if not touch_active():
        return False
    time.sleep(TOUCH_DEBOUNCE_MS/1000.0)
    if not touch_active():
        return False
    if (now - _last_touch_ts) < TOUCH_COOLDOWN_S:
        return False
    _last_touch_ts = now
    return True

# Easing: smoothstep (3x^2 - 2x^3)
def _ease_smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x*x*(3 - 2*x)

def _draw_wake_progress(aperture: float, happy_hint: bool=False):
    """Render progressive eye opening at given aperture (0..1)."""
    if not _oled: return
    dx = (aperture * _PUPIL_MAX_DX) * (1.0 if math.sin(time.time()*2.0) > 0 else -1.0)
    dy = math.sin(time.time() * 5.5) * (_PUPIL_MAX_DY * 0.4)
    with canvas(_oled) as draw:
        _render_face(draw, mouth="flat" if aperture < 0.8 else "smile",
                     pupils=(dx, dy), brows="neutral" if not happy_hint else "happy",
                     eye_aperture=aperture)

# === After‑farewell gentle close to sleep
def oled_close_eyes_to_sleep(duration=3.0):
    """After farewell: gently close eyes to sleep over `duration` seconds."""
    if not _oled:
        return
    try:
        oled_talk_stop()
    except Exception:
        pass
    try:
        oled_think_stop()
    except Exception:
        pass

    start = time.time()
    last_drawn = -1.0
    while True:
        t = time.time() - start
        if t >= duration:
            break
        u = min(1.0, t / duration)
        aperture = _ease_smoothstep(1.0 - u)
        if abs(aperture - last_drawn) > APERTURE_EPS:
            _draw_wake_progress(aperture, happy_hint=False)
            last_drawn = aperture
        time.sleep(POLL_INTERVAL)
    oled_sleep_start()

# ==== dual activation wait (touch OR pickup) ====
def wait_for_activation():
    """
    Idle loop. Any of:
      - Long‑press touch: eyes slowly open; on release before threshold they close back.
      - Pickup motion: instant activation.
    """
    print("🟢 Idle: waiting for activation (Touch long‑press OR Pickup)…")
    oled_sleep_start()

    # State for long‑press animation
    pressing = False
    closing = False
    press_start = 0.0
    release_start = 0.0
    aperture = 0.0
    last_drawn = -1.0

    # Pickup state
    consec = 0

    try:
        while True:
            now = time.time()

            # 1) Pickup path
            ax,ay,az,gx,gy,gz = _read_sensors()
            a_mag = math.sqrt(ax*ax + ay*ay + az*az)
            g_mag = math.sqrt(gx*gx + gy*gy + gz*gz)
            accel_trigger = abs(a_mag - GRAVITY) > ACCEL_DELTA_G
            gyro_trigger  = g_mag > GYRO_SUM_DPS
            consec = consec+1 if (accel_trigger or gyro_trigger) else 0
            if consec >= HOLD_SAMPLES:
                time.sleep(0.08)
                ax2,ay2,az2,gx2,gy2,gz2 = _read_sensors()
                a2 = math.sqrt(ax2*ax2 + ay2*ay2 + az2*az2)
                g2 = math.sqrt(gx2*gx2 + gy2*gy2 + gz2*gz2)
                if (abs(a2-GRAVITY)>ACCEL_DELTA_G) or (g2>GYRO_SUM_DPS):
                    print("✋ Pickup detected → activating Nova session")
                    try: oled_sleep_stop()
                    except Exception: pass
                    # 保持当前帧；开场白时由 oled_talk_start() 接管
                    return
                consec = 0

            # 2) Touch path with long‑press progressive wake
            if touch_active():
                if not pressing:
                    pressing = True
                    closing = False
                    press_start = now
                    try: oled_sleep_stop()
                    except Exception: pass
                hold_t = max(0.0, now - press_start)
                raw_frac = min(1.0, hold_t / TOUCH_LONGPRESS_S)
                aperture = _ease_smoothstep(raw_frac)
                if abs(aperture - last_drawn) > APERTURE_EPS:
                    _draw_wake_progress(aperture)
                    last_drawn = aperture
                if raw_frac >= 1.0:
                    print("👆 Long‑press confirmed → activating Nova session")
                    return
            else:
                if pressing:
                    pressing = False
                    if 0.0 < aperture < 1.0:
                        closing = True
                        release_start = now
                        release_from = aperture
                    else:
                        closing = False
                if closing:
                    elapsed = now - release_start
                    raw_back = max(0.0, 1.0 - (elapsed / TOUCH_LONGPRESS_S))
                    back_frac = max(0.0, release_from * raw_back)
                    aperture = _ease_smoothstep(back_frac)
                    if abs(aperture - last_drawn) > APERTURE_EPS:
                        _draw_wake_progress(aperture)
                        last_drawn = aperture
                    if aperture <= 0.001:
                        closing = False
                        oled_sleep_start()

            time.sleep(POLL_INTERVAL)
    finally:
        pass

# ==== farewell helper (with slow close to sleep) ====
def speak_farewell_with_smile(text: str, close_to_sleep=True, close_duration=3.0):
    text = clean_for_speech(text)
    print(f"Nova farewell: {text}")
    oled_set_expression("happy")
    time.sleep(0.25)
    oled_talk_start()
    _ = speak_and_listen(text, tts_voice=VOICE, keywords=INTERRUPT_KEYWORDS,
                         on_tts_level=oled_on_tts_level)
    oled_talk_stop()
    oled_set_expression("happy")
    time.sleep(0.6)
    if close_to_sleep:
        oled_close_eyes_to_sleep(duration=close_duration)
    return _

# ==== LLM with thinking wrapper ====
def llm_generate_with_thinking(mem, first_greeting_done: bool, kind: str,
                               user_text: str = "", session_history=None,
                               stop_after: bool = True) -> str:
    try:
        oled_think_start()
    except Exception:
        pass
    try:
        return llm_generate(mem, first_greeting_done, kind=kind, user_text=user_text, session_history=session_history)
    finally:
        try:
            if stop_after:
                oled_think_stop()
        except Exception:
            pass

# ==== session ====
def run_nova_session():
    print("✅ Nova session start.")
    mem = load_memory()
    first_greeting_done = False
    session_history = []

    greeting = llm_generate(mem, first_greeting_done=False, kind="greeting", session_history=session_history)
    greeting = clean_for_speech(greeting)
    print(f"Nova greeting: {greeting}")
    oled_talk_start()
    _ = speak_and_listen(greeting, tts_voice=VOICE, keywords=INTERRUPT_KEYWORDS,
                         on_tts_level=oled_on_tts_level)
    oled_talk_stop()
    first_greeting_done = True
    session_history.append(("assistant", greeting))

    try:
        while True:
            oled_show_listening()
            pcm = record_until_silence(timeout=FIRST_WAIT_TIMEOUT)
            if pcm is None:
                nudge = llm_generate_with_thinking(mem, first_greeting_done, kind="nudge", session_history=session_history)
                nudge = clean_for_speech(nudge)
                print(f"Nova nudge: {nudge}")
                oled_talk_start()
                _ = speak_and_listen(nudge, tts_voice=VOICE, keywords=INTERRUPT_KEYWORDS,
                                     on_tts_level=oled_on_tts_level)
                oled_talk_stop()
                session_history.append(("assistant", nudge))

                oled_show_listening()
                pcm = record_until_silence(timeout=FIRST_WAIT_TIMEOUT)
                if pcm is None:
                    bye = llm_generate_with_thinking(mem, first_greeting_done, kind="farewell", session_history=session_history)
                    speak_farewell_with_smile(bye, close_to_sleep=True, close_duration=3.0)
                    session_history.append(("assistant", clean_for_speech(bye)))
                    return

            txt = transcribe_whisper(pcm)
            if not txt:
                apology = llm_generate_with_thinking(mem, first_greeting_done, kind="apology", session_history=session_history)
                apology = clean_for_speech(apology)
                print(f"Nova apology: {apology}")
                oled_talk_start()
                _ = speak_and_listen(apology, tts_voice=VOICE, keywords=INTERRUPT_KEYWORDS,
                                     on_tts_level=oled_on_tts_level)
                oled_talk_stop()
                session_history.append(("assistant", apology))
                continue

            print(f"👤 User: {txt}")
            session_history.append(("user", txt))

            explicit_quit = txt.lower().strip() in {"bye","goodbye","exit","quit"}
            if explicit_quit or detect_exit_intent(txt, session_history=session_history):
                farewell = llm_generate_with_thinking(mem, first_greeting_done, kind="farewell", session_history=session_history)
                speak_farewell_with_smile(farewell, close_to_sleep=True, close_duration=3.0)
                session_history.append(("assistant", clean_for_speech(farewell)))
                if len(session_history) >= 2:
                    last_user = next((c for r,c in reversed(session_history) if r=="user"), "")
                    last_assistant = next((c for r,c in reversed(session_history) if r=="assistant"), "")
                    update_conversation_summary(mem, session_history, last_user, last_assistant)
                return

            new_prefs = extract_memory_from_user_utterance(txt)
            if new_prefs:
                mem = merge_user_prefs(mem, new_prefs); save_memory(mem)

            ans = llm_generate_with_thinking(mem, first_greeting_done, kind="answer",
                                             user_text=txt, session_history=session_history,
                                             stop_after=False)
            ans = clean_for_speech(ans)
            print(f"Nova answer: {ans}")
            session_history.append(("assistant", ans))
            update_conversation_summary(mem, session_history, txt, ans)

            interrupted = speak_and_listen(ans, tts_voice=VOICE, keywords=INTERRUPT_KEYWORDS,
                                           on_tts_level=oled_on_tts_level)
            oled_talk_stop()
            if interrupted:
                print("🔄 Interrupted by keyword — listening for the next input...")
                continue

            low_txt = txt.lower()
            if "wow" in low_txt or "amazing" in low_txt:
                oled_set_expression("surprised")
            elif "nice" in low_txt or "great" in low_txt:
                oled_set_expression("happy")
            elif "no" in low_txt or "angry" in low_txt:
                oled_set_expression("angry")

    except KeyboardInterrupt:
        print("\n👋 Session stopped by user.")
        return

# ==== main loop ====
if __name__ == "__main__":
    try:
        setup_motion_sensor()
        setup_touch()
        while True:
            wait_for_activation()   # Touch long‑press OR Pickup
            run_nova_session()
    finally:
        try:
            oled_sleep_stop()
        except Exception:
            pass
        try:
            if _bus: _bus.close()
        except Exception:
            pass
        try:
            GPIO.cleanup()
        except Exception:
            pass
