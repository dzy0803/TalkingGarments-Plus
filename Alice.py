# Alice.py — 保留原逻辑 + 接入 central_client（注册/心跳/对话日志）
import sys, os, json, tempfile, asyncio, queue, socket, time
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
import openai
from pydub import AudioSegment
from interruptible_tts import speak_and_listen

# ===== 接入 central_client（你的文件） =====
# 如果已在环境里 export CENTRAL_URL，会覆盖下面的默认
os.environ.setdefault("CENTRAL_URL", "http://192.168.1.121:8000")
import central_client as cc

# === CONFIG（原有） ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
MIN_AUDIO_LEN = 0.5
MAX_AUDIO_LEN = 20.0
SILENCE_TIMEOUT = 0.8

# 🔑 Keyword(s) that interrupt Alice while speaking（原有）
INTERRUPT_KEYWORDS = ["stop talking"]  # e.g. ["stop talking","stop","be quiet"]

vad = webrtcvad.Vad()
vad.set_mode(2)

audio_queue = queue.Queue()
USED_FILE = "used_sentences.json"

# === 运行元信息（新增） ===
AGENT_ID = os.getenv("AGENT_ID", "alice-rpi4-01")
AGENT_NAME = os.getenv("AGENT_NAME", "Alice")
AGENT_TYPE = "shopping_guide"   # 与 server.py 里的注释一致：'greeter'/'concierge'/'product' ...

def _get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]; s.close()
        return ip
    except Exception:
        return "0.0.0.0"

# === 原有工具 ===
def load_used_sentences():
    if os.path.exists(USED_FILE):
        with open(USED_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_used_sentence(s):
    used = load_used_sentences()
    used.add(s)
    with open(USED_FILE, "w") as f:
        json.dump(list(used), f)

def gpt_reply(prompt, temp=0.9, max_tokens=80):
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are Alice, the clothing store housekeeper speaking in English."},
            {"role":"user","content":prompt}
        ],
        max_tokens=max_tokens,
        temperature=temp
    )
    return resp.choices[0].message["content"].strip()

def get_unique_sentence(prompt, temp=0.9, max_tokens=80):
    used = load_used_sentences()
    for _ in range(3):
        s = gpt_reply(prompt, temp, max_tokens)
        if s not in used:
            save_used_sentence(s)
            return s
    save_used_sentence(s)
    return s

def whisper_transcribe(pcm_bytes):
    duration = len(pcm_bytes) / 2 / SAMPLE_RATE
    if duration < MIN_AUDIO_LEN:
        print(f"⚠️ Skip too short audio <{MIN_AUDIO_LEN}s")
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio = AudioSegment(data=pcm_bytes, sample_width=2, frame_rate=SAMPLE_RATE, channels=1)
        audio.export(f.name, format="wav")
        try:
            with open(f.name, "rb") as audio_file:
                tr = openai.Audio.transcribe("whisper-1", audio_file)
                return tr["text"].strip()
        except Exception as e:
            print(f"❌ Whisper error: {e}")
            return ""

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    audio_queue.put(bytes(indata))

def vad_recording(timeout=7):
    print(f"🎤 Listening… pause ≤1s won’t cut your sentence. Timeout={timeout}s")
    start_time = time.time()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        pcm_buffer = b""
        speaking = False
        silence_time = 0.0
        while True:
            # ✅ timeout for initial voice
            if not speaking and (time.time() - start_time > timeout):
                print("⏳ No voice detected within timeout")
                conv_id = cc.load_conversation_id()
                cc.log_message(conv_id, AGENT_ID, "event", "no_response_timeout",
                               meta={"phase": "wait_user", "timeout_sec": timeout})
                return None

            try:
                frame = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            if is_speech:
                pcm_buffer += frame
                speaking = True
                silence_time = 0.0
            else:
                if speaking:
                    silence_time += FRAME_DURATION / 1000.0
                    if silence_time >= SILENCE_TIMEOUT:
                        duration = len(pcm_buffer) / 2 / SAMPLE_RATE
                        if duration < MIN_AUDIO_LEN:
                            print(f"⚠️ Too short({duration:.2f}s), discard & retry…")
                            pcm_buffer = b""
                            speaking = False
                            silence_time = 0.0
                            continue
                        print(f"✅ End of speech, length={duration:.2f}s")
                        return pcm_buffer

            duration = len(pcm_buffer) / 2 / SAMPLE_RATE
            if duration >= MAX_AUDIO_LEN:
                print(f"⏳ Max length {MAX_AUDIO_LEN}s reached, force end.")
                return pcm_buffer

chat_history = [{
    "role":"system",
    "content":(
        "Your name is Alice. You are the elegant, warm, and professional housekeeper of a clothing store. "
        "You introduce various clothes in English, give basic info, styles, and interact like a friendly consultant. "
        "NEVER repeat exactly, always vary phrasing."
    )
}]

def chat_with_gpt(user_input):
    chat_history.append({"role":"user","content":user_input})
    trimmed = [chat_history[0]] + chat_history[-10:]
    resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=trimmed, temperature=0.8)
    reply = resp.choices[0].message["content"]
    print(f"Alice replies: {reply}")
    chat_history.append({"role":"assistant","content":reply})
    save_used_sentence(reply)
    return reply

def is_leaving_intent(user_text):
    check_prompt = f"""The customer says: "{user_text}".
Does this clearly mean they want to leave or end the conversation? Answer ONLY YES or NO."""
    result = gpt_reply(check_prompt, temp=0, max_tokens=5)
    return "yes" in result.lower()

# ====== 启动：注册 & 心跳（新增，不影响原有流程）======
cc.register(
    AGENT_ID, AGENT_NAME, AGENT_TYPE, location="rpi4",
    meta={"capabilities":["tts","stt","dialog"], "ip": _get_ip(), "keywords": INTERRUPT_KEYWORDS}
)

def _hb_meta():
    return {"ip": _get_ip(), "sample_rate": SAMPLE_RATE}

cc.heartbeat_loop(AGENT_ID, interval=10, meta_fn=_hb_meta)

if __name__ == "__main__":
    print("🎯 Alice is ready. You can just talk, I’m listening…")

    # ====== 会话继承 / 兜底（新增）======
    conv_id = cc.load_conversation_id()
    if conv_id:
        cc.log_message(conv_id, AGENT_ID, "event", "agent_joined",
                       meta={"agent_name": AGENT_NAME, "from": "handover_or_resume"})
    else:
        # 若没有 Jarvis 预先创建的会话，则兜底创建一个，防止打点丢失
        conv_id = cc.start_conversation(AGENT_ID, meta={"reason": "alice_orphan_start"})
        # central_client 会把 conv_id 写入 /tmp/current_conversation_id.txt

    # === 原有开场白 ===
    opening_prompt = (
        "You are Alice, the warm and elegant housekeeper of a clothing store. "
        "Create a short friendly greeting in English. "
        "Say something like: 'Welcome to our clothing store, my name is Alice, "
        "I’m the store’s housekeeper. I can introduce the basic information about different clothes. "
        "Please tell me what kind of clothes you are looking for?' "
        "Just invite them to speak naturally."
    )
    opening_line = get_unique_sentence(opening_prompt)
    print(f"Alice opening: {opening_line}")
    # 打点：assistant 开场可被关键词打断
    cc.log_message(conv_id, AGENT_ID, "assistant", opening_line,
                   meta={"interruptible": True, "keywords": INTERRUPT_KEYWORDS})
    interrupted = speak_and_listen(opening_line, keywords=INTERRUPT_KEYWORDS)
    if interrupted:
        cc.log_message(conv_id, AGENT_ID, "event", "assistant_interrupted", meta={"by_keyword": True})
    chat_history.append({"role":"assistant","content":opening_line})

    try:
        while True:
            # ✅ 1st wait 7s（原逻辑）
            pcm_data = vad_recording(timeout=7)
            if pcm_data is None:
                # 1st reminder（原逻辑）
                remind_prompt = (
                    "The customer didn’t reply for 7 seconds. "
                    "Generate a short warm reminder like: "
                    "'Would you like me to tell you more about our styles or special collections? "
                    "If you’re interested, I can explain further.' "
                    "Keep it inviting and end with a soft question."
                )
                remind_line = get_unique_sentence(remind_prompt)
                print(f"Alice reminder: {remind_line}")
                cc.log_message(conv_id, AGENT_ID, "assistant", remind_line,
                               meta={"interruptible": True, "keywords": INTERRUPT_KEYWORDS})
                interrupted = speak_and_listen(remind_line, keywords=INTERRUPT_KEYWORDS)
                if interrupted:
                    cc.log_message(conv_id, AGENT_ID, "event", "assistant_interrupted", meta={"by_keyword": True})

                # 2nd wait 7s（原逻辑）
                pcm_data = vad_recording(timeout=7)
                if pcm_data is None:
                    goodbye_prompt = (
                        "The customer still didn’t reply after a reminder. "
                        "Generate a short polite goodbye in English like "
                        "'Alright, I’ll let you browse freely. Have a wonderful day!'"
                    )
                    goodbye_line = get_unique_sentence(goodbye_prompt)
                    print(f"Alice final goodbye: {goodbye_line}")
                    cc.log_message(conv_id, AGENT_ID, "assistant", goodbye_line,
                                   meta={"interruptible": True, "keywords": INTERRUPT_KEYWORDS})
                    speak_and_listen(goodbye_line, keywords=INTERRUPT_KEYWORDS)
                    cc.log_message(conv_id, AGENT_ID, "event", "agent_exit",
                                   meta={"reason":"no_reply_after_reminder"})
                    sys.exit(0)

            user_text = whisper_transcribe(pcm_data)
            if not user_text:
                print("🤔 Didn’t catch that, try again…")
                cc.log_message(conv_id, AGENT_ID, "event", "empty_transcript")
                continue

            print(f"🗣️ Customer: {user_text}")
            cc.log_message(conv_id, AGENT_ID, "user", user_text)

            if is_leaving_intent(user_text):
                cc.log_message(conv_id, AGENT_ID, "event", "intent_classified", meta={"intent": "leave"})
                goodbye_prompt = (
                    "You are Alice, the clothing store housekeeper. "
                    "The customer says they are leaving. "
                    "Generate a short polite goodbye in English, make it warm, natural, and do NOT repeat previous goodbye sentences."
                )
                goodbye_line = get_unique_sentence(goodbye_prompt)
                print(f"Alice goodbye: {goodbye_line}")
                cc.log_message(conv_id, AGENT_ID, "assistant", goodbye_line,
                               meta={"interruptible": True, "keywords": INTERRUPT_KEYWORDS})
                speak_and_listen(goodbye_line, keywords=INTERRUPT_KEYWORDS)
                cc.log_message(conv_id, AGENT_ID, "event", "agent_exit", meta={"reason":"user_left"})
                sys.exit(0)

            # === 正常对答（原逻辑）
            response = chat_with_gpt(user_text)
            cc.log_message(conv_id, AGENT_ID, "assistant", response,
                           meta={"interruptible": True, "keywords": INTERRUPT_KEYWORDS})
            interrupted = speak_and_listen(response, keywords=INTERRUPT_KEYWORDS)
            if interrupted:
                cc.log_message(conv_id, AGENT_ID, "event", "assistant_interrupted", meta={"by_keyword": True})
                print("🔄 interrupted by keyword → enter next round conversation")
                continue

    except KeyboardInterrupt:
        cc.log_message(conv_id, AGENT_ID, "event", "keyboard_interrupt")
        print("\nGoodbye!")
