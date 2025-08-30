# Jarvis.py — 保留原功能 + 接入 central_client（注册/心跳/对话日志/交接）
import cv2, mediapipe as mp, time, os, asyncio, edge_tts, subprocess, queue, webrtcvad
from dotenv import load_dotenv
from picamera2 import Picamera2
import openai, sys, json, tempfile, audioop, sounddevice as sd
from pydub import AudioSegment
import socket

# ←—— 接入你的 central_client.py ——→
# 默认指向你的PC服务地址；若已在系统里 export CENTRAL_URL，会覆盖这里的默认。
os.environ.setdefault("CENTRAL_URL", "http://192.168.1.121:8000")
import central_client as cc

from interruptible_tts import speak_and_listen  # now supports keywords

# === API ===（原有）
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

VOICE_JARVIS = "en-US-GuyNeural"
USED_FILE = "used_sentences_jarvis.json"

SAMPLE_RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
MIN_AUDIO_LEN = 0.5
MAX_AUDIO_LEN = 20.0
SILENCE_TIMEOUT = 0.8

vad = webrtcvad.Vad()
vad.set_mode(2)

audio_queue = queue.Queue()
device_info = sd.query_devices(kind='input')
DEVICE_SR = int(device_info['default_samplerate'])

# === 运行元信息（新增） ===
AGENT_ID = os.getenv("AGENT_ID", "jarvis-rpi4-01")
AGENT_NAME = os.getenv("AGENT_NAME", "Jarvis")
AGENT_TYPE = "greeter"

def _get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"

# === Whisper speech-to-text（原有） ===
def whisper_transcribe(pcm_bytes):
    duration = len(pcm_bytes) / 2 / SAMPLE_RATE
    if duration < MIN_AUDIO_LEN:
        print(f"⚠️ 跳过 <{MIN_AUDIO_LEN}s 的音频")
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio = AudioSegment(
            data=pcm_bytes,
            sample_width=2,
            frame_rate=SAMPLE_RATE,
            channels=1
        )
        audio.export(f.name, format="wav")
        try:
            with open(f.name, "rb") as audio_file:
                result = openai.Audio.transcribe("whisper-1", audio_file)
                return result["text"].strip()
        except Exception as e:
            print(f"❌ Whisper出错: {e}")
            return ""

# === 动态录音（原有，加入打点） ===
def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    audio_queue.put(bytes(indata))

def vad_recording(timeout=7):
    print(f"🎤 Listening… waiting for voice, timeout={timeout}s")
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
            # ✅ 超时检测
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
                            print(f"⚠️ 太短({duration:.2f}s)，丢弃重录…")
                            pcm_buffer = b""
                            speaking = False
                            silence_time = 0.0
                            continue
                        print(f"✅ End of speech, length={duration:.2f}s")
                        return pcm_buffer

            duration = len(pcm_buffer) / 2 / SAMPLE_RATE
            if duration >= MAX_AUDIO_LEN:
                print(f"⏳ 超过最大长度 {MAX_AUDIO_LEN}s，强制结束")
                return pcm_buffer

# === GPT helpers（原有） ===
def load_used_sentences():
    if os.path.exists(USED_FILE):
        with open(USED_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_used_sentence(sentence):
    used = load_used_sentences()
    used.add(sentence)
    with open(USED_FILE, "w") as f:
        json.dump(list(used), f)

def gpt_reply(prompt, temp=0.9, max_tokens=120):
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Jarvis, the store greeter assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temp
    )
    return resp.choices[0].message["content"].strip()

def get_unique_sentence(prompt, temp=0.9, max_tokens=120):
    used = load_used_sentences()
    for _ in range(3):
        s = gpt_reply(prompt, temp, max_tokens)
        if s not in used:
            save_used_sentence(s)
            return s
    save_used_sentence(s)
    return s

def classify_intent(user_text):
    prompt = f"""
    The user says: "{user_text}".
    Classify their intent:
    - If they clearly say yes, sure, okay, or want to know more → answer CONTINUE
    - If they clearly say no, leave, or goodbye → answer NONE
    Only output CONTINUE or NONE.
    """
    result = gpt_reply(prompt, temp=0, max_tokens=10).lower()
    return "continue" if "continue" in result else "none"

# === 说话（原有TTS保留）+ 打点封装 ===
def speak_no_interrupt(text):
    # 发送 assistant 消息到中央
    conv_id = cc.load_conversation_id()
    cc.log_message(conv_id, AGENT_ID, "assistant", text, meta={"interruptible": False})
    # 原有播报
    tmp_file = "/tmp/tts_once.mp3"
    asyncio.run(edge_tts.Communicate(text, voice=VOICE_JARVIS).save(tmp_file))
    subprocess.run(["mpg123", "-q", tmp_file])

def speak_with_optional_interrupt(text, keywords=None):
    """用于欢迎语等可打断发言"""
    conv_id = cc.load_conversation_id()
    cc.log_message(conv_id, AGENT_ID, "assistant", text,
                   meta={"interruptible": True, "keywords": keywords or []})
    interrupted = speak_and_listen(text, tts_voice=VOICE_JARVIS, keywords=(keywords or []))
    if interrupted:
        cc.log_message(conv_id, AGENT_ID, "event", "assistant_interrupted", meta={"by_keyword": True})
    return interrupted

# === Camera setup（原有） ===
picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"format": "RGB888", "size": (1640, 1232)}
    )
)
picam2.start()

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# === 启动即注册 & 心跳（新增，非侵入） ===
cc.register(AGENT_ID, AGENT_NAME, AGENT_TYPE, location="rpi4",
            meta={"capabilities": ["vision","tts","stt","handoff"],
                  "voice": VOICE_JARVIS, "sample_rate": SAMPLE_RATE, "ip": _get_ip()})

def _hb_meta():
    return {"device_sr": DEVICE_SR, "ip": _get_ip()}

cc.heartbeat_loop(AGENT_ID, interval=10, meta_fn=_hb_meta)

# === 启动提示（原有） ===
speak_no_interrupt("System ready, please look at the camera.")
time.sleep(0.2)  # 可选：给播放器一点点缓冲时间

stay_threshold = 2
interaction_mode = False
stay_start = None

def is_face_present(face_results):
    return bool(face_results.multi_face_landmarks)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        frame = picam2.capture_array()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        now = time.time()
        person_now = is_face_present(results)

        if not interaction_mode:
            if person_now:
                if stay_start is None:
                    stay_start = now
                else:
                    if now - stay_start >= stay_threshold:
                        interaction_mode = True
                        print("✅ Face stayed 3s → Entering welcome")

                        # —— 会话开始（新增打点）——
                        conv_id = cc.start_conversation(
                            AGENT_ID,
                            meta={"reason": "face_stay", "stay_threshold_sec": stay_threshold}
                        )
                        if not conv_id:
                            # 若失败，也不影响主流程
                            pass

                        # GPT welcome (allow keyword interruption)（原有）
                        welcome_prompt = (
                            "You are Jarvis, the warm store greeter. "
                            "Welcome the customer to the clothing store. "
                            "Briefly explain what makes the store special (custom designs, quality, styles). "
                            "Then say: 'If you’re interested, I can ask Alice, our specialist, to help you.' "
                            "Politely guide them to say YES or similar if they want more details."
                        )
                        welcome = get_unique_sentence(welcome_prompt)
                        print(f"🤖 Jarvis says: {welcome}")

                        # 🔑 Only keyword 'stop talking' will interrupt during TTS（原有）
                        interrupted = speak_with_optional_interrupt(
                            welcome,
                            keywords=["stop talking"]
                        )

                        if interrupted:
                            print("🔄 Keyword interruption detected → wait with confirm-Alice logic")
                            # ✅ First wait 7s
                            pcm_data = vad_recording(timeout=7)
                            if pcm_data is None:
                                # First reminder
                                remind_prompt = (
                                    "No one replied for 7 seconds. "
                                    "Generate a short warm confirmation like: "
                                    "'Would you like to know more about our collection? "
                                    "If you’re interested, I can ask Alice, our specialist, to assist you.' "
                                    "Keep it inviting and end with a soft question."
                                )
                                remind_line = gpt_reply(remind_prompt, temp=0.7, max_tokens=50)
                                print(f"🤖 Jarvis reminder: {remind_line}")
                                speak_no_interrupt(remind_line)

                                # ✅ Second wait 7s
                                pcm_data = vad_recording(timeout=7)
                                if pcm_data is None:
                                    # Still no response → goodbye & reset
                                    goodbye_prompt = (
                                        "No one replied even after a reminder. "
                                        "Say a short polite goodbye like 'Alright, I’ll let you browse freely. Have a great day!'"
                                    )
                                    goodbye_line = gpt_reply(goodbye_prompt, temp=0.7, max_tokens=40)
                                    print(f"🤖 Jarvis final goodbye: {goodbye_line}")
                                    speak_no_interrupt(goodbye_line)
                                    cc.log_message(cc.load_conversation_id(), AGENT_ID, "event",
                                                   "end_conversation", meta={"reason": "no_reply_after_reminder"})
                                    interaction_mode = False
                                    stay_start = None
                                    continue

                            # ✅ If someone spoke
                            user_reply = whisper_transcribe(pcm_data)
                            conv_id = cc.load_conversation_id()
                            cc.log_message(conv_id, AGENT_ID, "user", user_reply or "")
                            print(f"🗣️ User interrupted with: {user_reply}")
                            if not user_reply.strip():
                                unclear_prompt = (
                                    "The user interrupted but their reply was unclear. "
                                    "Politely remind them in ONE short sentence that Alice, the store specialist, "
                                    "is available to help with more details if they want."
                                )
                                unclear_line = gpt_reply(unclear_prompt, temp=0.7, max_tokens=40)
                                speak_no_interrupt(unclear_line)
                            else:
                                intent = classify_intent(user_reply)
                                cc.log_message(conv_id, AGENT_ID, "event", "intent_classified", meta={"intent": intent})
                                if intent == "continue":
                                    followup_prompt = f"""
                                    The user interrupted and said: "{user_reply}".
                                    Reply naturally as Jarvis:
                                    - Briefly acknowledge or respond.
                                    - Mention Alice, the store specialist.
                                    - End by politely asking if they would like Alice to help them.
                                    Keep it short, warm.
                                    """
                                    followup_line = gpt_reply(followup_prompt, temp=0.7, max_tokens=80)
                                    print(f"🤖 Jarvis follow-up: {followup_line}")
                                    speak_no_interrupt(followup_line)
                                else:
                                    goodbye_prompt = (
                                        "The user interrupted and seems to want to leave. "
                                        "Say a short polite goodbye, add a nice wish like 'have a great day'."
                                    )
                                    goodbye_line = gpt_reply(goodbye_prompt, temp=0.7, max_tokens=40)
                                    print(f"🤖 Jarvis goodbye: {goodbye_line}")
                                    speak_no_interrupt(goodbye_line)
                                    cc.log_message(conv_id, AGENT_ID, "event",
                                                   "end_conversation", meta={"reason": "user_left_after_interrupt"})
                                    interaction_mode = False
                                    stay_start = None
                                    continue

            else:
                stay_start = None

        if interaction_mode:
            # ✅ Subsequent dialog: same two-phase timeout（原有）
            pcm_data = vad_recording(timeout=7)
            if pcm_data is None:
                remind_prompt = (
                    "No one replied for 7 seconds. "
                    "Generate a short warm confirmation like: "
                    "'Would you like to know more about our collection? "
                    "If you’re interested, I can ask Alice, our specialist, to assist you.' "
                    "Keep it inviting and end with a soft question."
                )
                remind_line = gpt_reply(remind_prompt, temp=0.7, max_tokens=50)
                print(f"🤖 Jarvis reminder: {remind_line}")
                speak_no_interrupt(remind_line)

                pcm_data = vad_recording(timeout=7)
                if pcm_data is None:
                    goodbye_prompt = (
                        "No one replied even after a reminder. "
                        "Say a short polite goodbye like 'Alright, I’ll let you browse freely. Have a great day!'"
                    )
                    goodbye_line = gpt_reply(goodbye_prompt, temp=0.7, max_tokens=40)
                    print(f"🤖 Jarvis final goodbye: {goodbye_line}")
                    speak_no_interrupt(goodbye_line)
                    cc.log_message(cc.load_conversation_id(), AGENT_ID, "event",
                                   "end_conversation", meta={"reason": "no_reply_after_reminder"})
                    interaction_mode = False
                    stay_start = None
                    continue

            user_reply = whisper_transcribe(pcm_data)
            conv_id = cc.load_conversation_id()
            cc.log_message(conv_id, AGENT_ID, "user", user_reply or "")
            if not user_reply.strip():
                goodbye_prompt = (
                    "You are Jarvis the greeter. "
                    "Say a short polite goodbye, add a nice wish like 'have a great day'."
                )
                goodbye = get_unique_sentence(goodbye_prompt)
                print(f"🤖 Jarvis says: {goodbye}")
                speak_no_interrupt(goodbye)
                cc.log_message(conv_id, AGENT_ID, "event",
                               "end_conversation", meta={"reason": "empty_transcript"})
                interaction_mode = False
                stay_start = None
                continue

            print(f"🗣️ User: {user_reply}")
            intent = classify_intent(user_reply)
            cc.log_message(conv_id, AGENT_ID, "event", "intent_classified", meta={"intent": intent})

            if intent == "continue":
                speak_no_interrupt("Great! I’ll hand you over to Alice, our specialist.")
                # —— 交接到 Alice（新增打点）——
                cc.handover(conv_id, from_agent=AGENT_ID, to_agent="Alice", reason="handoff_to_alice")
                # 保持原逻辑：退出由 main 启动 Alice
                sys.exit(0)
            else:
                goodbye_prompt = (
                    "You are Jarvis the greeter. "
                    "The user didn’t want to continue. Say a short polite goodbye and a nice wish like 'enjoy your day'."
                )
                goodbye = get_unique_sentence(goodbye_prompt)
                print(f"🤖 Jarvis says: {goodbye}")
                speak_no_interrupt(goodbye)
                cc.log_message(conv_id, AGENT_ID, "event",
                               "end_conversation", meta={"reason": "user_declined"})
                interaction_mode = False
                stay_start = None
                continue

        # —— 可视化绘制（原有）——
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

        cv2.imshow("Jarvis FaceMesh Wide View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            conv_id = cc.load_conversation_id()
            cc.log_message(conv_id, AGENT_ID, "event", "manual_quit")
            break

cv2.destroyAllWindows()

