import cv2, mediapipe as mp, time, os, asyncio, edge_tts, subprocess, queue, webrtcvad
from dotenv import load_dotenv
from picamera2 import Picamera2
import openai, sys, json, tempfile, audioop, sounddevice as sd
from pydub import AudioSegment

# === API ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

VOICE_JARVIS = "en-US-GuyNeural"
USED_FILE = "used_sentences_jarvis.json"

# === 音频参数 ===
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
MIN_AUDIO_LEN = 0.5
MAX_AUDIO_LEN = 20.0
SILENCE_TIMEOUT = 0.8  # 停顿超过0.8秒才算结束

# === VAD配置 ===
vad = webrtcvad.Vad()
vad.set_mode(2)  # 0-3 越大越敏感

audio_queue = queue.Queue()

device_info = sd.query_devices(kind='input')
DEVICE_SR = int(device_info['default_samplerate'])

# === Whisper speech-to-text ===
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

# === 动态录音（和测试版一致） ===
def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    audio_queue.put(bytes(indata))

def vad_recording():
    print("🎤 Listening… you can pause ≤1s without ending the sentence.")

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
            frame = audio_queue.get()
            if len(frame) < FRAME_SIZE * 2:
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

# === GPT helpers ===
def load_used_sentences():
    if os.path.exists(USED_FILE):
        with open(USED_FILE,"r") as f:
            return set(json.load(f))
    return set()

def save_used_sentence(sentence):
    used=load_used_sentences()
    used.add(sentence)
    with open(USED_FILE,"w") as f:
        json.dump(list(used),f)

def gpt_reply(prompt,temp=0.9,max_tokens=120):
    resp=openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are Jarvis, the store greeter assistant."},
            {"role":"user","content":prompt}
        ],
        max_tokens=max_tokens,
        temperature=temp
    )
    return resp.choices[0].message["content"].strip()

def get_unique_sentence(prompt,temp=0.9,max_tokens=120):
    used=load_used_sentences()
    for _ in range(3):
        s=gpt_reply(prompt,temp,max_tokens)
        if s not in used:
            save_used_sentence(s)
            return s
    save_used_sentence(s)
    return s

# === Intent classification ===
def classify_intent(user_text):
    prompt = f"""
    The user says: "{user_text}".
    Classify their intent:
    - If they clearly say yes, sure, okay, or want to know more → answer CONTINUE
    - Otherwise → answer NONE
    Only output CONTINUE or NONE.
    """
    result = gpt_reply(prompt, temp=0, max_tokens=10).lower()
    return "continue" if "continue" in result else "none"

# === Interruptible speak ===
def speak_interruptible(text, check_person_func, disappear_limit=5):
    tmp_file="/tmp/tts.mp3"
    asyncio.run(edge_tts.Communicate(text,voice=VOICE_JARVIS).save(tmp_file))
    proc = subprocess.Popen(["mpg123", "-q", tmp_file])
    disappear_start = None
    while proc.poll() is None:
        time.sleep(0.2)
        if not check_person_func():
            if disappear_start is None:
                disappear_start = time.time()
            elif time.time() - disappear_start >= disappear_limit:
                print(f"🚨 Face gone >{disappear_limit}s during speaking → stop & reset")
                proc.terminate()
                return False
        else:
            disappear_start = None
    return True

# === Camera setup ===
picam2=Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"format":"RGB888","size":(1640,1232)}
    )
)
picam2.start()

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

stay_threshold=3
interaction_mode=False
stay_start=None

def is_face_present(face_results):
    return bool(face_results.multi_face_landmarks)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        frame=picam2.capture_array()
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=face_mesh.process(rgb_frame)

        now=time.time()
        person_now=is_face_present(results)

        if not interaction_mode:
            if person_now:
                if stay_start is None:
                    stay_start=now
                else:
                    if now - stay_start >= stay_threshold:
                        interaction_mode=True
                        print("✅ Face stayed 3s → Entering welcome")
                        welcome_prompt = (
                            "You are Jarvis, the warm store greeter. "
                            "Welcome the customer to the clothing store. "
                            "Briefly explain what makes the store special (custom designs, quality, styles). "
                            "Then say: 'If you’re interested, I can ask Alice, our specialist, to help you.' "
                            "Politely guide them to say YES or similar if they want more details."
                        )
                        welcome = get_unique_sentence(welcome_prompt)
                        print(f"🤖 Jarvis says: {welcome}")
                        ok = speak_interruptible(welcome, lambda: is_face_present(results), disappear_limit=5)
                        if not ok:
                            interaction_mode=False
                            stay_start=None
                            continue
            else:
                stay_start=None

        if interaction_mode:
            # ✅ 改为动态VAD录音，而不是固定2秒
            pcm_data = vad_recording()
            user_reply = whisper_transcribe(pcm_data)
            if not user_reply.strip():
                goodbye_prompt = (
                    "You are Jarvis the greeter. "
                    "Say a short polite goodbye, add a nice wish like 'have a great day'."
                )
                goodbye = get_unique_sentence(goodbye_prompt)
                print(f"🤖 Jarvis says: {goodbye}")
                speak_interruptible(goodbye, lambda: is_face_present(results), disappear_limit=5)
                interaction_mode=False
                stay_start=None
                continue

            print(f"🗣️ User: {user_reply}")
            intent = classify_intent(user_reply)
            print(f"Intent classified: {intent}")

            if intent == "continue":
                speak_interruptible("Great! I’ll hand you over to Alice, our specialist.",
                                    lambda: is_face_present(results), disappear_limit=5)
                sys.exit(0)
            else:
                goodbye_prompt = (
                    "You are Jarvis the greeter. "
                    "The user didn’t want to continue. Say a short polite goodbye and a nice wish like 'enjoy your day'."
                )
                goodbye = get_unique_sentence(goodbye_prompt)
                print(f"🤖 Jarvis says: {goodbye}")
                speak_interruptible(goodbye, lambda: is_face_present(results), disappear_limit=5)
                interaction_mode=False
                stay_start=None
                continue

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
                )

        cv2.imshow("Jarvis FaceMesh Wide View",frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

cv2.destroyAllWindows()

