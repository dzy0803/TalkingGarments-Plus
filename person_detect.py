import cv2, mediapipe as mp, time, os, asyncio, edge_tts, subprocess
from dotenv import load_dotenv
from picamera2 import Picamera2
import openai, sys, json, tempfile, audioop, sounddevice as sd
from pydub import AudioSegment

# === API ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

VOICE_JARVIS = "en-US-GuyNeural"
USED_FILE = "used_sentences_jarvis.json"

# === sounddevice config ===
device_info = sd.query_devices(kind='input')
DEVICE_SR = int(device_info['default_samplerate'])

# === Whisper speech-to-text ===
def whisper_transcribe(pcm_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f:
        audio=AudioSegment(data=pcm_bytes,sample_width=2,frame_rate=16000,channels=1)
        audio.export(f.name,format="wav")
        audio_file=open(f.name,"rb")
        tr=openai.Audio.transcribe("whisper-1",audio_file)
        return tr["text"]

def record_short_audio(duration=2):  # ✅ 改成2秒
    print(f"🎤 Listening {duration}s...")
    audio = sd.rec(int(duration * DEVICE_SR), samplerate=DEVICE_SR, channels=1, dtype='int16')
    sd.wait()
    pcm16 = audioop.ratecv(audio.tobytes(), 2, 1, DEVICE_SR, 16000, None)[0]
    return pcm16

def listen_once_auto():
    pcm = record_short_audio(2)
    text = whisper_transcribe(pcm)
    print(f"Recognized: {text}")
    return text.strip()

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
    # 生成语音
    asyncio.run(edge_tts.Communicate(text,voice=VOICE_JARVIS).save(tmp_file))

    proc = subprocess.Popen(["mpg123", "-q", tmp_file])
    disappear_start = None
    while proc.poll() is None:  # 播放中
        time.sleep(0.2)
        if not check_person_func():  # 人脸不在
            if disappear_start is None:
                disappear_start = time.time()
            elif time.time() - disappear_start >= disappear_limit:
                print(f"🚨 Face gone >{disappear_limit}s during speaking → stop & reset")
                proc.terminate()
                return False  # 中途离开 → reset
        else:
            disappear_start = None  # 人回来了，重置计时
    return True  # 正常播放完毕

# === Camera setup ===
picam2=Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"format":"RGB888","size":(1640,1232)}  # Zoom out最大视野
    )
)
picam2.start()

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

stay_threshold=3         # 连续3秒进入交互

interaction_mode=False
stay_start=None

# === 实时检测人脸函数 ===
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

        # === 连续检测人脸3秒才进入交互 ===
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
                            # 欢迎语过程中人走了≥5秒 → reset
                            interaction_mode=False
                            stay_start=None
                            continue
            else:
                stay_start=None

        # === 交互状态 ===
        if interaction_mode:
            # 监听2秒
            user_reply = listen_once_auto()
            if not user_reply:
                # 沉默 → 简短道别并reset
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

            # 有语音 → 只接受肯定继续
            intent = classify_intent(user_reply)
            print(f"Intent classified: {intent}")

            if intent == "continue":
                speak_interruptible("Great! I’ll hand you over to Alice, our specialist.",
                                    lambda: is_face_present(results), disappear_limit=5)
                sys.exit(0)  # ✅ 切换到 Alice

            else:
                # 不是肯定答复 → 简短道别+祝愿后reset
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

        # === 绘制FaceMesh关键点 ===
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
