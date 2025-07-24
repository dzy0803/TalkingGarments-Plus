import sys, os, json, tempfile, asyncio, queue
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
import openai
from pydub import AudioSegment
import time   # ✅ 补上这个
from interruptible_tts import speak_and_listen

# === CONFIG ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
MIN_AUDIO_LEN = 0.5
MAX_AUDIO_LEN = 20.0
SILENCE_TIMEOUT = 0.8

vad = webrtcvad.Vad()
vad.set_mode(2)

audio_queue = queue.Queue()
USED_FILE="used_sentences.json"

def load_used_sentences():
    if os.path.exists(USED_FILE):
        with open(USED_FILE,"r") as f:
            return set(json.load(f))
    return set()

def save_used_sentence(s):
    used=load_used_sentences()
    used.add(s)
    with open(USED_FILE,"w") as f: json.dump(list(used),f)

def gpt_reply(prompt,temp=0.9,max_tokens=80):
    resp=openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are Alice, the clothing store housekeeper speaking in English."},
            {"role":"user","content":prompt}
        ],
        max_tokens=max_tokens,
        temperature=temp
    )
    return resp.choices[0].message["content"].strip()

def get_unique_sentence(prompt,temp=0.9,max_tokens=80):
    used=load_used_sentences()
    for _ in range(3):
        s=gpt_reply(prompt,temp,max_tokens)
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
    with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f:
        audio=AudioSegment(data=pcm_bytes,sample_width=2,frame_rate=16000,channels=1)
        audio.export(f.name,format="wav")
        try:
            with open(f.name,"rb") as audio_file:
                tr=openai.Audio.transcribe("whisper-1",audio_file)
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
            # ✅ 超时检测
            if not speaking and (time.time() - start_time > timeout):
                print("⏳ No voice detected within timeout")
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

chat_history=[{
    "role":"system",
    "content":(
        "Your name is Alice. You are the elegant, warm, and professional housekeeper of a clothing store. "
        "You introduce various clothes in English, give basic info, styles, and interact like a friendly consultant. "
        "NEVER repeat exactly, always vary phrasing."
    )
}]

def chat_with_gpt(user_input):
    chat_history.append({"role":"user","content":user_input})
    trimmed=[chat_history[0]]+chat_history[-10:]
    resp=openai.ChatCompletion.create(model="gpt-4o-mini",messages=trimmed,temperature=0.8)
    reply=resp.choices[0].message["content"]
    print(f"Alice replies: {reply}")
    chat_history.append({"role":"assistant","content":reply})
    save_used_sentence(reply)
    return reply

def is_leaving_intent(user_text):
    check_prompt = f"""The customer says: "{user_text}".
Does this clearly mean they want to leave or end the conversation? Answer ONLY YES or NO."""
    result = gpt_reply(check_prompt, temp=0, max_tokens=5)
    return "yes" in result.lower()

if __name__=="__main__":
    print("🎯 Alice is ready. You can just talk, I’m listening…")
    opening_prompt=(
        "You are Alice, the warm and elegant housekeeper of a clothing store. "
        "Create a short friendly greeting in English. "
        "Say something like: 'Welcome to our clothing store, my name is Alice, "
        "I’m the store’s housekeeper. I can introduce the basic information about different clothes. "
        "Please tell me what kind of clothes you are looking for?' "
        "Just invite them to speak naturally."
    )
    opening_line=get_unique_sentence(opening_prompt)
    print(f"Alice opening: {opening_line}")
    speak_and_listen(opening_line)
    chat_history.append({"role":"assistant","content":opening_line})

    try:
        while True:
            # ✅ 第一次等 7 秒
            pcm_data = vad_recording(timeout=7)
            if pcm_data is None:
                # 第一次提醒 → 确认是否想了解更多
                remind_prompt = (
                    "The customer didn’t reply for 7 seconds. "
                    "Generate a short warm reminder like: "
                    "'Would you like me to tell you more about our styles or special collections? "
                    "If you’re interested, I can explain further.' "
                    "Keep it inviting and end with a soft question."
                )
                remind_line = get_unique_sentence(remind_prompt)
                print(f"Alice reminder: {remind_line}")
                speak_and_listen(remind_line)

                # 再等第二次 7 秒
                pcm_data = vad_recording(timeout=7)
                if pcm_data is None:
                    goodbye_prompt = (
                        "The customer still didn’t reply after a reminder. "
                        "Generate a short polite goodbye in English like "
                        "'Alright, I’ll let you browse freely. Have a wonderful day!'"
                    )
                    goodbye_line = get_unique_sentence(goodbye_prompt)
                    print(f"Alice final goodbye: {goodbye_line}")
                    speak_and_listen(goodbye_line)
                    sys.exit(0)

            user_text = whisper_transcribe(pcm_data)
            if not user_text:
                print("🤔 Didn’t catch that, try again…")
                continue

            print(f"🗣️ Customer: {user_text}")

            if is_leaving_intent(user_text):
                goodbye_prompt=(
                    "You are Alice, the clothing store housekeeper. "
                    "The customer says they are leaving. "
                    "Generate a short polite goodbye in English, make it warm, natural, and do NOT repeat previous goodbye sentences."
                )
                goodbye_line=get_unique_sentence(goodbye_prompt)
                print(f"Alice goodbye: {goodbye_line}")
                speak_and_listen(goodbye_line)
                sys.exit(0)

            response=chat_with_gpt(user_text)
            interrupted = speak_and_listen(response)
            if interrupted:
                print("🔄 interrupted → enter next round conversation")
                continue

    except KeyboardInterrupt:
        print("\nGoodbye!")
