import sys, os, re, queue, audioop, sounddevice as sd
import json, tempfile, asyncio, subprocess, webrtcvad
from dotenv import load_dotenv
import openai, edge_tts
from pydub import AudioSegment

# === CONFIG ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

device_info = sd.query_devices(kind='input')
DEVICE_SR = int(device_info['default_samplerate'])

USED_FILE="used_sentences.json"

# === VAD params ===
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
MIN_AUDIO_LEN = 0.5
MAX_AUDIO_LEN = 20.0
SILENCE_TIMEOUT = 0.8

vad = webrtcvad.Vad()
vad.set_mode(2)

audio_queue = queue.Queue()

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

# === VAD dynamic recording ===
def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    audio_queue.put(bytes(indata))

def vad_recording():
    print("🎤 Listening… pause ≤1s won’t cut your sentence.")
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

async def speak_async(text):
    tts=edge_tts.Communicate(text,voice="en-US-JennyNeural")
    await tts.save("response.mp3")
    os.system("mpg123 response.mp3")

def speak(text): asyncio.run(speak_async(text))

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
    check_prompt=(
        f"The customer says: \"{user_text}\".\n"
        "Does this clearly mean they want to leave or end the conversation? Answer ONLY YES or NO."
    )
    result=gpt_reply(check_prompt,temp=0,max_tokens=5)
    return "yes" in result.lower()

if __name__=="__main__":
    print("🎯 Alice is ready. You can just talk, I’m listening…")
    # Opening
    opening_prompt=(
        "You are Alice, the warm and elegant housekeeper of a clothing store. "
        "Create a short friendly greeting in English. "
        "Say something like: 'Welcome to our clothing store, my name is Alice, "
        "I’m the store’s housekeeper. I can introduce the basic information about different clothes "
        "and how you can interact with me. Please tell me what kind of clothes you are looking for?' "
        "Do NOT mention pressing space, just invite them to speak naturally."
    )
    opening_line=get_unique_sentence(opening_prompt)
    print(f"Alice opening: {opening_line}")
    speak(opening_line)
    chat_history.append({"role":"assistant","content":opening_line})

    try:
        while True:
            pcm_data = vad_recording()
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
                speak(goodbye_line)
                sys.exit(0)

            response=chat_with_gpt(user_text)
            speak(response)

    except KeyboardInterrupt:
        print("\nGoodbye!")



