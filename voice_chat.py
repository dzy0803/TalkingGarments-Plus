import sys, os, re, queue, audioop, tty, termios, sounddevice as sd
import json, tempfile, asyncio, subprocess
from dotenv import load_dotenv
import openai, edge_tts
from pydub import AudioSegment

# === CONFIG ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
device_info = sd.query_devices(kind='input')
DEVICE_SR = int(device_info['default_samplerate'])

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
    with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f:
        audio=AudioSegment(data=pcm_bytes,sample_width=2,frame_rate=16000,channels=1)
        audio.export(f.name,format="wav")
        audio_file=open(f.name,"rb")
        tr=openai.Audio.transcribe("whisper-1",audio_file)
        return tr["text"]

def wait_for_space():
    fd=sys.stdin.fileno()
    old_settings=termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch=sys.stdin.read(1)
            if ch==' ':
                break
    finally:
        termios.tcsetattr(fd,termios.TCSADRAIN,old_settings)

def record_until_toggle():
    chunks,q=[],queue.Queue()
    def cb(indata,frames,time,status):
        if status and status.input_overflow: return
        q.put(bytes(indata))
    print("Press [space] → start recording")
    wait_for_space()
    print("Recording… press [space] → stop")
    with sd.RawInputStream(samplerate=DEVICE_SR,blocksize=8000,dtype='int16',channels=1,callback=cb):
        wait_for_space()
    print("Recording stopped, resampling…")
    while not q.empty(): chunks.append(q.get())
    raw=b''.join(chunks)
    pcm16=audioop.ratecv(raw,2,1,DEVICE_SR,16000,None)[0]
    return pcm16

def listen():
    pcm=record_until_toggle()
    text=whisper_transcribe(pcm)
    print(f"Recognized: {text}")
    return text.strip()

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
    print("Press [space] → record/stop → process. Ctrl+C to exit.")
    # Opening
    opening_prompt=(
        "You are Alice, the warm and elegant housekeeper of a clothing store. "
        "Create a short friendly greeting in English. "
        "Say something like: 'Welcome to our clothing store, my name is Alice, "
        "I’m the store’s housekeeper. I can introduce the basic information about different clothes "
        "and how you can interact with me. But first, please tell me what kind of clothes you are looking for?' "
        "At the end, also politely tell them: 'Press space to start recording, and press it again to finish recording so we can talk.' "
        "Make it natural, polite, slightly varied, and do NOT repeat any previous greeting exactly."
    )
    opening_line=get_unique_sentence(opening_prompt)
    print(f"Alice opening: {opening_line}")
    speak(opening_line)
    chat_history.append({"role":"assistant","content":opening_line})

    try:
        while True:
            user_text=listen()
            if not user_text: continue

            if is_leaving_intent(user_text):
                goodbye_prompt=(
                    "You are Alice, the clothing store housekeeper. "
                    "The customer says they are leaving. "
                    "Generate a short polite goodbye in English, make it warm, natural, and do NOT repeat previous goodbye sentences."
                )
                goodbye_line=get_unique_sentence(goodbye_prompt)
                print(f"Alice goodbye: {goodbye_line}")
                speak(goodbye_line)
                sys.exit(0)  # ✅ 回 main.py

            response=chat_with_gpt(user_text)
            speak(response)

    except KeyboardInterrupt:
        print("\nGoodbye!")



