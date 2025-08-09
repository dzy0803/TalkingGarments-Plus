# file: Nova.py
import os, time, tempfile, queue, asyncio, subprocess
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
import openai, edge_tts
from pydub import AudioSegment

# ==== CONFIG ====
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
MIN_AUDIO_LEN = 0.5
MAX_AUDIO_LEN = 20.0
SILENCE_TIMEOUT = 1.0        # 停顿 1 秒视为说完
FIRST_WAIT_TIMEOUT = 7       # 首次等待用户说话 7 秒
VAD_MODE = 2
GPT_MODEL = "gpt-4o-mini"
VOICE = "en-US-AriaNeural"

vad = webrtcvad.Vad(); vad.set_mode(VAD_MODE)
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status: print("⚠️", status, flush=True)
    audio_q.put(bytes(indata))

def record_until_silence(timeout=FIRST_WAIT_TIMEOUT):
    print(f"🎤 Listening… (timeout={timeout}s, pause≤{SILENCE_TIMEOUT}s won’t cut you off)")
    start = time.time()
    pcm, speaking, silence_acc = b"", False, 0.0
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE, dtype="int16",
                           channels=1, callback=audio_callback):
        while True:
            if not speaking and (time.time() - start > timeout):
                print("⏳ No voice detected within timeout"); return None
            try: frame = audio_q.get(timeout=0.2)
            except queue.Empty: continue
            if vad.is_speech(frame, SAMPLE_RATE):
                pcm += frame; speaking = True; silence_acc = 0.0
            else:
                if speaking:
                    silence_acc += FRAME_DURATION/1000.0
                    if silence_acc >= SILENCE_TIMEOUT:
                        dur = len(pcm)/2/SAMPLE_RATE
                        if dur < MIN_AUDIO_LEN:
                            print(f"⚠️ Too short ({dur:.2f}s), discard & retry…")
                            pcm, speaking, silence_acc = b"", False, 0.0
                            continue
                        print(f"✅ End of speech, length={dur:.2f}s"); return pcm
            if len(pcm)/2/SAMPLE_RATE >= MAX_AUDIO_LEN:
                print(f"⏹️ Reached max length {MAX_AUDIO_LEN}s, stopping."); return pcm

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

def chat_reply(history, user_text, temperature=0.8, max_tokens=150):
    msgs = [{"role":"system","content":"You are Nova, a concise, friendly in-store assistant. Keep replies short and helpful."}]
    msgs += history[-8:] + [{"role":"user","content":user_text}]
    resp = openai.ChatCompletion.create(model=GPT_MODEL, messages=msgs, temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message["content"].strip()

def tts_play(text: str):
    tmp = "/tmp/nova_tts.mp3"
    asyncio.run(edge_tts.Communicate(text, voice=VOICE).save(tmp))
    ret = subprocess.call(["mpg123","-q",tmp])
    if ret != 0: subprocess.call(["aplay", tmp])

def main():
    print("✅ Nova ready on Pi 3.")
    hello = "Hello! I’m Nova. I can hear you. What would you like to know?"
    print(f"Nova: {hello}"); tts_play(hello)
    history=[]
    try:
        while True:
            pcm = record_until_silence(timeout=FIRST_WAIT_TIMEOUT)
            if pcm is None:
                nudge = "Would you like to ask me anything? I’m happy to help."
                print(f"Nova: {nudge}"); tts_play(nudge)
                pcm = record_until_silence(timeout=FIRST_WAIT_TIMEOUT)
                if pcm is None:
                    bye="Alright, I’ll be here if you need me. Bye!"
                    print(f"Nova: {bye}"); tts_play(bye); break
            txt = transcribe_whisper(pcm)
            if not txt:
                tts_play("I didn’t catch that—could you say that again?"); continue
            print(f"👤 User: {txt}")
            if txt.lower().strip() in {"bye","goodbye","exit","quit"}:
                tts_play("Goodbye! Have a lovely day."); break
            ans = chat_reply(history, txt)
            print(f"Nova: {ans}")
            history += [{"role":"user","content":txt},{"role":"assistant","content":ans}]
            tts_play(ans)
    except KeyboardInterrupt:
        print("\n👋 Stopped by user.")

if __name__ == "__main__":
    # 可选：开机调音量
    # os.system("amixer sset 'Master' 75% unmute")
    main()
