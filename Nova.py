# file: test_agent_pi3.py
import os, sys, time, tempfile, queue, asyncio, subprocess, json
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
import openai
from pydub import AudioSegment
import edge_tts

# ===================== CONFIG =====================
load_dotenv()  # load OPENAI_API_KEY from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 16000               # 16k mono for VAD + Whisper
FRAME_DURATION = 30               # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
MIN_AUDIO_LEN = 0.5               # seconds; too-short chunks are discarded
MAX_AUDIO_LEN = 20.0              # seconds; safety cap
SILENCE_TIMEOUT = 1.0             # seconds of silence to end an utterance
FIRST_WAIT_TIMEOUT = 7            # seconds; initial wait for user speech

VAD_MODE = 2                      # 0-3, higher = more aggressive
VOICE = "en-US-AriaNeural"        # Edge TTS voice
GPT_MODEL = "gpt-4o-mini"

# ===================== GLOBALS =====================
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

audio_q = queue.Queue()

# ===================== AUDIO I/O =====================
def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status, flush=True)
    audio_q.put(bytes(indata))

def record_until_silence(timeout=FIRST_WAIT_TIMEOUT):
    """
    Open mic, wait for voice up to `timeout` seconds, then collect frames
    until we detect ~SILENCE_TIMEOUT of non-speech. Returns PCM bytes or None.
    """
    print(f"🎤 Listening… (timeout={timeout}s, pause≤{SILENCE_TIMEOUT}s won’t cut you off)")
    start = time.time()
    pcm = b""
    speaking = False
    silence_acc = 0.0

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            # initial "no speech" timeout
            if not speaking and (time.time() - start > timeout):
                print("⏳ No voice detected within timeout")
                return None

            try:
                frame = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            if is_speech:
                pcm += frame
                speaking = True
                silence_acc = 0.0
            else:
                if speaking:
                    silence_acc += FRAME_DURATION / 1000.0
                    if silence_acc >= SILENCE_TIMEOUT:
                        dur = len(pcm) / 2 / SAMPLE_RATE
                        if dur < MIN_AUDIO_LEN:
                            # too short, reset and keep waiting
                            print(f"⚠️ Too short ({dur:.2f}s), discard & retry…")
                            pcm = b""
                            speaking = False
                            silence_acc = 0.0
                            continue
                        print(f"✅ End of speech, length={dur:.2f}s")
                        return pcm

            # safety cap
            if len(pcm) / 2 / SAMPLE_RATE >= MAX_AUDIO_LEN:
                print(f"⏹️ Reached max length {MAX_AUDIO_LEN}s, stopping.")
                return pcm

# ===================== WHISPER STT =====================
def transcribe_whisper(pcm_bytes: bytes) -> str:
    """
    Save PCM16 mono 16k to WAV and call Whisper.
    """
    if not pcm_bytes:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio = AudioSegment(
            data=pcm_bytes, sample_width=2, frame_rate=SAMPLE_RATE, channels=1
        )
        audio.export(f.name, format="wav")
        try:
            with open(f.name, "rb") as af:
                # using same API style as your existing code
                out = openai.Audio.transcribe("whisper-1", af)
            text = out.get("text", "").strip()
            return text
        except Exception as e:
            print(f"❌ Whisper error: {e}")
            return ""

# ===================== GPT =====================
def chat_reply(history, user_text, temperature=0.8, max_tokens=150):
    """
    Simple chat turn with a short system persona. Returns assistant string.
    """
    msgs = [
        {"role": "system", "content": (
            "You are a friendly in-store assistant named Nova. "
            "Be concise, helpful, and natural. If the user says 'bye' or 'goodbye', "
            "reply briefly and end the session."
        )}
    ]
    # keep last 8 turns for brevity
    msgs += history[-8:]
    msgs.append({"role": "user", "content": user_text})

    resp = openai.ChatCompletion.create(
        model=GPT_MODEL, messages=msgs, temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message["content"].strip()

# ===================== TTS (Edge) =====================
def tts_play(text: str):
    """
    Synthesize to /tmp/tts.mp3 via edge-tts, then play with mpg123 (or aplay fallback).
    """
    tmp = "/tmp/tts_pi3.mp3"
    asyncio.run(edge_tts.Communicate(text, voice=VOICE).save(tmp))
    # Try mpg123, fallback to aplay
    ret = subprocess.call(["mpg123", "-q", tmp])
    if ret != 0:
        subprocess.call(["aplay", tmp])

# ===================== MAIN LOOP =====================
def main():
    print("✅ Test Agent for Pi 3 is ready.")
    greeting = "Hello! I’m Nova. I can hear you. What would you like to know?"
    print(f"Nova: {greeting}")
    tts_play(greeting)

    chat_history = []  # store [{"role":"user"/"assistant","content": "..."}]

    try:
        while True:
            pcm = record_until_silence(timeout=FIRST_WAIT_TIMEOUT)
            if pcm is None:
                # one gentle reminder, then quit if still no response
                reminder = "Would you like to ask me anything? I’m happy to help."
                print(f"Nova: {reminder}")
                tts_play(reminder)

                pcm = record_until_silence(timeout=FIRST_WAIT_TIMEOUT)
                if pcm is None:
                    bye = "Alright, I’ll be here if you need me. Bye!"
                    print(f"Nova: {bye}")
                    tts_play(bye)
                    break

            user_text = transcribe_whisper(pcm)
            if not user_text:
                print("🤔 Didn’t catch that—let’s try again.")
                tts_play("I didn’t catch that—could you say that again?")
                continue

            print(f"👤 User: {user_text}")

            # simple exit intents
            if user_text.lower().strip() in {"bye", "goodbye", "exit", "quit"}:
                bye = "Goodbye! Have a lovely day."
                print(f"Nova: {bye}")
                tts_play(bye)
                break

            reply = chat_reply(chat_history, user_text)
            print(f"Nova: {reply}")
            chat_history.append({"role": "user", "content": user_text})
            chat_history.append({"role": "assistant", "content": reply})
            tts_play(reply)

    except KeyboardInterrupt:
        print("\n👋 Stopped by user.")

if __name__ == "__main__":
    # Optionally set output volume on boot (ALSA):
    # os.system("amixer sset 'Master' 75% unmute")
    main()
