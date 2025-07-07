#!/usr/bin/env python3
from dotenv import load_dotenv
import sys
import os
import re
import queue
import json
import audioop
import tty
import termios
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import openai
import asyncio
import edge_tts

# ——— CONFIGURATION ———
load_dotenv()  # 👈 这句自动从 .env 文件读取变量
openai.api_key = os.getenv("OPENAI_API_KEY")
# Detect microphone’s native sample rate
device_info = sd.query_devices(kind='input')
DEVICE_SR = int(device_info['default_samplerate'])

# Load Vosk model
model = Model("models/vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

# Contextual memory (chat history)
chat_history = [
    {
        "role": "system",
        "content": (
            "You are a smart, emotionally expressive, and friendly male voice assistant "
            "with a warm, youthful tone. You speak like a real person, not a robot. "
            "Express emotions naturally — curiosity, joy, empathy, even gentle humor. "
            "Keep answers concise but meaningful. Don't be afraid to add a little personality."
        )
    }
]


def wait_for_space():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == ' ':
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def record_until_toggle():
    chunks = []
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status and status.input_overflow:
            return
        q.put(bytes(indata))

    print("Press [space] → start recording")
    wait_for_space()
    print("Recording… press [space] → stop")

    with sd.RawInputStream(
        samplerate=DEVICE_SR,
        blocksize=8000,
        dtype='int16',
        channels=1,
        callback=callback
    ):
        wait_for_space()

    print("Recording stopped, resampling…")
    while not q.empty():
        chunks.append(q.get())
    raw = b''.join(chunks)
    pcm16 = audioop.ratecv(raw, 2, 1, DEVICE_SR, 16000, None)[0]
    return pcm16


def listen():
    pcm = record_until_toggle()
    if recognizer.AcceptWaveform(pcm):
        text = json.loads(recognizer.Result()).get("text", "")
    else:
        text = json.loads(recognizer.FinalResult()).get("text", "")
    print(f"Recognized: {text}")
    return text.strip()


async def speak_async(text):
    tts = edge_tts.Communicate(text, voice="en-US-JennyNeural")
    await tts.save("response.mp3")
    os.system("mpg123 response.mp3")

def speak(text):
    asyncio.run(speak_async(text))


def chat_with_gpt(user_input):
    print("Asking GPT…")

    # Add user input to history
    chat_history.append({"role": "user", "content": user_input})

    # Limit memory to last 5 rounds (10 messages + system)
    MAX_HISTORY = 10
    trimmed_history = [chat_history[0]] + chat_history[-MAX_HISTORY:]

    # GPT call
    resp = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=trimmed_history,
        temperature=0.6
    )

    reply = resp.choices[0].message["content"]
    print(f"GPT replies: {reply}")

    # Add GPT response to history
    chat_history.append({"role": "assistant", "content": reply})
    return reply


def handle_volume_command(text):
    m = re.search(r'volume\s*(?:to)?\s*(\d{1,3})\s*%?', text, re.IGNORECASE)
    if m:
        level = max(0, min(int(m.group(1)), 150))
        os.system(f"amixer sset 'Master' {level}% unmute")
        speak(f"Volume set to {level} percent.")
        return True

    if re.search(r'(increase volume|turn it up|louder)', text, re.IGNORECASE):
        os.system("amixer sset 'Master' 10%+ unmute")
        speak("Volume increased.")
        return True

    if re.search(r'(decrease volume|turn it down|quieter)', text, re.IGNORECASE):
        os.system("amixer sset 'Master' 10%- unmute")
        speak("Volume decreased.")
        return True

    return False


if __name__ == "__main__":
    print("Press [space] → record/stop → process. Run as normal user (no sudo). Ctrl+C to exit.")
    try:
        while True:
            user_text = listen()
            if not user_text:
                continue

            if handle_volume_command(user_text):
                continue

            response = chat_with_gpt(user_text)
            speak(response)

    except KeyboardInterrupt:
        print("\nGoodbye!")
