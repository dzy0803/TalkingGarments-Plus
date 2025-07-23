#!/usr/bin/env python3
import os, sys, queue, asyncio, tempfile, audioop
import sounddevice as sd
import webrtcvad
from pydub import AudioSegment
from dotenv import load_dotenv
import openai
import edge_tts

# === 读取 API KEY ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === 音频参数 ===
SAMPLE_RATE = 16000       # Whisper需要16k
FRAME_DURATION = 30       # 每帧30ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

# === VAD配置 ===
vad = webrtcvad.Vad()
vad.set_mode(2)  # 0-3，越大越敏感

# === 参数（可调） ===
MIN_AUDIO_LEN = 0.5    # 至少0.5秒才算有效
MAX_AUDIO_LEN = 20.0   # 最长一句15秒
SILENCE_TIMEOUT = 0.8  # 停顿1秒才算结束

# === 队列缓存 ===
audio_queue = queue.Queue()

# === 对话上下文 ===
chat_history = [
    {
        "role": "system",
        "content": (
            "You are a friendly, natural, conversational AI assistant. "
            "Speak concisely, like a real person, and sound warm and engaging."
        )
    }
]

# === Whisper 语音转文本 ===
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
        except openai.error.InvalidRequestError:
            print("⚠️ Whisper拒绝，音频太短/无效")
            return ""
        except Exception as e:
            print(f"❌ Whisper出错: {e}")
            return ""

# === GPT-4o-mini 对话 ===
def chat_with_gpt(user_text):
    chat_history.append({"role": "user", "content": user_text})
    trimmed = [chat_history[0]] + chat_history[-10:]
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=trimmed,
        temperature=0.6
    )
    reply = resp.choices[0].message["content"]
    chat_history.append({"role": "assistant", "content": reply})
    return reply

# === Edge-TTS 朗读 ===
async def speak_async(text):
    tts = edge_tts.Communicate(text, voice="en-US-JennyNeural")
    await tts.save("reply.mp3")
    os.system("mpg123 reply.mp3 > /dev/null 2>&1")

def speak(text):
    asyncio.run(speak_async(text))

# === 音频回调 ===
def audio_callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    audio_queue.put(bytes(indata))

# === 自动分句录音（停顿≤1秒不结束） ===
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
                    if silence_time >= SILENCE_TIMEOUT:  # 停顿超过1秒
                        duration = len(pcm_buffer) / 2 / SAMPLE_RATE
                        if duration < MIN_AUDIO_LEN:
                            print(f"⚠️ 太短({duration:.2f}s)，丢弃重录…")
                            pcm_buffer = b""
                            speaking = False
                            silence_time = 0.0
                            continue

                        print(f"✅ End of speech, length={duration:.2f}s")
                        return pcm_buffer

            # 防止一直说 > MAX_AUDIO_LEN 秒
            duration = len(pcm_buffer) / 2 / SAMPLE_RATE
            if duration >= MAX_AUDIO_LEN:
                print(f"⏳ 超过最大长度 {MAX_AUDIO_LEN}s，强制结束")
                return pcm_buffer

# === 主循环：轮流对话 ===
def run_conversation():
    while True:
        pcm_data = vad_recording()
        user_text = whisper_transcribe(pcm_data)
        if not user_text.strip():
            print("🤔 没听清，再试一次…")
            continue

        print(f"🗣️ User: {user_text}")
        reply = chat_with_gpt(user_text)
        print(f"🤖 Assistant: {reply}")
        speak(reply)

if __name__ == "__main__":
    print("🎯 LLM Agent ready. Start speaking, pause ≤1s won’t cut your sentence.")
    run_conversation()
