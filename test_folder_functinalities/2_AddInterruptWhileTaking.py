#!/usr/bin/env python3
import os, sys, queue, asyncio, tempfile, subprocess, threading
import sounddevice as sd
import webrtcvad
from pydub import AudioSegment
from dotenv import load_dotenv
import openai
import edge_tts
import time
import threading

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
MAX_AUDIO_LEN = 20.0   # 最长一句20秒
SILENCE_TIMEOUT = 0.8  # 停顿0.8秒才算结束

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

# === 当前播放器进程句柄（用于打断TTS） ===
current_player = None
player_lock = threading.Lock()

# === 打断事件（让主线程知道被打断） ===
interrupt_event = threading.Event()

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

def stop_speaking():
    """停止当前TTS播放"""
    global current_player
    with player_lock:
        if current_player and current_player.poll() is None:
            print("🛑 停止TTS播放")
            current_player.terminate()
            current_player = None

def monitor_interrupt():
    """
    播放TTS时监听用户语音，必须满足：
      1. 播放0.5秒后才开始检测（避免回声）
      2. 必须连续1.3秒的语音帧 (~43帧) 才确认真人
      3. 第一次检测后仍需双次确认，防止误触
    """
    vad_for_interrupt = webrtcvad.Vad()
    vad_for_interrupt.set_mode(0)  # 最宽松，降低误判
    
    speech_frames = 0
    needed_frames = 43  # 43帧≈1.3秒连续语音才算打断
    start_delay = 0.5   # 前0.5s完全不监听，避免回声
    start_time = time.time()
    
    first_detected = False
    confirm_timeout = 1.0  # 第一次检测后1秒内必须继续说话才能确认
    
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        while True:
            # 播放结束就退出
            with player_lock:
                if current_player is None or current_player.poll() is not None:
                    break  
            
            # 播放开始的0.5秒直接丢弃，避免回声触发
            if time.time() - start_time < start_delay:
                try:
                    audio_queue.get(timeout=0.05)
                except queue.Empty:
                    pass
                continue
            
            # 获取音频帧
            try:
                frame = audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            
            # 检测帧是否为语音
            if vad_for_interrupt.is_speech(frame, SAMPLE_RATE):
                speech_frames += 1
            else:
                speech_frames = 0  # 中断后重新计数
            
            # 第一次检测到疑似语音 → 不立即打断，进入确认阶段
            if speech_frames > 5 and not first_detected:
                first_detected = True
                first_time = time.time()
                print("👂 检测到疑似语音，等待确认…")
            
            # 如果第一次检测后，用户没继续说话 → 取消检测
            if first_detected:
                if time.time() - first_time > confirm_timeout and speech_frames < 5:
                    first_detected = False
                    speech_frames = 0
                    print("❌ 检测取消：没有持续语音")
            
            # 最终确认：必须连续1.3秒的语音帧才打断
            if speech_frames >= needed_frames:
                print("✅ 确认真人持续说话 1.3秒 → 停止播放并进入录音")
                stop_speaking()
                interrupt_event.set()
                break

def speak_and_listen(text):
    """播放TTS，同时监听用户是否说话打断"""
    global current_player
    interrupt_event.clear()  # 播放前清理打断标志
    asyncio.run(speak_async(text))

    # 用 Popen 播放，可随时终止
    with player_lock:
        current_player = subprocess.Popen(
            ["mpg123", "-q", "reply.mp3"],  # -q 静音模式
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    # 开启监听线程
    t = threading.Thread(target=monitor_interrupt, daemon=True)
    t.start()

    # 等待播放器结束或被打断
    while True:
        if interrupt_event.is_set():
            print("🔄 语音被打断，提前退出播放")
            break
        with player_lock:
            if current_player is None or current_player.poll() is not None:
                break
        time.sleep(0.05)

    # 播放结束，清理状态
    with player_lock:
        if current_player:
            current_player.wait()
            current_player = None

    return interrupt_event.is_set()  # 返回是否被打断

# === 音频回调 ===
def audio_callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    audio_queue.put(bytes(indata))

# === 自动分句录音（停顿≤0.8s不结束） ===
def vad_recording():
    print("🎤 Listening… you can pause ≤0.8s without ending the sentence.")

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
                    if silence_time >= SILENCE_TIMEOUT:  # 停顿超过0.8秒
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

        # 新的播放函数，支持用户打断
        interrupted = speak_and_listen(reply)

        # 如果用户打断了，立刻进入录音，不等TTS播完
        if interrupted:
            print("🔄 被打断 → 立即进入下一轮录音")
            continue

if __name__ == "__main__":
    print("🎯 LLM Agent ready. Speak anytime, you can INTERRUPT the assistant and it will listen immediately!")
    run_conversation()
