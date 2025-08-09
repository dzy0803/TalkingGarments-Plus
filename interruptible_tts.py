# interruptible_tts.py
import os
import time
import queue
import asyncio
import subprocess
import threading
import sounddevice as sd
import webrtcvad
import tempfile
import wave
import openai  # Uses OPENAI_API_KEY from environment

# === Audio parameters ===
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

# === VAD for interruption detection ===
vad_interrupt = webrtcvad.Vad()
vad_interrupt.set_mode(0)  # Permissive to avoid missing speech

# === Shared state ===
audio_queue = queue.Queue()
current_player = None
player_lock = threading.Lock()
interrupt_event = threading.Event()


def audio_callback(indata, frames, time_info, status):
    """Callback for audio input, pushes audio data into the queue."""
    if status:
        print("⚠️", status)
    audio_queue.put(bytes(indata))


def stop_speaking():
    """Immediately stop the TTS playback process."""
    global current_player
    with player_lock:
        if current_player and current_player.poll() is None:
            print("🛑 Stopping TTS playback")
            current_player.terminate()
            current_player = None


def _transcribe_and_match(pcm_bytes, keywords, sample_rate=16000):
    """
    Transcribe a short PCM segment and check if any keyword appears.
    Returns True if matched, else False.
    """
    if not keywords or not pcm_bytes:
        return False

    # Write a temporary mono 16-bit WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)

        # Whisper transcription
        with open(tmp, "rb") as af:
            res = openai.Audio.transcribe("whisper-1", af)
            text = (res.get("text") or "").lower().strip()
            # Simple containment match (you may replace with regex or fuzzy)
            for kw in keywords:
                if kw.lower() in text:
                    print(f"🔎 Keyword hit: '{kw}' in '{text}'")
                    return True
        return False
    except Exception as e:
        print(f"❌ Keyword transcription error: {e}")
        return False
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def monitor_interrupt(check_person_func=None, disappear_limit=5, keywords=None):
    """
    Monitor microphone while TTS is playing.

    Modes:
    - If `keywords` provided: ONLY interrupt when transcribed segment contains one of them.
      We buffer voice segments (speech until a short silence), then transcribe on segment end.
    - If `keywords` not provided: fallback to legacy continuous-speech heuristic (≈1.3s).

    Additionally, if `check_person_func` is provided and returns False for > `disappear_limit`
    seconds, also interrupt.
    """
    # Legacy fallback threshold for continuous speech (if no keywords)
    needed_frames = int(1.3 / (FRAME_DURATION / 1000.0))
    speech_frames = 0

    # Avoid self-echo triggering at the very beginning
    start_delay = 0.5
    start_time = time.time()

    # Face disappear timer
    disappear_start = None

    # Segment buffering for keyword mode
    pcm_buffer = b""
    speaking = False
    silence_ms = 0.0
    silence_finish_threshold = 0.4  # 0.4s silence to mark segment end

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            # If player finished, stop monitoring
            with player_lock:
                if current_player is None or current_player.poll() is not None:
                    break

            # Skip first 0.5s to reduce echo-trigger
            if time.time() - start_time < start_delay:
                try:
                    audio_queue.get(timeout=0.05)
                except queue.Empty:
                    pass
                continue

            # Optional: stop if face disappears for too long
            if check_person_func:
                if not check_person_func():
                    if disappear_start is None:
                        disappear_start = time.time()
                    elif time.time() - disappear_start >= disappear_limit:
                        print(f"🚨 Face disappeared >{disappear_limit}s → stop & reset")
                        stop_speaking()
                        interrupt_event.set()
                        break
                else:
                    disappear_start = None

            try:
                frame = audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            is_speech = vad_interrupt.is_speech(frame, SAMPLE_RATE)

            # —— Keyword mode: buffer one utterance (speech → short silence), then transcribe & match ——
            if keywords:
                if is_speech:
                    pcm_buffer += frame
                    speaking = True
                    silence_ms = 0.0
                else:
                    if speaking:
                        silence_ms += FRAME_DURATION / 1000.0
                        if silence_ms >= silence_finish_threshold:
                            # A short utterance finished → transcribe and match
                            if _transcribe_and_match(pcm_buffer, keywords, SAMPLE_RATE):
                                print("✅ Keyword matched → stopping playback")
                                stop_speaking()
                                interrupt_event.set()
                                break
                            # Reset to continue listening for the next utterance
                            pcm_buffer = b""
                            speaking = False
                            silence_ms = 0.0
                # In keyword mode we do not use the legacy heuristic
                continue

            # —— Fallback mode (no keywords) ——
            if is_speech:
                speech_frames += 1
            else:
                speech_frames = 0

            if speech_frames >= needed_frames:
                print("✅ (fallback) continuous speech → stopping playback")
                stop_speaking()
                interrupt_event.set()
                break


def speak_and_listen(
    text,
    tts_voice="en-US-JennyNeural",
    check_person_func=None,
    disappear_limit=5,
    keywords=None,
):
    """
    Play TTS and allow interruption.

    Args:
        text: text to speak
        tts_voice: edge-tts voice name
        check_person_func: optional face-detection callback
        disappear_limit: seconds allowed without face (if provided)
        keywords: list[str] | None — if provided, ONLY these keywords interrupt playback

    Returns:
        True if playback was interrupted by user (keyword) or face disappear
        False if playback finished normally
    """
    global current_player
    interrupt_event.clear()

    # Generate TTS using edge-tts
    import edge_tts
    tmp_file = "/tmp/tts_reply.mp3"
    asyncio.run(edge_tts.Communicate(text, voice=tts_voice).save(tmp_file))

    # Start audio player (mpg123)
    with player_lock:
        current_player = subprocess.Popen(
            ["mpg123", "-q", tmp_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Start monitor thread
    t = threading.Thread(
        target=monitor_interrupt,
        kwargs={
            "check_person_func": check_person_func,
            "disappear_limit": disappear_limit,
            "keywords": keywords,
        },
        daemon=True,
    )
    t.start()

    # Wait until playback finishes or is interrupted
    while True:
        if interrupt_event.is_set():
            print("🔄 Playback interrupted (keyword/face)")
            break
        with player_lock:
            if current_player is None or current_player.poll() is not None:
                break
        time.sleep(0.05)

    # Cleanup after playback
    with player_lock:
        if current_player:
            current_player.wait()
            current_player = None

    return interrupt_event.is_set()

# import os
# import time
# import queue
# import asyncio
# import subprocess
# import threading
# import sounddevice as sd
# import webrtcvad


# # === Audio parameters ===
# SAMPLE_RATE = 16000
# FRAME_DURATION = 30  # ms
# FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

# # === VAD parameters for interruption detection ===
# vad_interrupt = webrtcvad.Vad()
# vad_interrupt.set_mode(0)  # More permissive to avoid missing speech

# # === Shared state ===
# audio_queue = queue.Queue()
# current_player = None
# player_lock = threading.Lock()
# interrupt_event = threading.Event()


# def audio_callback(indata, frames, time_info, status):
#     """Callback for audio input, pushes audio data into the queue."""
#     if status:
#         print("⚠️", status)
#     audio_queue.put(bytes(indata))


# def stop_speaking():
#     """Immediately stop the TTS playback process."""
#     global current_player
#     with player_lock:
#         if current_player and current_player.poll() is None:
#             print("🛑 Stopping TTS playback")
#             current_player.terminate()
#             current_player = None


# def monitor_interrupt(check_person_func=None, disappear_limit=5):
#     """
#     Monitor user speech to interrupt TTS (or stop if face disappears).
    
#     Rules:
#       - Start monitoring only after 0.5s (to avoid echo triggering)
#       - Require continuous speech for at least 1.3s to confirm interruption
#       - If check_person_func() returns False for > disappear_limit seconds, also interrupt
#     """
#     needed_frames = int(1.3 / (FRAME_DURATION / 1000.0))  # frames for ~1.0s speech
#     speech_frames = 0
#     first_detected = False
#     confirm_timeout = 1.0  # must continue talking within 1s after first detection
#     first_time = 0
#     start_delay = 0.5
#     start_time = time.time()
#     disappear_start = None

#     with sd.RawInputStream(
#         samplerate=SAMPLE_RATE,
#         blocksize=FRAME_SIZE,
#         dtype="int16",
#         channels=1,
#         callback=audio_callback,
#     ):
#         while True:
#             # If player finished, exit monitor
#             with player_lock:
#                 if current_player is None or current_player.poll() is not None:
#                     break

#             # Skip first 0.5s to avoid self echo triggering
#             if time.time() - start_time < start_delay:
#                 try:
#                     audio_queue.get(timeout=0.05)
#                 except queue.Empty:
#                     pass
#                 continue

#             # If a face detection callback is provided, check if face disappeared
#             if check_person_func:
#                 if not check_person_func():
#                     if disappear_start is None:
#                         disappear_start = time.time()
#                     elif time.time() - disappear_start >= disappear_limit:
#                         print(f"🚨 Face disappeared >{disappear_limit}s → stop & reset")
#                         stop_speaking()
#                         interrupt_event.set()
#                         break
#                 else:
#                     disappear_start = None

#             # Get one audio frame from queue
#             try:
#                 frame = audio_queue.get(timeout=0.05)
#             except queue.Empty:
#                 continue

#             # Detect if this frame is speech
#             if vad_interrupt.is_speech(frame, SAMPLE_RATE):
#                 speech_frames += 1
#             else:
#                 speech_frames = 0

#             # First detected some speech -> enter confirmation phase
#             if speech_frames > 5 and not first_detected:
#                 first_detected = True
#                 first_time = time.time()
#                 print("👂 Detected possible speech, waiting for confirmation…")

#             # If no continuous speech within confirmation window -> cancel detection
#             if first_detected:
#                 if time.time() - first_time > confirm_timeout and speech_frames < 5:
#                     first_detected = False
#                     speech_frames = 0
#                     print("❌ Canceled: no continuous speech")

#             # Final confirmation: must have continuous ~1.0s speech frames
#             if speech_frames >= needed_frames:
#                 print("✅ Confirmed continuous speech for 1.0s → stopping playback and entering recording")
#                 stop_speaking()
#                 interrupt_event.set()
#                 break


# def speak_and_listen(
#     text, tts_voice="en-US-JennyNeural", check_person_func=None, disappear_limit=5
# ):
#     """
#     Play TTS and allow user speech interruption.
    
#     Args:
#         text: text to speak
#         tts_voice: edge-tts voice name
#         check_person_func: optional face-detection callback (returns bool)
#         disappear_limit: how long face can disappear before stopping
    
#     Returns:
#         True if playback was interrupted by user speech or face disappear
#         False if playback finished normally
#     """
#     global current_player
#     interrupt_event.clear()

#     # Generate TTS using edge-tts
#     import edge_tts
#     tmp_file = "/tmp/tts_reply.mp3"
#     asyncio.run(edge_tts.Communicate(text, voice=tts_voice).save(tmp_file))

#     # Start audio player (mpg123)
#     with player_lock:
#         current_player = subprocess.Popen(
#             ["mpg123", "-q", tmp_file],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL,
#         )

#     # Start monitor thread to listen for interruption
#     t = threading.Thread(
#         target=monitor_interrupt,
#         kwargs={"check_person_func": check_person_func, "disappear_limit": disappear_limit},
#         daemon=True,
#     )
#     t.start()

#     # Wait until playback finishes or is interrupted
#     while True:
#         if interrupt_event.is_set():
#             print("🔄 Playback interrupted by user speech")
#             break
#         with player_lock:
#             if current_player is None or current_player.poll() is not None:
#                 break
#         time.sleep(0.05)

#     # Cleanup after playback
#     with player_lock:
#         if current_player:
#             current_player.wait()
#             current_player = None

#     return interrupt_event.is_set()
