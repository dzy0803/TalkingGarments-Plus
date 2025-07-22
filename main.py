#!/usr/bin/env python3

import os, asyncio, edge_tts   # ✅ 新增 edge_tts 依赖

async def startup_voice():
    tts = edge_tts.Communicate("Power on", voice="en-US-JennyNeural")
    await tts.save("/tmp/startup.mp3")
    os.system("mpg123 /tmp/startup.mp3")

# ✅ 运行时先说 Power on
asyncio.run(startup_voice())


import subprocess
import os
os.system("amixer sset 'Master' 75% unmute")

while True:
    print("\n=== 🚀 Starting Jarvis (person_detect.py) ===")
    ret = subprocess.run(["python3", "person_detect.py"])
    print(f"Jarvis exited with code {ret.returncode}")

    # 如果 Jarvis 正常退出，进入 Alice
    print("\n=== 🤖 Switching to Alice (voice_chat.py) ===")
    ret = subprocess.run(["python3", "voice_chat.py"])
    print(f"Alice exited with code {ret.returncode}")

    # Alice 正常退出后 → 回到 Jarvis
    print("\n=== 🔄 Returning to Jarvis ===")
