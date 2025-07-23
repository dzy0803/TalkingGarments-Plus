#!/usr/bin/env python3

import os, asyncio, edge_tts   # ✅ 新增 edge_tts 依赖
os.system("amixer sset 'Master' 75% unmute")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

async def startup_voice():
    tts = edge_tts.Communicate("System Activated, Please wait for a minute to boot the agents", voice="en-US-JennyNeural")
    await tts.save("/tmp/startup.mp3")
    os.system("mpg123 /tmp/startup.mp3")

# ✅ 运行时先说 Power on
asyncio.run(startup_voice())


import subprocess

while True:
    print("\n=== 🚀 Starting Jarvis (Jarvis.py) ===")
    ret = subprocess.run(["python3", "Jarvis.py"])
    print(f"Jarvis exited with code {ret.returncode}")

    # 如果 Jarvis 正常退出，进入 Alice
    print("\n=== 🤖 Switching to Alice (Alice.py) ===")
    ret = subprocess.run(["python3", "Alice.py"])
    print(f"Alice exited with code {ret.returncode}")

    # Alice 正常退出后 → 回到 Jarvis
    print("\n=== 🔄 Returning to Jarvis ===")
