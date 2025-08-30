#!/usr/bin/env python3

import os, asyncio, edge_tts   # ✅ 新增 edge_tts 依赖
from pathlib import Path
CONV_FILE = Path("/tmp/current_conversation_id.txt")
os.system("amixer sset 'Master' 70% unmute")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

async def startup_voice():
    tts = edge_tts.Communicate("System Activated, Please wait for a minute to boot the agents", voice="en-US-AriaNeural")
    await tts.save("/tmp/startup.mp3")
    os.system("mpg123 /tmp/startup.mp3")

# ✅ 运行时先说 Power on
asyncio.run(startup_voice())


import subprocess

while True:
    print("\n=== 🚀 Starting Jarvis (Jarvis.py) ===")
    ret = subprocess.run(["python3", "Jarvis.py"])
    print(f"Jarvis exited with code {ret.returncode}")

    # 如果 Jarvis 正常退出，进入 Alice，并把会话ID传过去
    env = os.environ.copy()
    if CONV_FILE.exists():
        env["TALOS_CONV_ID"] = CONV_FILE.read_text().strip()

    print("\n=== 🤖 Switching to Alice (Alice.py) ===")
    ret = subprocess.run(["python3", "Alice.py"], env=env)
    print(f"Alice exited with code {ret.returncode}")

    print("\n=== 🔄 Returning to Jarvis ===")

