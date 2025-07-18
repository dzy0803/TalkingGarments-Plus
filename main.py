#!/usr/bin/env python3
import subprocess

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
