#!/bin/bash
set -e

echo "🔄 Updating system packages..."
sudo apt update

echo "📦 Installing system dependencies..."
sudo apt install -y \
  python3-opencv python3-picamera2 python3-pip \
  libportaudio2 libportaudiocpp0 portaudio19-dev mpg123

echo "⬆️ Upgrading pip..."
pip3 install --upgrade pip --break-system-packages

echo "📝 Creating requirements.txt..."
cat > requirements.txt <<EOF
opencv-python
mediapipe
sounddevice
pydub
python-dotenv
openai==0.28
edge-tts
EOF

echo "📦 Installing Python dependencies (force system install)..."
pip3 install --break-system-packages -r requirements.txt

echo "✅ All dependencies installed successfully!"
