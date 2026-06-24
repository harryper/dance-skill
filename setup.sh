#!/bin/bash
# setup.sh — install deps and fetch mediapipe pose landmarker model.
# Run once after cloning.
set -e

cd "$(dirname "$0")"

echo "[1/3] pip install -r requirements.txt"
pip install -r requirements.txt

echo "[2/3] fetch pose_landmarker.task (5.5MB)"
if [ ! -f models/pose_landmarker.task ]; then
    mkdir -p models
    curl -sL -o models/pose_landmarker.task \
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
fi

echo "[3/3] check libGLESv2/libEGL availability"
python3 -c "
import ctypes
for lib in ('libGLESv2.so.2', 'libEGL.so.1'):
    try:
        ctypes.CDLL(lib)
        print(f'  {lib}: OK')
    except OSError:
        print(f'  {lib}: MISSING — install mesa-libGL mesa-libGLES (RHEL) or libgl1 libegl1 (Debian)')
"

echo "done. smoke test:"
echo "  python scripts/analyze.py path/to/dance.mp4 /tmp/work"