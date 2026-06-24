#!/usr/bin/env python3
"""
analyze.py — Dance edit skill: probe + mediapipe anchor tracking + BPM detect.

Usage:
    analyze.py <video_path> [out_dir]

Outputs:
    <out_dir>/report.json    — full analysis (printed to stdout too)
    <out_dir>/anchors.json   — per-frame {chest, foot} keypoints (interpolated)
    <out_dir>/audio.wav      — extracted audio for re-use by render.py

Side effects:
    Loads entire video into mediapipe at source fps. For 60s+ clips, set
    SAMPLE_FPS=15 via env var to halve cost.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import librosa
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SAMPLE_FPS = int(os.environ.get("SAMPLE_FPS", "0"))  # 0 = native
MISSING_CONF = 0.4
MODEL_PATH = os.environ.get(
    "POSE_MODEL",
    str(Path(__file__).parent.parent / "models" / "pose_landmarker.task"),
)


def probe(path: str) -> dict:
    """ffprobe → dict with width/height/fps/duration/has_audio."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    out = subprocess.check_output(cmd).decode()
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    fps_s = v.get("r_frame_rate", "30/1")
    num, den = fps_s.split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return {
        "path": path,
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": fps,
        "duration": float(data["format"]["duration"]),
        "codec": v["codec_name"],
        "has_audio": has_audio,
    }


def extract_audio(video: str, out_wav: str) -> bool:
    if not Path(out_wav).exists():
        subprocess.check_call([
            "ffmpeg", "-y", "-i", video, "-vn", "-ac", "1",
            "-ar", "22050", "-f", "wav", out_wav,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def detect_bpm(wav_path: str) -> dict:
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    if len(y) < sr:  # < 1s of audio
        return {"bpm": None, "confidence": 0.0}
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
    return {
        "bpm": round(bpm, 1),
        "beat_count": len(beat_times),
        "beat_times": [round(t, 3) for t in beat_times[:64]],
    }


def track_anchors(video: str, fps: float) -> list:
    """
    mediapipe Pose per frame → list of dicts:
        {frame, t, chest: (x, y, conf), foot: (x, y, conf)}
    x, y in 0..1 normalized coords. Missing → all None → linear-interp.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / fps)) if SAMPLE_FPS == 0 else max(1, round(src_fps / SAMPLE_FPS))

    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    raw = []  # (frame_idx, t, chest_xy_conf or None, foot_xy_conf or None)
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = pose.detect(mp_image)
            entry = {"frame": f, "t": round(f / src_fps, 4), "chest": None, "foot": None}
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]  # first/only person
                # chest: mid of left/right shoulder (11, 12)
                ls, rs = lm[11], lm[12]
                if ls.visibility > MISSING_CONF and rs.visibility > MISSING_CONF:
                    entry["chest"] = [
                        round((ls.x + rs.x) / 2, 5),
                        round((ls.y + rs.y) / 2, 5),
                        round((ls.visibility + rs.visibility) / 2, 3),
                    ]
                # foot: mid of left/right ankle (27, 28)
                la, ra = lm[27], lm[28]
                if la.visibility > MISSING_CONF and ra.visibility > MISSING_CONF:
                    entry["foot"] = [
                        round((la.x + ra.x) / 2, 5),
                        round((la.y + ra.y) / 2, 5),
                        round((la.visibility + ra.visibility) / 2, 3),
                    ]
            raw.append(entry)
        f += 1
    cap.release()
    pose.close()

    # Linear-interpolate missing chest/foot across time.
    def interp(entries, key):
        vals = [e[key] for e in entries]
        idxs = [i for i, v in enumerate(vals) if v is not None]
        if not idxs:
            return
        for i, e in enumerate(entries):
            if e[key] is not None:
                continue
            # find neighbors
            left = max([j for j in idxs if j <= i], default=None)
            right = min([j for j in idxs if j >= i], default=None)
            if left is None:
                e[key] = vals[right]
            elif right is None or left == right:
                e[key] = vals[left]
            else:
                a, b = vals[left], vals[right]
                t = (i - left) / (right - left)
                e[key] = [
                    round(a[0] + (b[0] - a[0]) * t, 5),
                    round(a[1] + (b[1] - a[1]) * t, 5),
                    round(a[2] + (b[2] - a[2]) * t, 3),
                ]
    interp(raw, "chest")
    interp(raw, "foot")
    return raw


def coverage_stats(anchors: list) -> dict:
    if not anchors:
        return {"chest": 0.0, "foot": 0.0}
    chest = sum(1 for a in anchors if a["chest"] is not None) / len(anchors)
    foot = sum(1 for a in anchors if a["foot"] is not None) / len(anchors)
    return {"chest": round(chest, 3), "foot": round(foot, 3)}


def recommend_window(anchors: list, fps: float, bpm: dict, target_dur: float = None, source_duration: float = None) -> dict:
    """
    Default: recommend the FULL source from 0 — don't truncate, BGM must stay
    intact. `target_dur` is reserved for future use; pass None for default.
    Score is computed on the full source as a quality indicator.
    """
    if not anchors:
        return {"start_time": 0.0, "duration": 0.0, "score": 0.0, "note": "no anchors"}
    if not bpm.get("bpm"):
        bpm_note = "no bpm"
    else:
        bpm_note = None

    n = len(anchors)
    # Prefer the actual source duration over anchor-count-derived, since the
    # latter has rounding loss at non-integer ratios.
    if source_duration is not None:
        full_dur = round(source_duration, 3)
    else:
        full_dur = round(n / fps, 3)

    # Quality score on full source — informational, not used to crop.
    chest_xs = [w["chest"][0] for w in anchors if w["chest"]]
    chest_ys = [w["chest"][1] for w in anchors if w["chest"]]
    foot_count = sum(1 for w in anchors if w["foot"] is not None)
    if chest_xs:
        score = (
            -abs(np.mean(chest_xs) - 0.5) * 2
            - max(0, abs(np.mean(chest_ys) - 0.4) - 0.1) * 2
            + foot_count / n
        )
        score = float(round(score, 4))
    else:
        score = 0.0

    out = {
        "start_time": 0.0,
        "duration": full_dur,
        "score": score,
    }
    if bpm_note:
        out["note"] = bpm_note
    else:
        out["bpm_used"] = bpm["bpm"]
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: analyze.py <video_path> [out_dir]", file=sys.stderr)
        sys.exit(1)
    video = sys.argv[1]
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/dance_work")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "audio.wav"

    meta = probe(video)
    print(f"[probe] {meta['width']}x{meta['height']} @ {meta['fps']:.2f}fps, {meta['duration']:.2f}s",
          file=sys.stderr)

    if meta["has_audio"]:
        extract_audio(video, str(wav))
        bpm_info = detect_bpm(str(wav))
        print(f"[bpm] {bpm_info['bpm']} ({bpm_info['beat_count']} beats)", file=sys.stderr)
    else:
        bpm_info = {"bpm": None, "beat_count": 0, "beat_times": []}

    # Track at native fps by default; mediapipe cost is ~30ms/frame on CPU.
    anchors = track_anchors(video, meta["fps"])
    print(f"[track] {len(anchors)} anchor samples", file=sys.stderr)

    cov = coverage_stats(anchors)
    # recommend uses *anchor sampling* fps, not source fps
    anchor_fps = (SAMPLE_FPS if SAMPLE_FPS > 0 else int(round(meta["fps"])))
    rec = recommend_window(anchors, anchor_fps, bpm_info, source_duration=meta["duration"])

    report = {
        "meta": meta,
        "bpm": bpm_info,
        "anchor_coverage": cov,
        "anchor_samples": len(anchors),
        "recommendation": rec,
    }

    (out_dir / "anchors.json").write_text(json.dumps({
        "fps": meta["fps"],
        "anchors": anchors,
    }))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
