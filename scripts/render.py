#!/usr/bin/env python3
"""
render.py — Dance edit skill: render final 9:16 vertical clip.

Reads:
    <out_dir>/anchors.json  — from analyze.py
    <out_dir>/report.json   — from analyze.py
    <out_dir>/audio.wav     — from analyze.py (optional)
    <source_video>          — original clip

Writes:
    <output_path>           — final MP4

Usage:
    render.py <source_video> <out_dir> <output_path> \
              [--start SEC] [--duration SEC] [--zoom Z] [--no-grade] [--no-subtitle]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DST_W, DST_H = 1080, 1920
DST_FPS = 30
TARGET_ANCHOR = (0.5, 0.36)  # chest lands here in output (centered, slightly above mid)


def find_font():
    """Locate a CJK-capable font for subtitle rendering."""
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def warp(frame, chest, zoom):
    """
    Crop+zoom around chest anchor. chest is normalized (x, y) in source.
    Returns DST_W x DST_H BGR.
    """
    h, w = frame.shape[:2]
    crop_w = w / zoom
    crop_h = h / zoom
    cx = chest[0] * w
    cy = chest[1] * h
    x0 = cx - crop_w / 2
    y0 = cy - crop_h / 2
    # Clamp
    x0 = max(0, min(w - crop_w, x0))
    y0 = max(0, min(h - crop_h, y0))
    x0, y0 = int(x0), int(y0)
    cw, ch = int(crop_w), int(crop_h)
    cropped = frame[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(cropped, (DST_W, DST_H), interpolation=cv2.INTER_LANCZOS4)


def grade_blue_orange(img):
    """
    Blue-orange split tone + contrast/saturation boost.
    Operates in-place on BGR uint8.
    """
    f = img.astype(np.float32) / 255.0
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    shadow = np.clip(1.0 - luma * 2.0, 0, 1)[..., None]
    highlight = np.clip((luma - 0.5) * 2.0, 0, 1)[..., None]
    # Shadows → blue/teal: B+, R-
    f[..., 0] += shadow[..., 0] * 0.10
    f[..., 2] -= shadow[..., 0] * 0.06
    # Highlights → orange: R+, B-
    f[..., 2] += highlight[..., 0] * 0.10
    f[..., 0] -= highlight[..., 0] * 0.06
    # Contrast
    f = (f - 0.5) * 1.15 + 0.5
    f = np.clip(f, 0, 1)
    # Saturation via HSV
    hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.20, 0, 255)
    f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def draw_subtitle(img, text, font_path):
    """Bottom-center subtitle with black outline, white fill."""
    if not text:
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font_size = 44
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (DST_W - tw) // 2
    y = int(DST_H * 0.86)
    # Outline
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def find_anchor_at_t(anchors, t):
    """Linear-interp chest (x,y) at time t from sampled anchor list."""
    if not anchors:
        return (0.5, 0.5)
    # binary search
    lo, hi = 0, len(anchors) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if anchors[mid]["t"] <= t:
            lo = mid
        else:
            hi = mid
    a, b = anchors[lo], anchors[hi]
    if b["t"] == a["t"]:
        return a["chest"][0], a["chest"][1]
    frac = (t - a["t"]) / (b["t"] - a["t"])
    cx = a["chest"][0] + (b["chest"][0] - a["chest"][0]) * frac
    cy = a["chest"][1] + (b["chest"][1] - a["chest"][1]) * frac
    return cx, cy


def render(source, out_dir, output, start, duration, zoom, do_grade, subtitle):
    anchors = json.loads((Path(out_dir) / "anchors.json").read_text())["anchors"]
    if not anchors:
        raise SystemExit("no anchors in cache — run analyze.py first")

    cap = cv2.VideoCapture(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start * src_fps)
    end_frame = int((start + duration) * src_fps)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame, total)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    font_path = find_font() if subtitle else None

    tmp_video = str(Path(out_dir) / "video_only.mp4")
    ffmpeg = subprocess.Popen([
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{DST_W}x{DST_H}", "-r", str(DST_FPS),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        tmp_video,
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    n_frames = 0
    t = start
    print(f"[render] {source} | {start:.2f}s +{duration:.2f}s | zoom={zoom}", file=sys.stderr)
    while True:
        if n_frames >= int(duration * DST_FPS):
            break
        ok, frame = cap.read()
        if not ok:
            break
        cx, cy = find_anchor_at_t(anchors, t)
        out = warp(frame, (cx, cy), zoom)
        if do_grade:
            out = grade_blue_orange(out)
        if subtitle:
            out = draw_subtitle(out, subtitle, font_path)
        ffmpeg.stdin.write(out.tobytes())
        n_frames += 1
        t += 1.0 / DST_FPS
    cap.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"[render] wrote {n_frames} frames to {tmp_video}", file=sys.stderr)

    # Mux audio if available
    audio = Path(out_dir) / "audio.wav"
    has_audio = audio.exists()
    cmd = ["ffmpeg", "-y", "-i", tmp_video]
    if has_audio:
        cmd += ["-ss", str(start), "-i", str(audio), "-t", str(duration)]
    cmd += ["-c:v", "copy"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    cmd += ["-movflags", "+faststart", output]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    Path(tmp_video).unlink(missing_ok=True)
    print(f"[render] final → {output}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("out_dir")
    ap.add_argument("output")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=18.0)
    ap.add_argument("--zoom", type=float, default=1.4)
    ap.add_argument("--no-grade", action="store_true")
    ap.add_argument("--no-subtitle", action="store_true")
    ap.add_argument("--subtitle", type=str, default="")
    args = ap.parse_args()
    render(
        source=args.source,
        out_dir=args.out_dir,
        output=args.output,
        start=args.start,
        duration=args.duration,
        zoom=args.zoom,
        do_grade=not args.no_grade,
        subtitle="" if args.no_subtitle else args.subtitle,
    )


if __name__ == "__main__":
    main()
