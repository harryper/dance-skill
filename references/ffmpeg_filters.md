# ffmpeg filter cheatsheet (for dance-edit skill)

## zoompan — anchor-locked crop

```
zoompan=z='1.4':
        x='chest_x*iw - iw/2/z':
        y='chest_y*ih - ih/2/z':
        d=1:s=1080x1920:fps=30
```

Gotchas:
- `z` is the zoom factor (1.0 = no zoom)
- `x`/`y` is the **top-left** of the crop window in the input
- For 30fps output, source must be 30fps (or use `-r 30` first)
- Total frames must equal `DST_FPS × duration`

## colorbalance — split tone

```
colorbalance=bs=0.15:bm=0.10:hs=0:hm=0.15:rm=0.05
```

- `bs` / `bm` = shadow blue / shadow magenta (positive = more blue in shadows)
- `hm` / `rm` = highlight magenta / red (positive = warmer highlights)
- values 0–1, typical 0.05–0.20 for subtle, 0.30+ for stylized

## curves — exact tone curve

```
curves=preset=darker
curves=r='0/0.1 0.5/0.5 1/0.95':g='0/0.1 0.5/0.5 1/0.95':b='0/0.05 0.5/0.5 1/1.0'
```

Three-point curve: black point / mid / white point per channel. Blue-orange = crush blue blacks, lift red highlights.

## eq — contrast + saturation

```
eq=contrast=1.15:saturation=1.20:brightness=0.02
```

- `contrast` 1.0 = neutral, 1.15 ≈ punchy
- `saturation` 1.0 = neutral, 1.20 ≈ vibrant
- avoid > 1.5, looks crunchy

## drawtext — bottom subtitle

```
drawtext=text='HELLO':
        fontfile=/usr/share/fonts/.../wqy-microhei.ttc:
        fontsize=44:
        x=(w-tw)/2:
        y=h*0.86:
        fontcolor=white:
        borderw=2:bordercolor=black:
        enable='between(t,0,3)'
```

- `enable='between(t,a,b)'` locks subtitle to a time window, prevents flicker
- For Chinese, MUST use CJK font (wqy-microhei or Noto Sans CJK)

## crop + scale (no zoompan, for static shot)

```
crop=ih*9/16:ih:    # 9:16 crop from middle
scale=1080:1920
```

## muxing audio from source

```
ffmpeg -i video_only.mp4 -ss <start> -i source.mp4 -t <dur> \
       -c:v copy -c:a aac -b:a 128k -shortest out.mp4
```

`-shortest` ends when the shorter stream ends. Use `-map 0:v -map 1:a` to be explicit.

## rawvideo pipe from Python

Python writes BGR24 to stdin:
```python
ffmpeg = subprocess.Popen([
    "ffmpeg", "-y",
    "-f", "rawvideo", "-pix_fmt", "bgr24",
    "-s", "1080x1920", "-r", "30",
    "-i", "pipe:0",
    "-c:v", "libx264", "-preset", "medium", "-crf", "20",
    "-pix_fmt", "yuv420p", "out.mp4",
], stdin=subprocess.PIPE)
for frame in processed:
    ffmpeg.stdin.write(frame.tobytes())
ffmpeg.stdin.close()
ffmpeg.wait()
```

## Common 9:16 resolutions

| 竖屏 | 用途 |
|---|---|
| 540×960 | 抖音/快手标清 |
| 720×1280 | 抖音高清 |
| 1080×1920 | 全屏高清(默认) |
| 1440×2560 | 2K(不必要) |
