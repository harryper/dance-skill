# dance-edit

抖音 / 视频号风格的舞蹈剪辑 skill。给一段舞蹈视频(横/竖屏均可,常见手持抖动),自动生成 9:16 竖屏成片——胸口锚点锁帧 + 蓝橙调色 + 节拍卡点。

## 用法

```
把舞蹈视频丢给 Claude,说"按模版剪"/"舞蹈卡点"/"dance edit"。
Claude 会:
  1. 跑 analyze.py 探测视频 + 追踪舞者 + 检测 BPM
  2. 把报告(BPM/推荐起止点/追踪覆盖率)给你看
  3. 你点头后跑 render.py 出 1080×1920 竖屏成片
```

直接用脚本:

```bash
# 1. 探针
python scripts/analyze.py input.mp4 /tmp/work

# 2. 看报告(看 report.json 里的 recommendation.start_time)
cat /tmp/work/report.json | python -m json.tool

# 3. 渲染(默认用源长,不切 BGM)
python scripts/render.py input.mp4 /tmp/work out.mp4 \
    --start 0 --zoom 1.4
# 要裁短就显式传 --duration
```

## 安装

```bash
# 系统依赖
# Debian/Ubuntu: sudo apt install libgl1 libegl1
# RHEL/CentOS:  sudo dnf install mesa-libGL mesa-libGLES

pip install -r requirements.txt

# 模型文件(models/pose_landmarker.task)已包含。
# 如果缺失,跑 setup.sh 重下:
./setup.sh
```

## 核心技法

| # | 技法 | 实现 |
|---|---|---|
| 1 | 锚点锁帧 | mediapipe PoseLandmarker 追踪胸口(landmark 11/12)+ 脚尖(27/28) |
| 2 | 背景反向 | 每帧 warp 把胸口贴到画面 (0.5, 0.36),背景反向滑动 |
| 3 | 节拍卡点 | librosa 测 BPM,源素材 ≥2 段时在节拍做 hard cut,**单段镜头跳过** |
| 4 | 蓝橙调色 | 阴影偏蓝 + 高光偏橙 + 对比 +15 + 饱和 +20 |

完整说明见 [`SKILL.md`](SKILL.md)。

## 输出规格

- 容器:MP4
- 视频:H.264, yuv420p, 30fps, **1080×1920**(9:16)
- 音频:AAC 128kbps(沿用源音频)

## 已知问题

- **mediapipe 0.10.x 需要 `libGLESv2.so.2` / `libEGL.so.1`**。系统装好 Mesa GL 后正常;无头服务器需要从 chromium 缓存复制(见 setup.sh)。
- 长视频(>2 分钟)用 `SAMPLE_FPS=10` 加速追踪。
- 9:16 竖屏源 + `zoom=1.0x` 没有"反向流动"效果,推荐 `zoom=1.4x+` 才有模版的招牌感。