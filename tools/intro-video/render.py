#!/usr/bin/env python3
"""Render the README intro video and its silent looping preview from tools/intro-video/scenes.json.

Every moving picture comes from the rollout clips in docs/showcase/; titles, the deploy diagram, the bar
chart and the captions are drawn here with Pillow. Narration is Amazon Polly (neural voice, a few cents
per render; cached under .temp/intro-video-cache/), and its word timings drive the scene lengths, the
diagram cues and the burned-in captions. Frames are piped as raw RGB into ffmpeg (H.264 + AAC for the
MP4, animated WebP for the preview).

    python tools/intro-video/render.py                  # intro.mp4 + intro-preview.webp
    python tools/intro-video/render.py --no-voice       # silent, timings estimated (no AWS call)
    python tools/intro-video/render.py --stills DIR --at 3,20,41    # PNG frames only, for review

Requires numpy, Pillow and ffmpeg (on PATH, $FFMPEG, or the imageio-ffmpeg wheel); boto3 for narration;
matplotlib is only used to locate its bundled DejaVu fonts (any TTF works via --font / --font-bold).
See README.md next to this file.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SHOWCASE = ROOT / "docs" / "showcase"
CACHE = ROOT / ".temp" / "intro-video-cache"
SR = 16000  # Polly neural PCM sample rate

BG = (14, 19, 27)
PANEL = (26, 33, 45)
PANEL_LIT = (38, 48, 64)
TEXT = (240, 243, 247)
MUTED = (150, 160, 176)
ACCENT = (255, 153, 0)  # AWS orange
OK = (64, 196, 120)
FAIL = (232, 84, 72)
BLUE = (96, 165, 250)

CONTENT_H = 628  # captions live below this line
LEAD_S, TAIL_S = 0.45, 0.75  # silence before / after the narration of each scene
FADE_S = 0.3


# ---------------------------------------------------------------- tools


def ffmpeg_exe() -> str:
    exe = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found: install it, set $FFMPEG, or pip install imageio-ffmpeg")


FONT_PATHS: dict[str, str | None] = {"regular": None, "bold": None, "mono": None}


def find_fonts(regular: str | None, bold: str | None, mono: str | None) -> None:
    found = {"regular": regular, "bold": bold, "mono": mono}
    try:
        from matplotlib import font_manager

        names = {"regular": "DejaVu Sans", "bold": "DejaVu Sans:bold", "mono": "DejaVu Sans Mono"}
        for k, name in names.items():
            if not found[k]:
                found[k] = font_manager.findfont(name, fallback_to_default=False)
    except Exception:  # noqa: BLE001 - fonts are optional, Pillow's default font is the fallback
        pass
    FONT_PATHS.update(found)


@functools.lru_cache(maxsize=None)
def font(size: int, kind: str = "regular") -> ImageFont.FreeTypeFont:
    path = FONT_PATHS.get(kind) or FONT_PATHS.get("regular")
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def text_w(f: ImageFont.FreeTypeFont, s: str) -> int:
    b = f.getbbox(s)
    return b[2] - b[0]


# ---------------------------------------------------------------- clips


@functools.lru_cache(maxsize=None)
def probe(path: Path) -> tuple[int, int]:
    err = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Video: .*?(\d{2,5})x(\d{2,5})", err)
    if not m:
        sys.exit(f"cannot read the video size of {path}")
    return int(m.group(1)), int(m.group(2))


@dataclass
class Clip:
    frames: np.ndarray  # (n, h, w, 3) uint8

    def at(self, i: int) -> np.ndarray:
        return self.frames[i % len(self.frames)]


@functools.lru_cache(maxsize=None)
def load_clip(src: str, box_w: int, box_h: int, n: int, fps: int, speed: float = 1.0, start: float = 0.0,
              fit: str = "contain", dim: float = 1.0, crop: tuple | None = None) -> Clip:
    """Decode `n` frames of docs/showcase/<src> at `fps`, `speed`x, scaled to fit (or cover) box_w x box_h.

    `crop` = (x, y, w, h) in source pixels, applied first (e.g. to leave out a burned-in task caption).
    """
    path = SHOWCASE / src
    if not path.is_file():
        sys.exit(f"missing clip: {path}")
    w, h = probe(path)
    pre = ""
    if crop:
        pre = f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]},"
        w, h = crop[2], crop[3]
    s = min(box_w / w, box_h / h) if fit == "contain" else max(box_w / w, box_h / h)
    tw, th = max(2, round(w * s / 2) * 2), max(2, round(h * s / 2) * 2)
    vf = f"{pre}setpts=PTS/{speed},fps={fps},scale={tw}:{th}:flags=lanczos"
    cmd = [ffmpeg_exe(), "-v", "error", "-ss", f"{start}", "-i", str(path), "-vf", vf, "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, th, tw, 3)
    if fit == "cover":
        y0, x0 = (th - box_h) // 2, (tw - box_w) // 2
        arr = arr[:, max(0, y0):max(0, y0) + box_h, max(0, x0):max(0, x0) + box_w]
    if dim != 1.0:
        arr = (arr.astype(np.float32) * dim).astype(np.uint8)
    return Clip(np.ascontiguousarray(arr))


def clip_for(c: dict, box_w: int, box_h: int, n: int, fps: int, fit: str | None = None, dim: float = 1.0) -> Clip:
    crop = tuple(c["crop"]) if c.get("crop") else None
    return load_clip(c["src"], box_w, box_h, n, fps, float(c.get("speed", 1.0)), float(c.get("start", 0.0)),
                     fit or c.get("fit", "contain"), dim, crop)


def paste(canvas: Image.Image, arr: np.ndarray, cx: int, cy: int) -> None:
    """Paste a frame centred on (cx, cy)."""
    h, w = arr.shape[:2]
    canvas.paste(Image.fromarray(arr), (cx - w // 2, cy - h // 2))


def tile(canvas: Image.Image, c: dict, x: int, y: int, w: int, h: int, i: int, n: int, fps: int) -> None:
    """A clip in a w x h box: the clip itself (contain), over a dimmed, cropped copy of it filling the box."""
    back = clip_for(c, w, h, n, fps, fit="cover", dim=0.28)
    canvas.paste(Image.fromarray(back.at(i)), (x, y))
    fit = c.get("fit", "contain")
    front = clip_for(c, w, h, n, fps, fit=fit)
    paste(canvas, front.at(i), x + w // 2, y + h // 2)


# ---------------------------------------------------------------- narration


@dataclass
class Narration:
    pcm: np.ndarray  # int16 mono at SR
    marks: list[dict] = field(default_factory=list)  # Polly word marks: time (ms), start/end byte offsets

    @property
    def seconds(self) -> float:
        return len(self.pcm) / SR


def narrate(text: str, voice: dict, no_voice: bool) -> Narration:
    if no_voice:  # estimate: ~2.6 words per second, words spread evenly
        words = list(re.finditer(r"\S+", text))
        dur = len(words) / 2.6
        marks = [{"time": int(1000 * k * dur / len(words)), "start": m.start(), "end": m.end(),
                  "value": m.group(0).strip(".,:;")} for k, m in enumerate(words)]
        return Narration(np.zeros(int(dur * SR), np.int16), marks)
    key = hashlib.sha256(json.dumps([voice, text], sort_keys=True).encode()).hexdigest()[:16]
    pcm_f, marks_f = CACHE / f"{key}.pcm", CACHE / f"{key}.marks.json"
    if not (pcm_f.is_file() and marks_f.is_file()):
        import boto3

        CACHE.mkdir(parents=True, exist_ok=True)
        polly = boto3.client("polly", region_name=voice.get("region", "us-east-1"))
        common = {"Text": text, "VoiceId": voice["id"], "Engine": voice.get("engine", "neural")}
        audio = polly.synthesize_speech(OutputFormat="pcm", SampleRate=str(SR), **common)["AudioStream"].read()
        sm = polly.synthesize_speech(OutputFormat="json", SpeechMarkTypes=["word"], **common)["AudioStream"].read()
        pcm_f.write_bytes(audio)
        marks_f.write_text(json.dumps([json.loads(line) for line in sm.decode().splitlines() if line.strip()]))
    return Narration(np.frombuffer(pcm_f.read_bytes(), np.int16), json.loads(marks_f.read_text()))


def caption_chunks(text: str, nar: Narration, max_chars: int = 54) -> list[tuple[float, float, str]]:
    """Group word marks into one-line captions: (start s, end s, text), times relative to the narration."""
    raw = text.encode()
    chunks, cur = [], []

    def flush():
        if cur:
            s = raw[cur[0]["start"]:cur[-1]["end"]].decode()
            nxt = raw[cur[-1]["end"]:cur[-1]["end"] + 1].decode(errors="ignore")
            chunks.append([cur[0]["time"] / 1000, s + (nxt if nxt in ".,:;?!" else "")])
            cur.clear()

    for m in nar.marks:
        if cur and len(raw[cur[0]["start"]:m["end"]]) > max_chars:
            flush()
        cur.append(m)
        tail = raw[m["end"]:m["end"] + 1].decode(errors="ignore")
        if tail in ".:;" or (tail == "," and len(raw[cur[0]["start"]:m["end"]]) > 28):
            flush()
    flush()
    out = []
    for k, (t0, s) in enumerate(chunks):
        t1 = chunks[k + 1][0] if k + 1 < len(chunks) else nar.seconds + 0.4
        out.append((t0, t1, s.strip().rstrip(",;")))
    return out


def cue_time(nar: Narration, word: str, default: float) -> float:
    for m in nar.marks:
        if m["value"].lower().strip(".,:;").startswith(word.lower()):
            return m["time"] / 1000
    return default


# ---------------------------------------------------------------- drawing helpers


def blank() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def centered(d: ImageDraw.ImageDraw, cx: int, y: int, s: str, f, fill=TEXT) -> None:
    d.text((cx - text_w(f, s) // 2, y), s, font=f, fill=fill)


def chip(img: Image.Image, x: int, y: int, s: str, size: int = 22, color=ACCENT, alpha: int = 200) -> None:
    """A label on a dark translucent plate with a coloured bar on the left; (x, y) = top-left."""
    f = font(size, "bold")
    tw = text_w(f, s)
    pad = size // 2
    ov = Image.new("RGBA", (tw + 2 * pad + 6, size + pad + 4), (0, 0, 0, alpha))
    od = ImageDraw.Draw(ov)
    od.rectangle((0, 0, 5, ov.height), fill=color + (255,))
    od.text((6 + pad, (ov.height - size) // 2 - 2), s, font=f, fill=TEXT)
    img.paste(ov, (x, y), ov)


def ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def arrow(d: ImageDraw.ImageDraw, p0, p1, frac: float, color, width: int = 4) -> None:
    if frac <= 0:
        return
    x0, y0 = p0
    x1 = x0 + (p1[0] - x0) * frac
    y1 = y0 + (p1[1] - y0) * frac
    d.line((x0, y0, x1, y1), fill=color, width=width)
    if frac >= 0.999:
        v = np.array([p1[0] - x0, p1[1] - y0], float)
        v /= np.linalg.norm(v)
        nrm = np.array([-v[1], v[0]])
        tip = np.array(p1, float)
        a, b = tip - 14 * v + 8 * nrm, tip - 14 * v - 8 * nrm
        d.polygon([tuple(tip), tuple(a), tuple(b)], fill=color)


# ---------------------------------------------------------------- scenes
# Each scene renderer returns frame(i) -> RGB image for i in [0, n).


def grid_background(clips: list[dict], n: int, fps: int, dim: float):
    cols, rows = 3, 2
    tw, th = W // cols + 1, H // rows + 1
    loaded = [clip_for(c, tw, th, n, fps, fit="cover", dim=dim) for c in clips[:cols * rows]]

    def draw(i: int) -> Image.Image:
        img = blank()
        for k, cl in enumerate(loaded):
            img.paste(Image.fromarray(cl.at(i)), ((k % cols) * (W // cols), (k // cols) * (H // rows)))
        return img

    return draw


def scene_title(sc, n, fps, nar):
    bg = grid_background(sc["clips"], n, fps, 0.30)

    def frame(i):
        img = bg(i)
        t = i / fps
        a = int(255 * ease((t - 0.2) / 0.6))
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        d.rectangle((0, 230, W, 450), fill=(10, 14, 20, int(a * 0.9)))
        f1, f2 = font(66, "bold"), font(30)
        d.text(((W - text_w(f1, sc["title"])) // 2, 262), sc["title"], font=f1, fill=TEXT + (a,))
        d.rectangle((W // 2 - 60, 352, W // 2 + 60, 357), fill=ACCENT + (a,))
        d.text(((W - text_w(f2, sc["subtitle"])) // 2, 378), sc["subtitle"], font=f2, fill=(205, 212, 222, a))
        img.paste(ov, (0, 0), ov)
        return img

    return frame


def scene_grid(sc, n, fps, nar):
    clips = sc["clips"]
    mx, top, gap = 40, 96, 14
    tw = (W - 2 * mx - 2 * gap) // 3
    th = (CONTENT_H - top - 8 - gap) // 2

    def frame(i):
        img = blank()
        d = ImageDraw.Draw(img)
        centered(d, W // 2, 30, sc["headline"], font(34, "bold"))
        t = i / fps
        for k, c in enumerate(clips):
            x, y = mx + (k % 3) * (tw + gap), top + (k // 3) * (th + gap)
            if t < 0.15 + 0.3 * k:
                d.rectangle((x, y, x + tw, y + th), fill=PANEL)
                continue
            tile(img, c, x, y, tw, th, i, n, fps)
            chip(img, x + 10, y + 10, c["label"], 22)
        return img

    return frame


def scene_pipeline(sc, n, fps, nar):
    nodes = sc["nodes"]
    total = n / fps
    cues = [LEAD_S + cue_time(nar, nd["cue"], (k + 1) * (total - 2) / (len(nodes) + 1)) for k, nd in enumerate(nodes)]
    nw, nh, gap = 250, 132, 54
    x0 = (W - 4 * nw - 3 * gap) // 2
    row1, row2 = 168, 420
    # snake: CDK, AzSelector, Install, Benchmark on row 1; S3, Email, Terminate right-to-left on row 2
    pos = [(x0 + k * (nw + gap), row1) for k in range(4)] + [(x0 + (3 - k) * (nw + gap), row2) for k in range(3)]
    bench = next(k for k, nd in enumerate(nodes) if nd.get("clip"))
    pw, ph = nw, 2 * nh - 40
    crop = tuple(nodes[bench]["crop"]) if nodes[bench].get("crop") else None
    clip = load_clip(nodes[bench]["clip"], pw - 8, ph - 40, n, fps, 4.0, crop=crop)
    cmd = "$ " + sc["command"]
    mono = font(28, "mono")

    def frame(i):
        t = i / fps
        img = blank()
        d = ImageDraw.Draw(img)
        # terminal strip with the command being typed
        d.rounded_rectangle((150, 34, W - 150, 98), 10, fill=(6, 8, 12), outline=(60, 70, 86), width=2)
        k = int(len(cmd) * ease(t / 1.4))
        shown = cmd[:k] + ("_" if (i // 8) % 2 == 0 or k < len(cmd) else "")
        d.text((176, 50), shown, font=mono, fill=(120, 230, 150))
        # EC2 group around Install + Benchmark
        gx0, gx1 = pos[2][0] - 16, pos[3][0] + nw + 16
        lit_ec2 = t >= cues[1] + 0.3
        col = ACCENT if lit_ec2 else (70, 80, 96)
        d.rounded_rectangle((gx0, row1 - 44, gx1, row1 + nh + 16), 12, outline=col, width=2)
        d.text((gx0 + 14, row1 - 38), "EC2 GPU instance", font=font(20, "bold"), fill=col)
        for k in range(len(nodes) - 1):
            (ax, ay), (bx, by) = pos[k], pos[k + 1]
            if ay == by:
                p0 = (ax + nw + 4, ay + nh // 2) if bx > ax else (ax - 4, ay + nh // 2)
                p1 = (bx - 6, by + nh // 2) if bx > ax else (bx + nw + 6, by + nh // 2)
            else:
                p0, p1 = (ax + nw // 2, ay + nh + 18), (bx + nw // 2, by - 6)
            frac = ease((t - cues[k + 1] + 0.35) / 0.35)
            arrow(d, p0, p1, frac, ACCENT if frac >= 1 else (90, 100, 116))
        for k, nd in enumerate(nodes):
            x, y = pos[k]
            lit = t >= cues[k]
            d.rounded_rectangle((x, y, x + nw, y + nh), 12, fill=PANEL_LIT if lit else PANEL,
                                outline=ACCENT if lit else (52, 60, 74), width=3 if lit else 2)
            centered(d, x + nw // 2, y + 18, nd["title"], font(28, "bold"), TEXT if lit else MUTED)
            for j, line in enumerate(nd["lines"]):
                centered(d, x + nw // 2, y + 64 + 28 * j, line, font(21), (210, 216, 226) if lit else (110, 118, 132))
        # rollout preview under the CDK node once the benchmark runs
        px, py = pos[0][0], row2 - 10
        if t >= cues[bench]:
            d.rounded_rectangle((px, py, px + pw, py + ph), 12, fill=(6, 8, 12), outline=OK, width=2)
            paste(img, clip.at(int((t - cues[bench]) * fps)), px + pw // 2, py + 22 + (ph - 40) // 2)
            d.text((px + 12, py + ph - 32), "rollout video", font=font(19, "bold"), fill=OK)
        return img

    return frame


def scene_montage(sc, n, fps, nar):
    clips = sc["clips"]
    slot = n / len(clips)
    held: dict[int, tuple[Clip, Clip]] = {}  # only the clip on screen stays decoded

    def frame(i):
        k = min(len(clips) - 1, int(i / slot))
        j = int(i - k * slot)
        c = clips[k]
        m = int(slot) + 2
        img = blank()
        if k not in held:
            held.clear()
            load_clip.cache_clear()
            held[k] = (clip_for(c, W, CONTENT_H, m, fps, fit="cover", dim=0.22),
                       clip_for(c, W - 80, CONTENT_H - 30, m, fps, fit="contain"))
        back, front = held[k]
        img.paste(Image.fromarray(back.at(j)), (0, 0))
        paste(img, front.at(j), W // 2, CONTENT_H // 2)
        chip(img, 36, 30, c["label"], 26)
        # progress dots
        d = ImageDraw.Draw(img)
        for q in range(len(clips)):
            cx = W - 36 - (len(clips) - 1 - q) * 22
            d.ellipse((cx - 6, 44 - 6, cx + 6, 44 + 6), fill=ACCENT if q == k else (90, 98, 112))
        return img

    return frame


def scene_compare(sc, n, fps, nar):
    clips = sc["clips"]
    size = 470
    gap = 60
    x0 = (W - 2 * size - gap) // 2
    y0 = 104

    def frame(i):
        img = blank()
        d = ImageDraw.Draw(img)
        centered(d, W // 2, 30, sc["headline"], font(34, "bold"))
        for k, c in enumerate(clips):
            x = x0 + k * (size + gap)
            col = OK if c.get("ok") else FAIL
            d.rounded_rectangle((x - 5, y0 - 5, x + size + 5, y0 + size + 5), 10, outline=col, width=4)
            tile(img, c, x, y0, size, size, i, n, fps)
            centered(d, x + size // 2, y0 + size + 16, c["label"], font(23, "bold"), col)
        return img

    return frame


def scene_bars(sc, n, fps, nar):
    bars = sc["bars"]
    ch = sc["change"]
    t_change = LEAD_S + cue_time(nar, ch["cue"], (n / fps) * 0.7)
    lx, cx0, cx1 = 370, 390, 1000
    top, rh = 116, 58

    def xof(v):
        return cx0 + (cx1 - cx0) * v

    def frame(i):
        t = i / fps
        img = blank()
        d = ImageDraw.Draw(img)
        centered(d, W // 2, 30, sc["headline"], font(34, "bold"))
        f, fb, fs = font(24), font(24, "bold"), font(20)
        # axis
        ybot = top + rh * (len(bars) + 1) + 18
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            d.line((xof(v), top - 8, xof(v), ybot), fill=(44, 52, 66), width=1)
            centered(d, int(xof(v)), ybot + 6, f"{v:g}", fs, MUTED)
        rows = [(b["label"], b["k"] / b["n"], b, BLUE) for b in bars]
        if t >= t_change:
            rows.append((ch["label"], ch["rate"], None, FAIL))
        for k, (label, rate, b, col) in enumerate(rows):
            y = top + rh * k + (14 if b is None else 0)
            start = 0.25 + 0.18 * k if b is not None else t_change
            g = ease((t - start) / 0.8)
            fl = fb if b is None else f
            d.text((lx - text_w(fl, label), y + 8), label, font=fl, fill=FAIL if b is None else TEXT)
            d.rounded_rectangle((cx0, y + 6, max(cx0 + 2, xof(rate * g)), y + rh - 14), 5, fill=col)
            if g >= 1:
                txt = f"{rate:.2f}  ({b['k']}/{b['n']})" if b else f"{rate:.2f}  (was {ch['was']:.2f})"
                d.text((cx1 + 22, y + 8), txt, font=fb if b is None else f, fill=FAIL if b is None else TEXT)
                if b:  # 95% Wilson interval
                    ym = y + (rh - 8) // 2
                    d.line((xof(b["lo"]), ym, xof(b["hi"]), ym), fill=(250, 250, 250), width=3)
                    for v in (b["lo"], b["hi"]):
                        d.line((xof(v), ym - 9, xof(v), ym + 9), fill=(250, 250, 250), width=3)
        d.text((cx0, ybot + 36), "white whisker = 95% Wilson interval   ·   source: " + sc["source"], font=fs, fill=MUTED)
        return img

    return frame


def scene_cost(sc, n, fps, nar):
    total = n / fps
    phases = sc["phases"]
    x0, x1, y = 140, W - 140, 330
    clip = load_clip(sc["clip"], 420, 140, n, fps, 1.0)
    t0, t1 = 0.4, total - 1.6

    def frame(i):
        t = i / fps
        img = blank()
        d = ImageDraw.Draw(img)
        centered(d, W // 2, 30, sc["headline"], font(34, "bold"))
        cur = x0 + (x1 - x0) * ease((t - t0) / (t1 - t0))
        acc = x0
        for p in phases:
            w = (x1 - x0) * p["w"]
            seg = (acc, y, acc + w - 6, y + 64)
            active = p.get("gpu") and acc <= cur < acc + w
            done = cur >= acc
            fill = (ACCENT if active else (150, 96, 20)) if p.get("gpu") and done else (PANEL_LIT if done else PANEL)
            d.rounded_rectangle(seg, 10, fill=fill)
            centered(d, int(acc + w / 2 - 3), y + 18, p["label"], font(22, "bold"), (20, 20, 20) if p.get("gpu") and done else TEXT)
            if p.get("gpu"):
                gx = int(acc + w / 2)
                d.text((gx - text_w(font(20, "bold"), "EC2 GPU billing") // 2, y - 190), "EC2 GPU billing", font=font(20, "bold"), fill=ACCENT)
                if active:
                    paste(img, clip.at(i), gx, y - 92)
            acc += w
        d.line((cur, y - 20, cur, y + 84), fill=TEXT, width=3)
        if t > t0 + 1.0:
            a = ease((t - t0 - 1.0) / 0.6)
            col = tuple(int(c * a + b * (1 - a)) for c, b in zip(ACCENT, BG))
            centered(d, W // 2, 440, sc["range"], font(36, "bold"), col)
            mono = font(26, "mono")
            s = "$ python vlasim.py destroy --vla <target>"
            colm = tuple(int(c * a + b * (1 - a)) for c, b in zip((120, 230, 150), BG))
            centered(d, W // 2, 512, s, mono, colm)
        return img

    return frame


def scene_end(sc, n, fps, nar):
    bg = grid_background(sc["clips"], n, fps, 0.22)

    def frame(i):
        img = bg(i)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        d.rectangle((0, 180, W, 520), fill=(10, 14, 20, 228))
        f = font(34, "bold")
        d.text(((W - text_w(f, sc["url"])) // 2, 214), sc["url"], font=f, fill=TEXT)
        d.rectangle((W // 2 - 60, 282, W // 2 + 60, 287), fill=ACCENT)
        mono = font(28, "mono")
        for k, c in enumerate(sc["commands"]):
            s = "$ " + c
            d.text(((W - text_w(mono, s)) // 2, 330 + 50 * k), s, font=mono, fill=(120, 230, 150))
        fs = font(22)
        s = "MIT-0 license  ·  runs in your own AWS account"
        d.text(((W - text_w(fs, s)) // 2, 450), s, font=fs, fill=MUTED)
        img.paste(ov, (0, 0), ov)
        return img

    return frame


def scene_preview(sc, n, fps, nar):
    clips = sc["clips"]
    cols, rows = 3, 2
    band = 120
    gap = 6
    tw = (W - (cols - 1) * gap) // cols
    th = (H - band - (rows - 1) * gap) // rows

    def frame(i):
        img = blank()
        for k, c in enumerate(clips[:cols * rows]):
            x, y = (k % cols) * (tw + gap), (k // cols) * (th + gap)
            tile(img, c, x, y, tw, th, i, n, fps)
            chip(img, x + 12, y + 12, c["label"], 28)
        d = ImageDraw.Draw(img)
        y = H - band
        d.rectangle((0, y, W, H), fill=(10, 14, 20))
        d.rectangle((0, y, W, y + 4), fill=ACCENT)
        centered(d, W // 2, y + 16, sc["title"], font(46, "bold"))
        centered(d, W // 2, y + 74, sc["subtitle"], font(28), (205, 212, 222))
        return img

    return frame


SCENES = {"title": scene_title, "grid": scene_grid, "pipeline": scene_pipeline, "montage": scene_montage,
          "compare": scene_compare, "bars": scene_bars, "cost": scene_cost, "end": scene_end,
          "preview": scene_preview}


# ---------------------------------------------------------------- timeline + output


@dataclass
class Timed:
    sc: dict
    nar: Narration
    start: float
    n: int
    frame: object = None
    captions: list = field(default_factory=list)


def build(cfg: dict, fps: int, no_voice: bool) -> list[Timed]:
    out, t = [], 0.0
    for sc in cfg["scenes"]:
        nar = narrate(sc["narration"], cfg["voice"], no_voice)
        dur = max(float(sc.get("min_s", 0)), LEAD_S + nar.seconds + TAIL_S)
        n = int(round(dur * fps))
        tm = Timed(sc, nar, t, n)
        tm.frame = SCENES[sc["type"]](sc, n, fps, nar)
        tm.captions = caption_chunks(sc["narration"], nar)
        out.append(tm)
        t += n / fps
    return out


def draw_caption(img: Image.Image, s: str) -> None:
    f = font(30)
    tw = text_w(f, s)
    ov = Image.new("RGBA", (tw + 40, 50), (0, 0, 0, 185))
    ImageDraw.Draw(ov).text((20, 6), s, font=f, fill=(255, 255, 255))
    img.paste(ov, ((W - ov.width) // 2, H - 72), ov)


def render_frame(tm: Timed, i: int, fps: int, captions: bool) -> Image.Image:
    img = tm.frame(i)
    t = i / fps
    if captions:
        for c0, c1, s in tm.captions:
            if LEAD_S + c0 <= t < LEAD_S + c1:
                draw_caption(img, s)
    fade = min(1.0, (i + 1) / (FADE_S * fps), (tm.n - i) / (FADE_S * fps))
    if fade < 1.0:
        img = Image.fromarray((np.asarray(img, np.float32) * fade).astype(np.uint8))
    return img


def write_audio(timeline: list[Timed], fps: int, path: Path) -> float:
    total = sum(tm.n for tm in timeline) / fps
    pcm = np.zeros(int(total * SR) + SR, np.int16)
    for tm in timeline:
        s = int((tm.start + LEAD_S) * SR)
        pcm[s:s + len(tm.nar.pcm)] = tm.nar.pcm
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm[:int(total * SR)].tobytes())
    return total


def encode(frames, out: Path, fps: int, args: list[str], inputs: list[str] = ()) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", *inputs, *args, str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for img in frames:
        p.stdin.write(img.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit(f"ffmpeg failed for {out}")


def main() -> None:
    global W, H
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default=str(Path(__file__).with_name("scenes.json")))
    ap.add_argument("--no-voice", action="store_true", help="no Polly call: silent audio, estimated timings")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--only", choices=["video", "preview"], help="render one of the two outputs")
    ap.add_argument("--crf", type=int, default=27, help="x264 quality (higher = smaller file)")
    ap.add_argument("--stills", metavar="DIR", help="write PNG frames at --at seconds instead of encoding")
    ap.add_argument("--at", default="", help="comma-separated times in seconds for --stills")
    ap.add_argument("--font", help="regular TTF (default: matplotlib's DejaVu Sans)")
    ap.add_argument("--font-bold")
    ap.add_argument("--font-mono")
    a = ap.parse_args()

    cfg = json.loads(Path(a.scenes).read_text())
    W, H = cfg["size"]
    fps = int(cfg["fps"])
    find_fonts(a.font, a.font_bold, a.font_mono)
    timeline = build(cfg, fps, a.no_voice)
    total = sum(tm.n for tm in timeline) / fps
    for tm in timeline:
        print(f"{tm.start:6.2f}s  {tm.n / fps:5.2f}s  {tm.sc['id']:<9} narration {tm.nar.seconds:5.2f}s")
    print(f"total {total:.2f}s")

    def all_frames():
        for tm in timeline:
            load_clip.cache_clear()  # decoded clips of one scene at a time
            for i in range(tm.n):
                yield render_frame(tm, i, fps, not a.no_captions)

    if a.stills:
        d = Path(a.stills)
        d.mkdir(parents=True, exist_ok=True)
        for s in [float(x) for x in a.at.split(",") if x]:
            tm = next((x for x in timeline if x.start <= s < x.start + x.n / fps), timeline[-1])
            i = min(tm.n - 1, int((s - tm.start) * fps))
            p = d / f"frame-{s:05.1f}s-{tm.sc['id']}.png"
            render_frame(tm, i, fps, not a.no_captions).save(p)
            print("wrote", p)
        return

    if a.only in (None, "video"):
        wav = CACHE / "narration.wav"
        write_audio(timeline, fps, wav)
        out = ROOT / cfg["output"]
        encode(all_frames(), out, fps,
               ["-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-b:a", "48k", "-ac", "1", "-movflags", "+faststart", "-shortest"],
               ["-i", str(wav)])
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {total:.1f} s)")

    if a.only in (None, "preview"):
        pv = cfg["preview"]
        sc = cfg["preview_scene"]
        n = int(sc["duration_s"] * pv["fps"])
        fr = SCENES[sc["type"]](sc, n, pv["fps"], None)
        out = ROOT / pv["output"]
        h = round(pv["width"] * H / W / 2) * 2
        encode((fr(i) for i in range(n)), out, pv["fps"],
               ["-vf", f"scale={pv['width']}:{h}:flags=lanczos", "-c:v", "libwebp_anim", "-lossless", "0",
                "-quality", str(pv["quality"]), "-compression_level", "6", "-loop", "0"])
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {n / pv['fps']:.1f} s)")


W, H = 1280, 720

if __name__ == "__main__":
    main()
