"""Render a short (~35 s) narrated walkthrough video of the real result figures.

Every visual is an existing file from results/; the cursor and captions only point at what is
already in those figures. Narration uses the offline Windows speech synthesizer (System.Speech);
if it is unavailable the video is rendered silent with captions only.

Usage (Windows, from the repository root):  python scripts/make_demo_video.py
Requires: Pillow, numpy, imageio-ffmpeg (pip install imageio-ffmpeg).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "demo" / "tactilesense_demo.mp4"

W, H, FPS = 1920, 1080, 30
BG, INK, MUTED, ACCENT = (252, 252, 251), (11, 11, 11), (82, 81, 78), (42, 120, 214)
FONT_DIR = Path("C:/Windows/Fonts")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default(size)


@dataclass
class Scene:
    title: str
    caption: str
    narration: str
    image: str | None = None
    # cursor targets in *image pixel* coordinates; each is (x, y, ring?) visited in order
    targets: list[tuple[float, float, bool]] = field(default_factory=list)
    min_seconds: float = 4.5
    crop: tuple[int, int, int, int] | None = None  # (left, top, right, bottom) in image pixels
    split_at: int | None = None  # wide figures: cut at this x and stack the right part below


SCENES = [
    Scene(
        "TactileSense AI",
        "Which digit did the robot touch? Predicted from ONE tactile-sensor image.",
        "Tactile Sense A I. From one tactile-sensor image, can a model tell which digit a robot touched?",
        image="images/original_tactile_image.png",
        targets=[(150, 128, False), (55, 128, True)],
    ),
    Scene(
        "1 · OpenCV preprocessing",
        "Subtract the sensor's lighting  →  only gel deformation remains  →  contact region found",
        "OpenCV removes the sensor's lighting, leaving the gel deformation and the contact region.",
        image="images/opencv_preprocessing.png",
        targets=[(172, 230, True), (540, 230, True), (1640, 230, True), (2017, 240, True)],
        split_at=1090,
        min_seconds=6.5,
    ),
    Scene(
        "2 · Results on 100 unseen digits",
        "One touch: CNN 0.17, HOG + SVM 0.17  —  chance is 0.10. Both learn; neither clearly wins.",
        "On a hundred unseen digits, both models reach about seventeen percent. Chance is ten.",
        image="plots/model_comparison.png",
        targets=[(240, 500, True), (365, 500, True), (600, 543, True)],
    ),
    Scene(
        "3 · More touches, better answer",
        "Averaging 32 touches of the same digit roughly doubles accuracy: 0.32 – 0.36",
        "Averaging thirty two touches roughly doubles that.",
        image="plots/multi_touch_fusion.png",
        targets=[(168, 493, True), (537, 460, False), (903, 418, False), (1087, 405, True)],
    ),
    Scene(
        "4 · Where it gets confused",
        "Diagonal = correct. One touch sees only part of a stroke, so a 5 is often read as a 3.",
        "One touch sees part of a stroke, so a five is often read as a three.",
        image="plots/confusion_matrix_cnn.png",
        targets=[(130, 97, False), (790, 757, False), (350, 463, True)],
    ),
    Scene(
        "5 · An honest failure",
        "No digit stroke here — the sensor pressed the mat — yet the CNN said \"8\" with 99.7 %.",
        "A failure: the sensor hit the mat, yet the CNN said eight with near certainty.",
        image="images/prediction_failure_example.png",
        targets=[(198, 310, True), (1548, 322, True)],
        crop=(0, 70, 1680, 598),
    ),
    Scene(
        "6 · Live Streamlit demo",
        "Pick a test frame  →  see preprocessing and both models' predictions",
        "The Streamlit demo shows it all live.",
        image="images/streamlit_demo_prediction.png",
        targets=[(440, 445, True), (430, 478, False), (915, 445, True)],
        crop=(360, 85, 1320, 790),
        min_seconds=5.5,
    ),
]

HEADER_H, CAPTION_H, MARGIN = 120, 170, 60


# --------------------------------------------------------------------------- narration


def synthesize(text: str, path: Path) -> bool:
    """Offline TTS with Windows System.Speech; returns False if unavailable."""
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "try { $s.SelectVoice('Microsoft Zira Desktop') } catch {};"
        "$s.Rate = 2;"
        f"$s.SetOutputToWaveFile('{path}');"
        f"$s.Speak(@'\n{text}\n'@);"
        "$s.Dispose()"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)
        return path.is_file()
    except (OSError, subprocess.CalledProcessError):
        return False


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
        return data, w.getframerate()


# --------------------------------------------------------------------------- drawing


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


def base_frame(scene: Scene, index: int) -> tuple[Image.Image, tuple[float, float, float]]:
    """Static slide: header, centred figure, caption. Returns image and (scale, x0, y0) of figure."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, HEADER_H], fill=(255, 255, 255))
    d.line([0, HEADER_H, W, HEADER_H], fill=(228, 227, 223), width=2)
    d.text((MARGIN, HEADER_H / 2), scene.title, font=font(52, True), fill=INK, anchor="lm")
    d.text((W - MARGIN, HEADER_H / 2), f"{index + 1} / {len(SCENES)}", font=font(30), fill=MUTED, anchor="rm")

    transform = (1.0, 0.0, 0.0)
    if scene.image:
        fig = Image.open(RES / scene.image).convert("RGB")
        if scene.crop:
            fig = fig.crop(scene.crop)
        if scene.split_at:
            left, right = fig.crop((0, 0, scene.split_at, fig.height)), fig.crop((scene.split_at, 0, fig.width, fig.height))
            stacked = Image.new("RGB", (max(left.width, right.width), 2 * fig.height), (255, 255, 255))
            stacked.paste(left, (0, 0))
            stacked.paste(right, (0, fig.height))
            fig = stacked
        box_w, box_h = W - 2 * MARGIN, H - HEADER_H - CAPTION_H - 40
        scale = min(box_w / fig.width, box_h / fig.height)
        if index == 0:
            scale = min(scale, 2.4)  # small raw frame: keep pixels crisp but not huge
        fw, fh = int(fig.width * scale), int(fig.height * scale)
        fig = fig.resize((fw, fh), Image.NEAREST if index == 0 else Image.LANCZOS)
        x0, y0 = (W - fw) // 2, HEADER_H + 20 + (box_h - fh) // 2
        img.paste(fig, (x0, y0))
        transform = (scale, x0, y0)

    cap_top = H - CAPTION_H
    d.rectangle([0, cap_top, W, H], fill=(255, 255, 255))
    d.line([0, cap_top, W, cap_top], fill=(228, 227, 223), width=2)
    d.rectangle([0, cap_top, 12, H], fill=ACCENT)
    fnt = font(40, index == 0)
    lines = wrap(d, scene.caption, fnt, W - 2 * MARGIN)
    total = len(lines) * 52
    for i, line in enumerate(lines):
        d.text((W / 2, cap_top + (CAPTION_H - total) / 2 + i * 52 + 26), line, font=fnt, fill=INK, anchor="mm")
    return img, transform


CURSOR = [(0, 0), (0, 34), (9, 26), (15, 40), (21, 37), (15, 24), (27, 24)]


def draw_cursor(d: ImageDraw.ImageDraw, x: float, y: float) -> None:
    pts = [(x + px * 1.3, y + py * 1.3) for px, py in CURSOR]
    d.polygon([(px + 2, py + 3) for px, py in pts], fill=(0, 0, 0, 70))  # soft shadow
    d.polygon(pts, fill=(255, 255, 255), outline=(0, 0, 0), width=2)


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def cursor_track(points: list[tuple[float, float, bool]], duration: float):
    """Return f(t) -> (x, y, ring_strength). Cursor enters, then moves/dwells over targets."""
    start = (W * 0.92, H * 0.75)
    n = len(points)
    lead = 0.12 * duration                      # time before first arrival
    seg = (duration * 0.86 - lead) / max(n, 1)  # time per target (move + dwell)

    def at(t: float):
        if t < lead:
            a = ease(t / lead)
            x0, y0 = start
            x1, y1, _ = points[0]
            return x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, 0.0
        k = min(int((t - lead) // seg), n - 1)
        local = (t - lead - k * seg) / seg
        x1, y1, ring = points[k]
        if k + 1 < n and local > 0.55:          # move to the next target in the last 45 %
            a = ease((local - 0.55) / 0.45)
            x2, y2, _ = points[k + 1]
            return x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, 0.0
        strength = min(local / 0.15, 1.0) if ring else 0.0
        return x1, y1, strength

    return at


def render_scene(scene: Scene, index: int, duration: float):
    base, (scale, x0, y0) = base_frame(scene, index)
    h = Image.open(RES / scene.image).height if scene.image else 0
    targets = []
    for x, y, ring in scene.targets:
        if scene.crop:
            x, y = x - scene.crop[0], y - scene.crop[1]
            h = scene.crop[3] - scene.crop[1]
        if scene.split_at and x >= scene.split_at:
            x, y = x - scene.split_at, y + h
        targets.append((x0 + x * scale, y0 + y * scale, ring))
    track = cursor_track(targets, duration) if targets else None
    frames = int(round(duration * FPS))
    for f in range(frames):
        t = f / FPS
        frame = base.copy()
        if track:
            x, y, ring = track(t)
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            if ring > 0:
                r = 34 + 10 * np.sin(t * 6)
                od.ellipse([x - r, y - r, x + r, y + r], outline=(227, 73, 72, int(230 * ring)), width=6)
                od.ellipse([x - r - 14, y - r - 14, x + r + 14, y + r + 14], fill=(227, 73, 72, int(35 * ring)))
            draw_cursor(od, x, y)
            frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
        # short fade in/out so scene changes are smooth
        fade = min(f / (0.25 * FPS), (frames - 1 - f) / (0.25 * FPS), 1.0)
        if fade < 1.0:
            frame = Image.blend(Image.new("RGB", (W, H), BG), frame, max(fade, 0.0))
        yield np.asarray(frame)


# --------------------------------------------------------------------------- main


def main() -> None:
    try:
        import imageio_ffmpeg
    except ImportError:
        sys.exit("Install imageio-ffmpeg: pip install imageio-ffmpeg")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clips, rate, durations = [], 22050, []
        for i, scene in enumerate(SCENES):
            wav = tmp / f"scene{i}.wav"
            if synthesize(scene.narration, wav):
                audio, rate = read_wav(wav)
                clips.append(audio)
                durations.append(max(scene.min_seconds, len(audio) / rate + 0.5))
            else:
                clips.append(None)
                durations.append(max(scene.min_seconds, 4.5))
        print("scene durations [s]:", [round(d, 2) for d in durations], "total", round(sum(durations), 1))

        has_audio = any(c is not None for c in clips)
        audio_path = tmp / "narration.wav"
        if has_audio:
            track = []
            for clip, dur in zip(clips, durations):
                n = int(round(dur * FPS)) / FPS * rate  # match the rendered frame count exactly
                pad_front = int(0.3 * rate)
                body = np.zeros(int(n), dtype=np.int16)
                if clip is not None:
                    body[pad_front:pad_front + len(clip)] = clip[: len(body) - pad_front]
                track.append(body)
            with wave.open(str(audio_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(np.concatenate(track).tobytes())

        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-"]
        if has_audio:
            cmd += ["-i", str(audio_path), "-c:a", "aac", "-b:a", "128k"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
                "-movflags", "+faststart", "-shortest", str(OUT)]
        OUT.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i, (scene, dur) in enumerate(zip(SCENES, durations)):
            for frame in render_scene(scene, i, dur):
                proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit("ffmpeg failed")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
