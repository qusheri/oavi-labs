from __future__ import annotations

import csv
import math
import shutil
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ALPHABET_VARIANT_4 = "АБВГДЕЖЅЗИIКЛМНОПРСТѸФХЦЧШЩЪЫЬѢЮѴѮѰѠѲѦѪ"
ROMANTIC_PHRASE = "ЛЮБОВЬ"

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\times.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\seguisym.ttf"),
    Path(r"C:\Windows\Fonts\calibri.ttf"),
]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def font_path() -> Path | None:
    for path in FONT_CANDIDATES:
        if path.exists():
            return path
    return None


def load_font(size: int = 52) -> ImageFont.ImageFont:
    path = font_path()
    if path:
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def crop_white(img: Image.Image, threshold: int = 245, border: int = 3) -> Image.Image:
    gray = img.convert("L")
    arr = np.array(gray)
    ys, xs = np.where(arr < threshold)
    if len(xs) == 0:
        return gray
    left = max(int(xs.min()) - border, 0)
    top = max(int(ys.min()) - border, 0)
    right = min(int(xs.max()) + border + 1, gray.width)
    bottom = min(int(ys.max()) + border + 1, gray.height)
    return gray.crop((left, top, right, bottom))


def render_symbol(symbol: str, size: int = 52, padding: int = 12) -> Image.Image:
    font = load_font(size)
    probe = Image.new("L", (size * 4, size * 4), 255)
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((padding, padding), symbol, font=font)
    width = max(1, bbox[2] - bbox[0] + 2 * padding)
    height = max(1, bbox[3] - bbox[1] + 2 * padding)
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), symbol, fill=0, font=font)
    return crop_white(img)


def render_text(text: str, size: int = 52, spacing: int = 8, padding: int = 12) -> Image.Image:
    glyphs = [render_symbol(ch, size=size, padding=5) for ch in text if ch != " "]
    width = sum(g.width for g in glyphs) + spacing * max(0, len(glyphs) - 1) + 2 * padding
    height = max(g.height for g in glyphs) + 2 * padding
    img = Image.new("L", (width, height), 255)
    x = padding
    for glyph in glyphs:
        y = padding + (height - 2 * padding - glyph.height) // 2
        img.paste(glyph, (x, y))
        x += glyph.width + spacing
    return crop_white(img, border=padding)


def binarize(img: Image.Image, threshold: int = 180) -> np.ndarray:
    return (np.array(img.convert("L")) < threshold).astype(np.uint8)


def profiles(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    b = binarize(img)
    return b.sum(axis=0), b.sum(axis=1)


def extract_features(img: Image.Image) -> dict[str, float | int]:
    b = binarize(img)
    h, w = b.shape
    mass = int(b.sum())
    yy, xx = np.indices(b.shape)
    if mass:
        cx = float((xx * b).sum() / mass)
        cy = float((yy * b).sum() / mass)
        mx = float((((yy - cy) ** 2) * b).sum())
        my = float((((xx - cx) ** 2) * b).sum())
    else:
        cx = cy = mx = my = 0.0

    y_edges = [0, h // 2, h]
    x_edges = [0, w // 2, w]
    quarters = []
    densities = []
    for yi in range(2):
        for xi in range(2):
            part = b[y_edges[yi] : y_edges[yi + 1], x_edges[xi] : x_edges[xi + 1]]
            q_mass = int(part.sum())
            quarters.append(q_mass)
            densities.append(q_mass / max(1, part.size))
    return {
        "width": w,
        "height": h,
        "mass": mass,
        "q1_mass": quarters[0],
        "q2_mass": quarters[1],
        "q3_mass": quarters[2],
        "q4_mass": quarters[3],
        "q1_density": densities[0],
        "q2_density": densities[1],
        "q3_density": densities[2],
        "q4_density": densities[3],
        "cx": cx,
        "cy": cy,
        "cx_norm": cx / max(1, w - 1),
        "cy_norm": cy / max(1, h - 1),
        "mx": mx,
        "my": my,
        "mx_norm": mx / max(1, mass * h * h),
        "my_norm": my / max(1, mass * w * w),
    }


def classifier_vector(features: dict[str, object]) -> np.ndarray:
    width = float(features["width"])
    height = float(features["height"])
    area = max(1.0, width * height)
    return np.array(
        [
            float(features["mass"]) / area,
            float(features["cx_norm"]),
            float(features["cy_norm"]),
            float(features["mx_norm"]),
            float(features["my_norm"]),
        ],
        dtype=np.float64,
    )


def similarity(distance: float) -> float:
    return 1.0 / (1.0 + distance)


def resize_for_match(img: Image.Image, size: tuple[int, int] = (48, 48)) -> Image.Image:
    return ImageOps.pad(img.convert("L"), size, color=255, method=Image.Resampling.LANCZOS)


def segment_symbols(img: Image.Image, min_width: int = 2, gap_threshold: int = 0) -> list[tuple[int, int, int, int]]:
    b = binarize(img)
    cols = b.sum(axis=0)
    runs: list[tuple[int, int]] = []
    start = None
    for x, value in enumerate(cols):
        if value > gap_threshold and start is None:
            start = x
        elif value <= gap_threshold and start is not None:
            if x - start >= min_width:
                runs.append((start, x))
            start = None
    if start is not None and b.shape[1] - start >= min_width:
        runs.append((start, b.shape[1]))

    boxes = []
    for left, right in runs:
        part = b[:, left:right]
        rows = np.where(part.sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        top = int(rows.min())
        bottom = int(rows.max()) + 1
        boxes.append((left, top, right, bottom))
    return boxes


def draw_boxes(img: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    for i, box in enumerate(boxes, start=1):
        draw.rectangle(box, outline=(220, 30, 30), width=2)
        draw.text((box[0] + 2, max(0, box[1] - 12)), str(i), fill=(220, 30, 30))
    return out


def save_profile_plot(values: np.ndarray, path: Path, title: str, horizontal: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 2.5) if not horizontal else (2.8, 5), dpi=120)
    idx = np.arange(len(values))
    if horizontal:
        ax.barh(idx, values, color="#444")
        ax.invert_yaxis()
        ax.set_ylabel("y")
    else:
        ax.bar(idx, values, color="#444")
        ax.set_xlabel("x")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]], delimiter: str = ";") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def save_histogram(before: np.ndarray, after: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3), dpi=120)
    ax.hist(before.ravel(), bins=64, range=(0, 255), alpha=0.55, label="original")
    ax.hist(after.ravel(), bins=64, range=(0, 255), alpha=0.55, label="contrast")
    ax.set_xlabel("L")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def linear_contrast(gray: np.ndarray) -> np.ndarray:
    arr = gray.astype(np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    if hi <= lo:
        return gray.astype(np.uint8)
    out = (arr - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def ngtdm_matrix(gray: np.ndarray, d: int = 1, levels: int = 16) -> tuple[np.ndarray, dict[str, float]]:
    q = np.clip((gray.astype(np.float32) * levels / 256).astype(np.int32), 0, levels - 1)
    h, w = q.shape
    s = np.zeros(levels, dtype=np.float64)
    n = np.zeros(levels, dtype=np.float64)
    for y in range(d, h - d):
        for x in range(d, w - d):
            center = q[y, x]
            neigh = q[y - d : y + d + 1, x - d : x + d + 1].astype(np.float64)
            avg = (neigh.sum() - center) / (neigh.size - 1)
            s[center] += abs(center - avg)
            n[center] += 1
    total = n.sum()
    p = n / total if total else n
    active = np.where(n > 0)[0]
    eps = 1e-12

    if len(active) <= 1:
        cos = con = bus = 0.0
    else:
        denom = 0.0
        for i in active:
            for j in active:
                denom += p[i] * p[j] * (i - j) ** 2
        cos = float(1.0 / (eps + np.sum(p[active] * s[active])))
        con = float(denom * np.sum(s[active]) / (total + eps))
        bus = float(np.sum(p[active] * s[active]) / (eps + np.sum(p[active] ** 2)))
    return np.vstack([n, s]), {"COS": cos, "CON": con, "BUS": bus}


def save_matrix_image(matrix: np.ndarray, path: Path) -> None:
    arr = np.log1p(matrix.astype(np.float64))
    arr = arr / arr.max() * 255 if arr.max() else arr
    Image.fromarray(arr.astype(np.uint8)).resize((480, 80), Image.Resampling.NEAREST).save(path)


def synthetic_texture(size: int = 256) -> Image.Image:
    y, x = np.indices((size, size))
    base = 120 + 55 * np.sin(x / 8) + 35 * np.cos(y / 13)
    checks = ((x // 24 + y // 24) % 2) * 35
    noise = np.random.default_rng(4).normal(0, 12, (size, size))
    arr = np.clip(base + checks + noise, 0, 255).astype(np.uint8)
    rgb = np.dstack([arr, np.roll(arr, 9, axis=1), np.roll(arr, 15, axis=0)])
    return Image.fromarray(rgb, "RGB")


def save_wav(path: Path, data: np.ndarray, sample_rate: int = 22050) -> None:
    data = np.clip(data, -1, 1)
    pcm = (data * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        sample_rate = f.getframerate()
        frames = f.readframes(f.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return data, sample_rate


def stft(signal: np.ndarray, sample_rate: int, window_size: int = 1024, hop: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = np.hanning(window_size)
    frames = []
    for start in range(0, max(1, len(signal) - window_size + 1), hop):
        frame = signal[start : start + window_size]
        if len(frame) < window_size:
            frame = np.pad(frame, (0, window_size - len(frame)))
        frames.append(np.abs(np.fft.rfft(frame * window)))
    spec = np.array(frames).T
    freqs = np.fft.rfftfreq(window_size, 1 / sample_rate)
    times = np.arange(spec.shape[1]) * hop / sample_rate
    return spec, freqs, times


def save_spectrogram(spec: np.ndarray, freqs: np.ndarray, times: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.pcolormesh(times, freqs + 1, 20 * np.log10(spec + 1e-8), shading="auto", cmap="magma")
    ax.set_yscale("log")
    ax.set_ylim(50, min(10000, freqs.max()))
    ax.set_xlabel("time, s")
    ax.set_ylabel("frequency, Hz")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def synth_plucked_string(duration: float = 4.0, sample_rate: int = 22050) -> np.ndarray:
    rng = np.random.default_rng(4)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    notes = [196.0, 246.94, 293.66, 392.0]
    signal = np.zeros_like(t)
    for i, freq in enumerate(notes):
        start = int(i * duration / len(notes) * sample_rate)
        local = t[: len(t) - start]
        env = np.exp(-3.2 * local)
        tone = sum((1 / k) * np.sin(2 * np.pi * freq * k * local) for k in range(1, 7))
        signal[start:] += 0.18 * env * tone[: len(t) - start]
    signal += rng.normal(0, 0.025, len(signal))
    return signal / max(1e-9, np.max(np.abs(signal)))


def moving_average(signal: np.ndarray, radius: int = 5) -> np.ndarray:
    kernel = np.ones(radius * 2 + 1) / (radius * 2 + 1)
    return np.convolve(signal, kernel, mode="same")


def synth_vowel(formants: list[float], duration: float = 2.0, sample_rate: int = 22050) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    f0 = np.linspace(120, 260, len(t))
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    signal = sum((1 / k) * np.sin(k * phase) for k in range(1, 18))
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    envelope = np.zeros_like(freqs)
    for formant in formants:
        envelope += np.exp(-0.5 * ((freqs - formant) / 70) ** 2)
    voiced = np.fft.irfft(spec * envelope, n=len(signal))
    voiced *= np.hanning(len(voiced))
    return voiced / max(1e-9, np.max(np.abs(voiced)))


def dominant_frequency_range(spec: np.ndarray, freqs: np.ndarray, threshold: float = 0.2) -> tuple[float, float]:
    energy = spec.mean(axis=1)
    mask = energy > energy.max() * threshold
    selected = freqs[mask]
    selected = selected[selected > 40]
    if len(selected) == 0:
        return 0.0, 0.0
    return float(selected.min()), float(selected.max())


def strongest_formants(spec: np.ndarray, freqs: np.ndarray, count: int = 3) -> list[float]:
    energy = spec.mean(axis=1)
    band = (freqs >= 200) & (freqs <= 3500)
    idx = np.argsort(energy[band])[-count:]
    selected = freqs[band][idx]
    return sorted(float(x) for x in selected)
