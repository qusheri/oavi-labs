import math
import numpy as np
from pathlib import Path
from PIL import Image


input_path = Path("01.png")
out_dir = Path("lab_output")
out_dir.mkdir(exist_ok=True)

img = Image.open(input_path).convert("RGB")
arr = np.array(img).astype(np.uint8)


def save_gray(channel: np.ndarray, path: Path):
    Image.fromarray(channel.astype(np.uint8), mode="L").save(path)


def rgb_to_hsi(rgb: np.ndarray):
    rgb_n = rgb.astype(np.float64) / 255.0
    R = rgb_n[..., 0]
    G = rgb_n[..., 1]
    B = rgb_n[..., 2]

    I = (R + G + B) / 3.0

    min_rgb = np.minimum(np.minimum(R, G), B)
    sum_rgb = R + G + B
    S = np.zeros_like(I)
    mask = sum_rgb > 1e-12
    S[mask] = 1.0 - 3.0 * min_rgb[mask] / sum_rgb[mask]

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    den = np.maximum(den, 1e-12)

    theta = np.arccos(np.clip(num / den, -1.0, 1.0))

    H = np.zeros_like(I)
    H[B <= G] = theta[B <= G]
    H[B > G] = 2 * np.pi - theta[B > G]
    H = H / (2 * np.pi)

    return H, S, I


def hsi_to_rgb(H: np.ndarray, S: np.ndarray, I: np.ndarray):
    H = (H % 1.0) * 2 * np.pi

    R = np.zeros_like(I)
    G = np.zeros_like(I)
    B = np.zeros_like(I)

    eps = 1e-12

    m1 = (H >= 0) & (H < 2 * np.pi / 3)
    H1 = H[m1]
    B[m1] = I[m1] * (1 - S[m1])
    R[m1] = I[m1] * (
        1 + (S[m1] * np.cos(H1)) / np.maximum(np.cos(np.pi / 3 - H1), eps)
    )
    G[m1] = 3 * I[m1] - (R[m1] + B[m1])

    m2 = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)
    H2 = H[m2] - 2 * np.pi / 3
    R[m2] = I[m2] * (1 - S[m2])
    G[m2] = I[m2] * (
        1 + (S[m2] * np.cos(H2)) / np.maximum(np.cos(np.pi / 3 - H2), eps)
    )
    B[m2] = 3 * I[m2] - (R[m2] + G[m2])

    m3 = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)
    H3 = H[m3] - 4 * np.pi / 3
    G[m3] = I[m3] * (1 - S[m3])
    B[m3] = I[m3] * (
        1 + (S[m3] * np.cos(H3)) / np.maximum(np.cos(np.pi / 3 - H3), eps)
    )
    R[m3] = 3 * I[m3] - (G[m3] + B[m3])

    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return rgb


def bilinear_resize_manual(image: np.ndarray, scale_x: float, scale_y: float):
    src_h, src_w = image.shape[:2]
    dst_h = max(1, int(round(src_h * scale_y)))
    dst_w = max(1, int(round(src_w * scale_x)))

    if image.ndim == 2:
        image_f = image[..., None].astype(np.float64)
    else:
        image_f = image.astype(np.float64)

    channels = image_f.shape[2]
    out = np.zeros((dst_h, dst_w, channels), dtype=np.float64)

    x_ratio = src_w / dst_w
    y_ratio = src_h / dst_h

    for y_d in range(dst_h):
        y_s = (y_d + 0.5) * y_ratio - 0.5
        y0 = int(math.floor(y_s))
        y1 = min(y0 + 1, src_h - 1)
        y0 = max(y0, 0)
        wy = y_s - y0

        for x_d in range(dst_w):
            x_s = (x_d + 0.5) * x_ratio - 0.5
            x0 = int(math.floor(x_s))
            x1 = min(x0 + 1, src_w - 1)
            x0 = max(x0, 0)
            wx = x_s - x0

            top = (1 - wx) * image_f[y0, x0] + wx * image_f[y0, x1]
            bottom = (1 - wx) * image_f[y1, x0] + wx * image_f[y1, x1]
            out[y_d, x_d] = (1 - wy) * top + wy * bottom

    out = np.clip(np.round(out), 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return out[..., 0]
    return out


# -----------------------------
# 1. Цветовые модели
# -----------------------------

R = arr[..., 0]
G = arr[..., 1]
B = arr[..., 2]

save_gray(R, out_dir / "1_R_channel.png")
save_gray(G, out_dir / "2_G_channel.png")
save_gray(B, out_dir / "3_B_channel.png")

H, S, I = rgb_to_hsi(arr)
I_img = np.clip(np.round(I * 255), 0, 255).astype(np.uint8)
save_gray(I_img, out_dir / "4_HSI_intensity.png")

I_inv = 1.0 - I
rgb_inv_intensity = hsi_to_rgb(H, S, I_inv)
Image.fromarray(rgb_inv_intensity).save(out_dir / "5_inverted_intensity_rgb.png")


# -----------------------------
# 2. Передискретизация
# -----------------------------

M = 2
N = 3
K = M / N

stretch = bilinear_resize_manual(arr, M, M)
compress = bilinear_resize_manual(arr, 1 / N, 1 / N)
two_pass = bilinear_resize_manual(stretch, 1 / N, 1 / N)
one_pass = bilinear_resize_manual(arr, K, K)

Image.fromarray(stretch).save(out_dir / f"6_stretch_x{M}.png")
Image.fromarray(compress).save(out_dir / f"7_compress_x1_{N}.png")
Image.fromarray(two_pass).save(out_dir / f"8_resample_two_pass_x{M}_{N}.png")
Image.fromarray(one_pass).save(out_dir / f"9_resample_one_pass_x{M}_{N}.png")

print("Готово. Результаты сохранены в папку:", out_dir.resolve())