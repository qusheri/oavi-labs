from pathlib import Path

from PIL import Image
import numpy as np


SOBEL_GX = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
], dtype=np.int32)

SOBEL_GY = np.array([
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1]
], dtype=np.int32)


def to_grayscale_weighted(rgb: np.ndarray) -> np.ndarray:
    """
    Перевод RGB -> полутон.
    Y = 0.299R + 0.587G + 0.114B
    """
    gray = (
        0.299 * rgb[:, :, 0] +
        0.587 * rgb[:, :, 1] +
        0.114 * rgb[:, :, 2]
    )
    return np.clip(gray, 0, 255).astype(np.uint8)


def normalize_to_255(arr: np.ndarray) -> np.ndarray:
    """
    Нормализация массива в диапазон 0..255.
    """
    arr = arr.astype(np.float32)
    min_val = arr.min()
    max_val = arr.max()

    if max_val == min_val:
        return np.zeros(arr.shape, dtype=np.uint8)

    norm = (arr - min_val) * 255.0 / (max_val - min_val)
    return np.clip(norm, 0, 255).astype(np.uint8)


def convolve_3x3(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Свёртка 3x3 без библиотечных функций.
    Граничные пиксели обрабатываются повторением ближайшего края.
    """
    h, w = image.shape
    result = np.zeros((h, w), dtype=np.int32)

    for y in range(h):
        for x in range(w):
            s = 0

            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    yy = min(max(y + ky, 0), h - 1)
                    xx = min(max(x + kx, 0), w - 1)
                    s += int(image[yy, xx]) * int(kernel[ky + 1, kx + 1])

            result[y, x] = s

    return result


def threshold_binary(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    Бинаризация по порогу.
    """
    return np.where(image >= threshold, 255, 0).astype(np.uint8)


def process_one_image(input_path: Path, output_dir: Path, threshold: int) -> None:
    img = Image.open(input_path)

    original_rgb = img.convert("RGB")
    rgb_array = np.array(original_rgb, dtype=np.float32)

    gray = to_grayscale_weighted(rgb_array)

    gx_raw = convolve_3x3(gray, SOBEL_GX)
    gy_raw = convolve_3x3(gray, SOBEL_GY)

    g_raw = np.abs(gx_raw) + np.abs(gy_raw)

    gx_norm = normalize_to_255(gx_raw)
    gy_norm = normalize_to_255(gy_raw)
    g_norm = normalize_to_255(g_raw)

    g_binary = threshold_binary(g_norm, threshold)

    stem = input_path.stem

    original_path = output_dir / f"{stem}_original.bmp"
    gray_path = output_dir / f"{stem}_gray.bmp"
    gx_path = output_dir / f"{stem}_gx.bmp"
    gy_path = output_dir / f"{stem}_gy.bmp"
    g_path = output_dir / f"{stem}_g.bmp"
    binary_path = output_dir / f"{stem}_g_binary.bmp"

    original_rgb.save(original_path, format="BMP")
    Image.fromarray(gray, mode="L").save(gray_path, format="BMP")
    Image.fromarray(gx_norm, mode="L").save(gx_path, format="BMP")
    Image.fromarray(gy_norm, mode="L").save(gy_path, format="BMP")
    Image.fromarray(g_norm, mode="L").save(g_path, format="BMP")
    Image.fromarray(g_binary, mode="L").save(binary_path, format="BMP")

    print(f"Обработано: {input_path.name}")
    print(f"  -> {original_path.name}")
    print(f"  -> {gray_path.name}")
    print(f"  -> {gx_path.name}")
    print(f"  -> {gy_path.name}")
    print(f"  -> {g_path.name}")
    print(f"  -> {binary_path.name}")


def main():
    input_dir = Path("../input_zhest")
    output_dir = Path("output_images")

    allowed_ext = {".png", ".bmp"}
    threshold = 100

    if not input_dir.exists():
        print(f"Папка не найдена: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_ext
    ]

    if not files:
        print("Подходящие изображения не найдены.")
        print("Поддерживаются только .png и .bmp")
        return

    print(f"Найдено файлов: {len(files)}")

    for file_path in files:
        try:
            process_one_image(file_path, output_dir, threshold)
        except Exception as e:
            print(f"Ошибка при обработке {file_path.name}: {e}")

    print("Готово.")


if __name__ == "__main__":
    main()