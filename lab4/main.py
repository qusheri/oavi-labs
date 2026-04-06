from pathlib import Path

from PIL import Image
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


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

BINARY_THRESHOLD = 30


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
    padded = np.pad(image, pad_width=1, mode="edge")
    result = np.empty((h, w), dtype=np.int32)
    kernel = kernel.astype(np.int32)

    chunk_rows = 256
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        padded_chunk = padded[y0:y1 + 2, :]
        windows = sliding_window_view(padded_chunk, (3, 3)).astype(np.int32)
        result[y0:y1] = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))

    return result


def threshold_binary(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    Бинаризация по порогу.
    Контуры делаем чёрными на белом фоне,
    чтобы они были различимы в отчёте.
    """
    return np.where(image >= threshold, 0, 255).astype(np.uint8)


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
    print(f"  Порог бинаризации G: {threshold}")

    stem = input_path.stem

    original_path = output_dir / f"{stem}_original.png"
    gray_path = output_dir / f"{stem}_gray.png"
    gx_path = output_dir / f"{stem}_gx.png"
    gy_path = output_dir / f"{stem}_gy.png"
    g_path = output_dir / f"{stem}_g.png"
    binary_path = output_dir / f"{stem}_g_binary.png"

    original_rgb.save(original_path, format="PNG")
    Image.fromarray(gray, mode="L").save(gray_path, format="PNG")
    Image.fromarray(gx_norm, mode="L").save(gx_path, format="PNG")
    Image.fromarray(gy_norm, mode="L").save(gy_path, format="PNG")
    Image.fromarray(g_norm, mode="L").save(g_path, format="PNG")
    Image.fromarray(g_binary, mode="L").save(binary_path, format="PNG")

    print(f"Обработано: {input_path.name}")
    print(f"  -> {original_path.name}")
    print(f"  -> {gray_path.name}")
    print(f"  -> {gx_path.name}")
    print(f"  -> {gy_path.name}")
    print(f"  -> {g_path.name}")
    print(f"  -> {binary_path.name}")


def main():
    base_dir = Path(__file__).resolve().parent
    input_dir = (base_dir.parent / "input_zhest").resolve()
    output_dir = base_dir / "output_images"

    allowed_ext = {".png", ".bmp"}
    threshold = BINARY_THRESHOLD

    if not input_dir.exists():
        print(f"Папка не найдена: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in allowed_ext
        ],
        key=lambda path: path.name,
    )

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
