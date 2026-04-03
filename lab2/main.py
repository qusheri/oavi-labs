import os
from pathlib import Path

from PIL import Image
import numpy as np


def to_grayscale_weighted(rgb: np.ndarray) -> np.ndarray:
    """
    Перевод RGB -> полутоновое изображение без библиотечной функции grayscale.
    Y = 0.299R + 0.587G + 0.114B
    """
    gray = (
        0.299 * rgb[:, :, 0] +
        0.587 * rgb[:, :, 1] +
        0.114 * rgb[:, :, 2]
    )
    return np.clip(gray, 0, 255).astype(np.uint8)


def bradley_roth_binarization(gray: np.ndarray, window_size: int = 3, t: float = 0.08) -> np.ndarray:
    """
    Адаптивная бинаризация Брэдли и Рота.
    gray        - 2D массив uint8
    window_size - размер окна, должен быть нечётным
    t           - коэффициент порога
    """
    if window_size % 2 == 0:
        raise ValueError("window_size должен быть нечётным")

    h, w = gray.shape
    r = window_size // 2

    integral = np.zeros((h + 1, w + 1), dtype=np.uint64)
    integral[1:, 1:] = gray.cumsum(axis=0, dtype=np.uint64).cumsum(axis=1, dtype=np.uint64)

    binary = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        y0 = max(0, y - r)
        y1 = min(h - 1, y + r)

        for x in range(w):
            x0 = max(0, x - r)
            x1 = min(w - 1, x + r)

            area = (y1 - y0 + 1) * (x1 - x0 + 1)

            local_sum = (
                integral[y1 + 1, x1 + 1]
                - integral[y0, x1 + 1]
                - integral[y1 + 1, x0]
                + integral[y0, x0]
            )

            local_mean = local_sum / area

            if gray[y, x] < local_mean * (1.0 - t):
                binary[y, x] = 0
            else:
                binary[y, x] = 255

    return binary


def process_image(input_path: Path, output_dir: Path, window_size: int = 3, t: float = 0.08) -> None:
    img = Image.open(input_path).convert("RGB")
    rgb = np.array(img, dtype=np.float32)

    gray = to_grayscale_weighted(rgb)
    binary = bradley_roth_binarization(gray, window_size=window_size, t=t)

    stem = input_path.stem
    gray_path = output_dir / f"{stem}_grayscale.png"
    binary_path = output_dir / f"{stem}_binary_bradley.png"

    Image.fromarray(gray, mode="L").save(gray_path, format="PNG")
    Image.fromarray(binary, mode="L").save(binary_path, format="PNG")

    print(f"Обработано: {input_path.name}")
    print(f"  -> {gray_path.name}")
    print(f"  -> {binary_path.name}")


def main():
    input_dir = Path(r"../input_zhest")
    output_dir = Path(r"output_images")

    allowed_ext = {".png", ".bmp"}
    window_size = 3
    t = 0.08

    if not input_dir.exists():
        print(f"Папка не найдена: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_ext]

    if not files:
        print("Подходящие изображения не найдены.")
        print("Поддерживаются только .png и .bmp")
        return

    print(f"Найдено файлов: {len(files)}")

    for file_path in files:
        try:
            process_image(file_path, output_dir, window_size=window_size, t=t)
        except Exception as e:
            print(f"Ошибка при обработке {file_path.name}: {e}")

    print("Готово.")


if __name__ == "__main__":
    main()