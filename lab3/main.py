from pathlib import Path

from PIL import Image
import numpy as np


def median_filter_3x3(image: np.ndarray) -> np.ndarray:
    """
    Медианный фильтр 3x3.
    Работает для 2D массива uint8.
    Без использования библиотечных функций фильтрации.
    """
    h, w = image.shape
    result = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            window = []

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy = min(max(y + dy, 0), h - 1)
                    xx = min(max(x + dx, 0), w - 1)
                    window.append(int(image[yy, xx]))

            window.sort()
            result[y, x] = window[4]

    return result


def is_monochrome(image: np.ndarray) -> bool:
    """
    Считаем изображение монохромным,
    если в нём только значения 0 и 255.
    """
    unique_vals = np.unique(image)
    return set(unique_vals.tolist()).issubset({0, 255})


def xor_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    XOR для монохромного изображения.
    """
    return np.bitwise_xor(img1, img2).astype(np.uint8)


def abs_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Модуль разности для полутонового изображения.
    """
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    return np.clip(diff, 0, 255).astype(np.uint8)


def enhance_difference(diff: np.ndarray, factor: int = 10) -> np.ndarray:
    """
    Усиление контраста разностного изображения.
    """
    enhanced = diff.astype(np.int16) * factor
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def process_one_image(input_path: Path, output_dir: Path) -> None:
    """
    Обрабатывает одно изображение:
    1. медианный фильтр 3x3
    2. разностное изображение
    """
    img = Image.open(input_path).convert("L")
    src = np.array(img, dtype=np.uint8)

    filtered = median_filter_3x3(src)

    mono = is_monochrome(src)

    if mono:
        diff = xor_difference(src, filtered)
    else:
        diff = abs_difference(src, filtered)

    stem = input_path.stem

    original_path = output_dir / f"{stem}_original.png"
    filtered_path = output_dir / f"{stem}_median3x3.png"
    diff_path = output_dir / f"{stem}_diff.png"
    diff_x10_path = output_dir / f"{stem}_diff_x10.png"

    Image.fromarray(src, mode="L").save(original_path, format="PNG")
    Image.fromarray(filtered, mode="L").save(filtered_path, format="PNG")
    Image.fromarray(diff, mode="L").save(diff_path, format="PNG")

    print(f"Обработано: {input_path.name}")
    print(f"  Тип: {'монохром' if mono else 'полутон'}")
    print(f"  -> {filtered_path.name}")
    print(f"  -> {diff_path.name}")

    if not mono:
        enhanced = enhance_difference(diff, factor=10)
        Image.fromarray(enhanced, mode="L").save(diff_x10_path, format="PNG")


def main():
    input_dir = Path(r"../input_zhest")
    output_dir = Path(r"output_images")

    allowed_ext = {".png", ".bmp"}

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
            process_one_image(file_path, output_dir)
        except Exception as e:
            print(f"Ошибка при обработке {file_path.name}: {e}")

    print("Готово.")


if __name__ == "__main__":
    main()