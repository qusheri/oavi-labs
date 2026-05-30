from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from oavi_tools import (
    linear_contrast,
    ngtdm_matrix,
    reset_dir,
    save_histogram,
    save_matrix_image,
    synthetic_texture,
    write_csv,
)


def process_image(name: str, image: Image.Image, output_dir: Path) -> dict[str, object]:
    image_dir = output_dir / name
    image_dir.mkdir(parents=True, exist_ok=True)
    image.save(image_dir / "original.png")
    gray = np.array(image.convert("L"))
    Image.fromarray(gray).save(image_dir / "gray.png")
    contrasted = linear_contrast(gray)
    Image.fromarray(contrasted).save(image_dir / "gray_contrast_linear.png")
    save_histogram(gray, contrasted, image_dir / "histograms.png")

    matrix_before, features_before = ngtdm_matrix(gray, d=1)
    matrix_after, features_after = ngtdm_matrix(contrasted, d=1)
    save_matrix_image(matrix_before, image_dir / "ngtdm_original.png")
    save_matrix_image(matrix_after, image_dir / "ngtdm_contrast.png")

    row = {"image": name}
    for key, value in features_before.items():
        row[f"{key}_original"] = f"{value:.6f}"
    for key, value in features_after.items():
        row[f"{key}_contrast"] = f"{value:.6f}"
    return row


def main() -> None:
    base = Path(__file__).resolve().parent
    output_dir = base / "output"
    reset_dir(output_dir)

    images = {
        "synthetic_texture": synthetic_texture(),
        "document_01": Image.open(base.parent / "input_zhest" / "01.png").convert("RGB").resize((360, 260)),
    }
    rows = [process_image(name, image, output_dir) for name, image in images.items()]
    write_csv(output_dir / "ngtdm_features.csv", rows)
    print("Lab8: NGTDM d=1, COS/CON/BUS, linear contrast")


if __name__ == "__main__":
    main()
