from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from oavi_tools import (
    ALPHABET_VARIANT_4,
    ROMANTIC_PHRASE,
    classifier_vector,
    crop_white,
    extract_features,
    render_symbol,
    render_text,
    reset_dir,
    segment_symbols,
    similarity,
    write_csv,
)


def recognize(text: str, size: int, templates: dict[str, dict[str, object]]) -> tuple[str, list[str], int, float]:
    image = render_text(text, size=size, spacing=10)
    boxes = segment_symbols(image)
    lines = []
    recognized = []
    for i, box in enumerate(boxes, start=1):
        segment = crop_white(image.crop(box), border=0)
        vector = classifier_vector(extract_features(segment))
        hypotheses = []
        for symbol, features in templates.items():
            distance = float(np.linalg.norm(vector - classifier_vector(features)))
            hypotheses.append((symbol, similarity(distance)))
        hypotheses.sort(key=lambda item: item[1], reverse=True)
        recognized.append(hypotheses[0][0])
        joined = ", ".join(f"('{symbol}', {score:.4f})" for symbol, score in hypotheses)
        lines.append(f"{i}: [{joined}]")
    target = text.replace(" ", "")
    result = "".join(recognized)
    errors = sum(a != b for a, b in zip(target, result)) + abs(len(target) - len(result))
    accuracy = (len(target) - errors) / len(target) * 100 if target else 0.0
    return result, lines, errors, accuracy


def main() -> None:
    base = Path(__file__).resolve().parent
    output_dir = base / "output"
    reset_dir(output_dir)

    templates = {symbol: extract_features(crop_white(render_symbol(symbol, size=52), border=0)) for symbol in ALPHABET_VARIANT_4}
    result_52, lines_52, errors_52, acc_52 = recognize(ROMANTIC_PHRASE, 52, templates)
    result_56, lines_56, errors_56, acc_56 = recognize(ROMANTIC_PHRASE, 56, templates)

    (output_dir / "hypotheses_size52.txt").write_text("\n".join(lines_52), encoding="utf-8")
    (output_dir / "hypotheses_size56.txt").write_text("\n".join(lines_56), encoding="utf-8")
    write_csv(
        output_dir / "summary.csv",
        [
            {"font_size": 52, "expected": ROMANTIC_PHRASE.replace(" ", ""), "recognized": result_52, "errors": errors_52, "accuracy_percent": f"{acc_52:.2f}"},
            {"font_size": 56, "expected": ROMANTIC_PHRASE.replace(" ", ""), "recognized": result_56, "errors": errors_56, "accuracy_percent": f"{acc_56:.2f}"},
        ],
    )
    print("Lab7: hypotheses and summary were written to output")


if __name__ == "__main__":
    main()
