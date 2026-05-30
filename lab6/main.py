from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from oavi_tools import (
    ALPHABET_VARIANT_4,
    ROMANTIC_PHRASE,
    draw_boxes,
    profiles,
    render_symbol,
    render_text,
    reset_dir,
    save_profile_plot,
    segment_symbols,
    write_csv,
)


def main() -> None:
    base = Path(__file__).resolve().parent
    output_dir = base / "output"
    segments_dir = base / "segments"
    profiles_dir = base / "profiles"
    reset_dir(output_dir)
    reset_dir(segments_dir)
    reset_dir(profiles_dir)

    phrase = render_text(ROMANTIC_PHRASE, size=52, spacing=10)
    phrase.save(output_dir / "phrase.bmp")
    px, py = profiles(phrase)
    save_profile_plot(px, output_dir / "phrase_profile_x.png", "Phrase vertical profile")
    save_profile_plot(py, output_dir / "phrase_profile_y.png", "Phrase horizontal profile", horizontal=True)

    boxes = segment_symbols(phrase, gap_threshold=0)
    draw_boxes(phrase, boxes).save(output_dir / "phrase_segmented.png")
    rows = []
    for index, box in enumerate(boxes, start=1):
        segment = phrase.crop(box)
        segment.save(segments_dir / f"segment_{index:02d}.bmp")
        rows.append({"index": index, "left": box[0], "top": box[1], "right": box[2], "bottom": box[3]})
    write_csv(output_dir / "segments.csv", rows)

    for index, symbol in enumerate(ALPHABET_VARIANT_4, start=1):
        img = render_symbol(symbol, size=52)
        sx, sy = profiles(img)
        save_profile_plot(sx, profiles_dir / f"{index:02d}_U{ord(symbol):04X}_x.png", f"{symbol}: X")
        save_profile_plot(sy, profiles_dir / f"{index:02d}_U{ord(symbol):04X}_y.png", f"{symbol}: Y", horizontal=True)

    print(f"Lab6: segmented {len(boxes)} symbols")


if __name__ == "__main__":
    main()
