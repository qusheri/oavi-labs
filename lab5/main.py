from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from oavi_tools import (
    ALPHABET_VARIANT_4,
    extract_features,
    profiles,
    render_symbol,
    reset_dir,
    save_profile_plot,
    write_csv,
)


def main() -> None:
    base = Path(__file__).resolve().parent
    symbols_dir = base / "symbols"
    profiles_dir = base / "profiles"
    output_dir = base / "output"
    reset_dir(symbols_dir)
    reset_dir(profiles_dir)
    reset_dir(output_dir)

    rows = []
    for index, symbol in enumerate(ALPHABET_VARIANT_4, start=1):
        img = render_symbol(symbol, size=52)
        file_name = f"{index:02d}_U{ord(symbol):04X}.png"
        img.save(symbols_dir / file_name)

        row = {"index": index, "symbol": symbol, "code": f"U+{ord(symbol):04X}", "file": file_name}
        row.update(extract_features(img))
        rows.append(row)

        px, py = profiles(img)
        save_profile_plot(px, profiles_dir / f"{index:02d}_profile_x.png", f"{symbol}: X")
        save_profile_plot(py, profiles_dir / f"{index:02d}_profile_y.png", f"{symbol}: Y", horizontal=True)

    write_csv(output_dir / "features.csv", rows)
    print(f"Lab5: generated {len(rows)} symbols for variant 4")


if __name__ == "__main__":
    main()
