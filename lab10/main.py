from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from oavi_tools import (
    dominant_frequency_range,
    reset_dir,
    save_spectrogram,
    save_wav,
    stft,
    strongest_formants,
    synth_vowel,
    write_csv,
)


def main() -> None:
    base = Path(__file__).resolve().parent
    output_dir = base / "output"
    reset_dir(output_dir)

    sample_rate = 22050
    signals = {
        "vowel_A": synth_vowel([730, 1090, 2440], sample_rate=sample_rate),
        "vowel_I": synth_vowel([270, 2290, 3010], sample_rate=sample_rate),
        "extra_variant4_signal": synth_vowel([500, 1700, 2500], sample_rate=sample_rate),
    }

    rows = []
    for name, signal in signals.items():
        wav_path = output_dir / f"{name}.wav"
        save_wav(wav_path, signal, sample_rate)
        spec, freqs, times = stft(signal, sample_rate, window_size=2048, hop=220)
        save_spectrogram(spec, freqs, times, output_dir / f"{name}_spectrogram.png", name)
        f_min, f_max = dominant_frequency_range(spec, freqs)
        formants = strongest_formants(spec, freqs, count=3)
        rows.append(
            {
                "sample": name,
                "min_frequency_hz": f"{f_min:.2f}",
                "max_frequency_hz": f"{f_max:.2f}",
                "formant_1_hz": f"{formants[0]:.2f}",
                "formant_2_hz": f"{formants[1]:.2f}",
                "formant_3_hz": f"{formants[2]:.2f}",
            }
        )
    write_csv(output_dir / "voice_analysis.csv", rows)
    print("Lab10: generated voice-like samples and spectrograms")


if __name__ == "__main__":
    main()
