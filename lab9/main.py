from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from oavi_tools import (
    moving_average,
    read_wav,
    reset_dir,
    save_spectrogram,
    save_wav,
    stft,
    synth_plucked_string,
    write_csv,
)


def main() -> None:
    base = Path(__file__).resolve().parent
    output_dir = base / "output"
    reset_dir(output_dir)

    sample_rate = 22050
    signal = synth_plucked_string(sample_rate=sample_rate)
    save_wav(output_dir / "instrument_noisy.wav", signal, sample_rate)

    loaded, sr = read_wav(output_dir / "instrument_noisy.wav")
    denoised = moving_average(loaded, radius=4)
    save_wav(output_dir / "instrument_denoised.wav", denoised, sr)

    spec_before, freqs, times = stft(loaded, sr)
    spec_after, _, _ = stft(denoised, sr)
    save_spectrogram(spec_before, freqs, times, output_dir / "spectrogram_before.png", "Before denoising")
    save_spectrogram(spec_after, freqs, times, output_dir / "spectrogram_after.png", "After denoising")

    noise_estimate = float(np.std(loaded - denoised))
    rows = []
    time_step = 0.1
    freq_step = 50
    energy = spec_before ** 2
    for ti, t in enumerate(times):
        if abs((t / time_step) - round(t / time_step)) > 0.05:
            continue
        band_energy = []
        for f0 in np.arange(50, 3000, freq_step):
            mask = (freqs >= f0) & (freqs < f0 + freq_step)
            band_energy.append((float(energy[mask, ti].sum()), f0, f0 + freq_step, t))
        rows.extend(sorted(band_energy, reverse=True)[:1])
    top = sorted(rows, reverse=True)[:10]
    write_csv(
        output_dir / "energy_peaks.csv",
        [{"energy": f"{e:.6f}", "f_from": f1, "f_to": f2, "time": f"{t:.3f}"} for e, f1, f2, t in top],
    )
    write_csv(output_dir / "noise_summary.csv", [{"noise_std_estimate": f"{noise_estimate:.6f}"}])
    print(f"Lab9: noise std estimate {noise_estimate:.6f}")


if __name__ == "__main__":
    main()
