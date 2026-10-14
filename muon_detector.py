"""
Signalsphere: Cosmic Muon Detector Simulator

Simulates a two-paddle coincidence muon telescope:
  1. muon arrivals and their angular distribution (cos^2 law)
  2. detector pulses, noise and threshold detection
  3. coincidence counting, which rejects background
  4. Poisson statistics check on the counts

Run:  python muon_detector.py  (opens the results page in Chrome)
Needs: pip install numpy matplotlib
"""
import base64
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # saves the plot used by the standalone browser report
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------------- settings you can change ----------------
FLUX = 1.0 / 60.0   # sea-level muons per cm^2 per second (about 1 per cm^2 per minute)
SIDE = 10.0         # paddle side length, cm (10 x 10 cm)
GAP = 10.0          # distance between the two paddles, cm
EFF = 0.90          # chance a paddle fires when a muon passes through it
BG = 5.0            # background singles per paddle, counts/s (noise, ambient gamma)
TAU = 100e-9        # coincidence resolving time, seconds
WINDOW = 10         # counting window, seconds
N_WIN = 360         # number of windows (1 hour of data)
THRESHOLD = 0.3     # pulse detection threshold (arbitrary units)


def geometric_acceptance(n=200_000):
    """Fraction of muons through paddle 1 that also cross paddle 2."""
    cos_t = np.maximum(rng.random(n) ** (1 / 3), 1e-9)   # cos^2 law: cos(theta) = u^(1/3)
    sin_t = np.sqrt(1 - cos_t ** 2)
    phi = rng.random(n) * 2 * np.pi
    shift = GAP * sin_t / cos_t                  # sideways drift between the paddles
    dx = np.abs(shift * np.cos(phi))
    dy = np.abs(shift * np.sin(phi))
    overlap = np.clip(1 - dx / SIDE, 0, None) * np.clip(1 - dy / SIDE, 0, None)
    return overlap.mean(), np.degrees(np.arccos(cos_t))


def pulse_demo(rate, seconds=4.0, fs=50_000):
    """Build a noisy detector signal with muon pulses and count threshold crossings."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    impulses = np.zeros(n)
    n_events = rng.poisson(rate * seconds)
    idx = rng.integers(0, n - 500, n_events)
    impulses[idx] = rng.lognormal(0.0, 0.3, n_events)    # pulse heights vary
    k = np.arange(500) / fs
    kernel = np.exp(-k / 100e-6)                          # sharp rise, 100 us decay
    signal = np.convolve(impulses, kernel)[:n]
    signal = signal + rng.normal(0, 0.05, n)              # electronic noise

    above = signal > THRESHOLD
    edges = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    hits, last = 0, -10 ** 9
    for e in edges:
        if e - last > 0.002 * fs:      # 2 ms dead time ignores threshold chatter
            hits += 1
            last = e
    return t, signal, n_events, hits


def poisson_check(counts, name):
    """Poisson counts have variance equal to the mean."""
    m, v = counts.mean(), counts.var(ddof=1)
    print(f"{name:<14} mean = {m:7.2f}   variance = {v:7.2f}   variance/mean = {v / m:5.2f}")
    return m


def write_results_page(image_path, values):
    """Create a self-contained browser page with the simulation results."""
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    cards = "".join(
        f"<article><span>{label}</span><strong>{value}</strong></article>"
        for label, value in values
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Signalsphere | Muon Detector Results</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; padding: 40px 6vw; background: #101827; color: #e8eef7; }}
    main {{ max-width: 1200px; margin: auto; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ color: #a8b5c8; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 28px 0; }}
    article {{ padding: 18px; border: 1px solid #29364a; border-radius: 12px; background: #182337; }}
    article span, article strong {{ display: block; }}
    article span {{ color: #a8b5c8; font-size: .9rem; }}
    article strong {{ margin-top: 8px; font-size: 1.35rem; }}
    img {{ display: block; width: 100%; height: auto; border-radius: 12px; background: white; }}
  </style>
</head>
<body>
  <main>
    <h1>Cosmic Muon Detector</h1>
    <p>Simulation results for a two-paddle coincidence muon telescope.</p>
    <section class="metrics" aria-label="Simulation metrics">{cards}</section>
    <h2>Detector plots</h2>
    <img src="data:image/png;base64,{image_data}" alt="Detector signal, muon angles, coincidence counts, and count rates">
  </main>
</body>
</html>
"""
    report_path = image_path.with_name("signalsphere_results.html")
    report_path.write_text(page, encoding="utf-8")
    site_dir = Path(__file__).resolve().parent / "site"
    site_dir.mkdir(exist_ok=True)
    (site_dir / "index.html").write_text(page, encoding="utf-8")
    return report_path


def open_in_chrome(page_path):
    """Open the results page in Chrome, reporting an error if Chrome is unavailable."""
    chrome_paths = [
        Path(os.environ[name]) / "Google" / "Chrome" / "Application" / "chrome.exe"
        for name in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)")
        if os.environ.get(name)
    ]
    chrome = shutil.which("chrome.exe")
    if chrome:
        chrome_paths.insert(0, Path(chrome))

    for chrome_path in chrome_paths:
        if chrome_path.is_file():
            subprocess.Popen([str(chrome_path), "--new-window", page_path.as_uri()])
            return

    raise FileNotFoundError(
        "Google Chrome was not found. Install Chrome or add chrome.exe to PATH."
    )


def main():
    acc, zenith = geometric_acceptance()
    muon_rate = FLUX * SIDE ** 2                 # muons per second crossing one paddle
    single = EFF * muon_rate + BG                # one paddle on its own
    coinc_true = muon_rate * acc * EFF ** 2      # real muons crossing both paddles
    coinc_acc = 2 * single * single * TAU        # accidental coincidences
    coinc = coinc_true + coinc_acc

    singles = rng.poisson(single * WINDOW, N_WIN)
    coincs = rng.poisson(coinc * WINDOW, N_WIN)

    print("=== Signalsphere muon detector ===")
    print(f"Muons crossing one paddle : {muon_rate:.2f} /s")
    print(f"Geometric acceptance      : {acc:.2f}")
    print(f"Singles rate              : {single:.2f} /s "
          f"({100 * EFF * muon_rate / single:.0f}% are muons)")
    print(f"Coincidence rate          : {coinc:.2f} /s "
          f"({100 * coinc_true / coinc:.1f}% are muons)")
    print()
    print("Poisson check (one count per 10 s window):")
    poisson_check(singles, "Singles")
    m = poisson_check(coincs, "Coincidences")

    t, sig, n_events, hits = pulse_demo(EFF * muon_rate)
    print()
    print(f"Pulse demo: {n_events} muon pulses generated, {hits} detected by the threshold")

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].plot(t, sig, lw=0.7)
    ax[0, 0].axhline(THRESHOLD, color="r", ls="--", label="threshold")
    ax[0, 0].set(title=f"Detector signal: {hits} pulses detected ({n_events} true muons)",
                 xlabel="time (s)", ylabel="amplitude (a.u.)")
    ax[0, 0].legend()

    ax[0, 1].hist(zenith, bins=45, range=(0, 90), color="tab:green")
    ax[0, 1].set(title="Muon zenith angles (cos^2 law x solid angle)",
                 xlabel="angle from vertical (deg)", ylabel="muons")

    ks = np.arange(max(0, int(coincs.min()) - 2), int(coincs.max()) + 3)
    pmf = [math.exp(-m + int(k) * math.log(m) - math.lgamma(int(k) + 1)) for k in ks]
    ax[1, 0].hist(coincs, bins=np.arange(ks[0] - 0.5, ks[-1] + 1.5),
                  density=True, alpha=0.6, label="measured")
    ax[1, 0].plot(ks, pmf, "ro-", label="Poisson prediction")
    ax[1, 0].set(title="Coincidence counts vs Poisson",
                 xlabel=f"counts per {WINDOW} s window", ylabel="probability")
    ax[1, 0].legend()

    minutes = np.arange(N_WIN) * WINDOW / 60
    ax[1, 1].plot(minutes, singles / WINDOW, label="singles (one paddle)", alpha=0.8)
    ax[1, 1].plot(minutes, coincs / WINDOW, label="coincidences (both paddles)")
    ax[1, 1].set(title="Count rate over one hour",
                 xlabel="time (min)", ylabel="counts per second")
    ax[1, 1].set_ylim(0, 11)
    ax[1, 1].legend(loc="upper right")

    fig.tight_layout()
    image_path = Path("signalsphere_results.png").resolve()
    fig.savefig(image_path, dpi=150)
    report_path = write_results_page(image_path, [
        ("Muon rate through one paddle", f"{muon_rate:.2f} /s"),
        ("Geometric acceptance", f"{acc:.2%}"),
        ("Singles rate", f"{single:.2f} /s"),
        ("True coincidence rate", f"{coinc_true:.4f} /s"),
        ("Total coincidence rate", f"{coinc:.4f} /s"),
        ("Mean coincidences per window", f"{m:.2f} / {WINDOW} s"),
        ("Detected pulses", f"{hits} / {n_events} generated"),
    ])
    print("Saved plots to signalsphere_results.png")
    print(f"Opening results page: {report_path}")
    open_in_chrome(report_path)


if __name__ == "__main__":
    main()
