# Signalsphere: Cosmic Muon Detector Simulator

A Python simulation of a two-paddle coincidence muon telescope.

## What it does
- Generates cosmic-ray muons with the cos^2 zenith-angle law
- Models detector pulses, electronic noise and a threshold detector
- Uses coincidence counting (both paddles must fire) to reject background
- Checks that the counts follow Poisson statistics (variance = mean)

## Run it
```
pip install -r requirements.txt
python muon_detector.py
```
It prints the rates, saves the plot as `signalsphere_results.png`, and opens a
self-contained results page (`signalsphere_results.html`) in a new Google Chrome
window.

## Publish the website
Running the simulator also generates `site/index.html`, the static page used by
GitHub Pages. Push the project to the `main` branch to trigger the Pages
deployment workflow. In the repository settings, enable Pages with **GitHub
Actions** as the build and deployment source. After the workflow succeeds, the
site will be available at
`https://stic-bit.github.io/SignalSphere/`.

## Why it matters
Coincidence detection is how real particle and radiation detectors separate true events
from noise. The same ideas are used in cosmic-ray telescopes and nuclear instrumentation.

## Limits
This is a simulation, not a measurement. Rates come from standard sea-level values
(about 1 muon per cm^2 per minute) and a simple detector model.
