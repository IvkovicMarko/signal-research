"""Experiment 1: how strong must a signal be before this pipeline can see it?

Generates synthetic panels at a range of true information coefficients and sample
lengths, and measures the fraction of runs in which the IC t-statistic clears the
conventional threshold. The output is a power curve.

The question it answers is the one you should ask before every backtest: given the
amount of data I have, what is the weakest effect I could reliably detect? If the
answer is "IC of 0.05" and realistic signals have IC of 0.02, then a null result tells
you nothing, and a positive result is probably noise.

Run: python experiments/01_power_study.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import ic_summary, information_coefficient
from src.synthetic import forward_returns, make_panel

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

TRUE_ICS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
SAMPLE_LENGTHS = [120, 360, 720]
N_ASSETS = 49          # matches the real-data cross-section
N_REPLICATIONS = 300
T_THRESHOLD = 1.96

rows = []
for n_periods in SAMPLE_LENGTHS:
    for true_ic in TRUE_ICS:
        detections = 0
        for replication in range(N_REPLICATIONS):
            panel = make_panel(
                n_assets=N_ASSETS,
                n_periods=n_periods,
                ic=true_ic,
                seed=10_000 * n_periods + 100 * replication + int(true_ic * 1000),
            )
            ic = information_coefficient(panel.signals, forward_returns(panel.returns))
            if abs(ic_summary(ic)["t_stat"]) > T_THRESHOLD:
                detections += 1
        power = detections / N_REPLICATIONS
        rows.append({"n_periods": n_periods, "true_ic": true_ic, "power": power})
        print(f"T={n_periods:4d}  true IC={true_ic:.3f}  power={power:.3f}")

frame = pd.DataFrame(rows)
frame.to_csv(RESULTS / "power_study.csv", index=False)

# The true_ic = 0 column is the false-positive rate and should sit near 5%.
# If it does not, the pipeline is miscalibrated and nothing else here is trustworthy.
print("\nFalse-positive rate at true IC = 0 (should be close to 0.05):")
print(frame[frame["true_ic"] == 0.0][["n_periods", "power"]].to_string(index=False))

plt.figure(figsize=(7, 4.5))
for n_periods in SAMPLE_LENGTHS:
    subset = frame[frame["n_periods"] == n_periods]
    plt.plot(subset["true_ic"], subset["power"], marker="o", label=f"T = {n_periods} months")
plt.axhline(0.80, linestyle="--", linewidth=1, color="grey", label="80% power")
plt.axhline(0.05, linestyle=":", linewidth=1, color="grey", label="5% size")
plt.xlabel("true information coefficient")
plt.ylabel("probability of detection")
plt.title(f"Detection power, {N_ASSETS} assets, {N_REPLICATIONS} replications")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(RESULTS / "power_curve.png", dpi=160)
print(f"\nWrote {RESULTS / 'power_curve.png'}")
