"""Experiment 2: the cost of searching. This is the central result of the project.

Every candidate signal here has a true information coefficient of exactly zero. There
is nothing to find. We nonetheless do what a researcher under deadline pressure does:
test N candidates, keep the best in-sample Sharpe ratio, and report it.

Three numbers come out of each configuration:

  * the best in-sample Sharpe, which is what a naive backtest would print;
  * the same signal's out-of-sample Sharpe, which is centred on zero because the
    signal is worthless and the selection was pure luck;
  * the fraction of runs where the deflated Sharpe ratio would still have declared
    the winner significant, which is the false-positive rate after correction.

The analytic prediction sigma*sqrt(2 ln N) is plotted alongside.

Candidates are nested: each replication draws MAX_TRIALS of them once, and the
configuration with N trials uses the first N. That makes the curve smooth across N
and avoids regenerating the same distributions repeatedly.

Run: python experiments/02_selection_bias.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import sharpe_ratio
from src.portfolio import run_backtest, to_dollar_neutral_weights
from src.selection import deflated_sharpe_ratio, expected_max_sharpe_under_null
from src.synthetic import make_null_candidates

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

TRIAL_COUNTS = [1, 2, 5, 10, 25, 50, 100, 200]
MAX_TRIALS = max(TRIAL_COUNTS)
N_ASSETS = 49
N_PERIODS = 720            # 60 years of monthly data, generous by any standard
IN_SAMPLE = 480            # first 40 years, used to select the winner
N_REPLICATIONS = 150
PERIODS_PER_YEAR = 12

records = []
for replication in range(N_REPLICATIONS):
    candidates, returns = make_null_candidates(
        n_assets=N_ASSETS,
        n_periods=N_PERIODS,
        n_candidates=MAX_TRIALS,
        seed=50_000 + replication,
    )

    in_sample, out_of_sample, skews, kurtoses = [], [], [], []
    for candidate in candidates:
        pnl = run_backtest(to_dollar_neutral_weights(candidate), returns)
        in_sample.append(sharpe_ratio(pnl.iloc[:IN_SAMPLE], PERIODS_PER_YEAR))
        out_of_sample.append(sharpe_ratio(pnl.iloc[IN_SAMPLE:], PERIODS_PER_YEAR))
    in_sample = np.asarray(in_sample)
    out_of_sample = np.asarray(out_of_sample)

    for n_trials in TRIAL_COUNTS:
        subset = in_sample[:n_trials]
        winner = int(np.argmax(subset))
        spread = float(np.std(subset, ddof=1)) if n_trials > 1 else 0.0
        records.append({
            "replication": replication,
            "n_trials": n_trials,
            "best_in_sample": subset[winner],
            "winner_out_of_sample": out_of_sample[winner],
            "deflated": deflated_sharpe_ratio(
                observed_sharpe=subset[winner] / np.sqrt(PERIODS_PER_YEAR),
                n_trials=n_trials,
                n_periods=IN_SAMPLE,
                sharpe_std=spread / np.sqrt(PERIODS_PER_YEAR),
            ),
        })

    if (replication + 1) % 25 == 0:
        print(f"  {replication + 1}/{N_REPLICATIONS} replications")

raw = pd.DataFrame(records)

# Sharpe ratios estimated over IN_SAMPLE periods have dispersion ~1/sqrt(T);
# annualised, that is sqrt(12/480).
theoretical_spread = np.sqrt(PERIODS_PER_YEAR / IN_SAMPLE)

summary = (
    raw.groupby("n_trials")
    .agg(
        mean_best_in_sample=("best_in_sample", "mean"),
        mean_winner_out_of_sample=("winner_out_of_sample", "mean"),
        naive_t_stat=("best_in_sample", lambda s: s.mean() * np.sqrt(IN_SAMPLE / PERIODS_PER_YEAR)),
        false_positive_rate=("deflated", lambda s: float((s > 0.95).mean())),
    )
    .reset_index()
)
summary["analytic_threshold"] = [
    expected_max_sharpe_under_null(n, theoretical_spread) for n in summary["n_trials"]
]
summary.to_csv(RESULTS / "selection_bias.csv", index=False)

print(f"\n{N_REPLICATIONS} replications, {N_PERIODS} months, {N_ASSETS} assets, "
      f"true IC = 0 everywhere")
print(f"selection on the first {IN_SAMPLE} months, evaluation on the remaining "
      f"{N_PERIODS - IN_SAMPLE}\n")
print(f"{'N':>5} {'best IS SR':>12} {'predicted':>11} {'naive t':>9} "
      f"{'winner OOS':>12} {'false pos.':>11}")
for row in summary.itertuples():
    print(f"{row.n_trials:5d} {row.mean_best_in_sample:12.3f} "
          f"{row.analytic_threshold:11.3f} {row.naive_t_stat:9.2f} "
          f"{row.mean_winner_out_of_sample:12.3f} {row.false_positive_rate:11.3f}")

plt.figure(figsize=(7.5, 4.5))
plt.plot(summary["n_trials"], summary["mean_best_in_sample"],
         marker="o", label="best in-sample Sharpe (what a naive backtest reports)")
plt.plot(summary["n_trials"], summary["analytic_threshold"],
         linestyle="--", color="tab:red", label=r"prediction: $\sigma\,E[\max_N Z]$")
plt.plot(summary["n_trials"], summary["mean_winner_out_of_sample"],
         marker="s", label="same signal, out of sample (the truth: zero)")
plt.axhline(0.0, color="black", linewidth=0.8)
plt.xscale("log")
plt.xlabel("number of candidate signals tested")
plt.ylabel("annualised Sharpe ratio")
plt.title("Selection bias with no true signal present")
plt.legend(fontsize=8)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(RESULTS / "selection_bias.png", dpi=160)
print(f"\nWrote {RESULTS / 'selection_bias.png'}")
