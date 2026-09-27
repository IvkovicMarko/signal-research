"""Experiment 3: the same pipeline, unchanged, on real data.

Sweeps the candidate signal family over Ken French industry portfolios and reports
each signal twice: naively, and after correcting for the fact that we tested all of
them. Experiments 1 and 2 established what this pipeline can and cannot see; this is
where that calibration gets spent.

Report whatever comes out. If nothing survives the correction, that is the result and
it is a more informative one than a signal that survives only because it was not
tested properly.

Run: python experiments/03_real_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.data import load_industry_portfolios
from src.metrics import ic_summary, information_coefficient, max_drawdown, sharpe_ratio
from src.portfolio import (
    cross_sectional_rank,
    run_backtest,
    to_dollar_neutral_weights,
    turnover,
)
from src.selection import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
)
from src.signals import build_candidate_family
from src.synthetic import forward_returns
from src.validation import walk_forward_splits

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

COST_PER_UNIT_TURNOVER = 0.0010    # 10bp, a deliberately optimistic placeholder
PERIODS_PER_YEAR = 12

returns = load_industry_portfolios(49)
candidates = build_candidate_family(returns)
n_trials = len(candidates)
print(f"{returns.shape[0]} months, {returns.shape[1]} industries, {n_trials} candidate signals\n")

fwd = forward_returns(returns)

rows = []
pnl_by_name = {}
for name, signal in candidates.items():
    # Rank before weighting: industry momentum has fat cross-sectional tails, and
    # raw z-scores would hand most of the book to one or two extreme industries.
    weights = to_dollar_neutral_weights(cross_sectional_rank(signal))
    warmup = int(signal.notna().any(axis=1).argmax())
    pnl = run_backtest(weights, returns, cost_per_unit_turnover=COST_PER_UNIT_TURNOVER)
    pnl = pnl.iloc[warmup + 1:].dropna()
    pnl_by_name[name] = pnl

    ic = information_coefficient(signal, fwd)
    ic_stats = ic_summary(ic)
    annual_sharpe = sharpe_ratio(pnl, PERIODS_PER_YEAR)
    per_period_sharpe = annual_sharpe / np.sqrt(PERIODS_PER_YEAR)

    rows.append({
        "signal": name,
        "mean_ic": ic_stats["mean"],
        "ic_t_stat": ic_stats["t_stat"],
        "sharpe": annual_sharpe,
        "sharpe_per_period": per_period_sharpe,
        "n_periods": len(pnl),
        "annual_turnover": turnover(weights).mean() * PERIODS_PER_YEAR,
        "max_drawdown": max_drawdown(pnl),
        "skew": float(stats.skew(pnl)),
        "excess_kurtosis": float(stats.kurtosis(pnl)),
        "naive_p_value": 2 * (1 - stats.norm.cdf(abs(ic_stats["t_stat"]))),
    })

frame = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

# Deflation needs an estimate of how widely trial Sharpes would scatter under the
# null, and there are two defensible choices which here disagree. Report both.
#
#   iid      - the theoretical dispersion of a Sharpe estimated over T periods,
#              1/sqrt(T). This is what "could noise alone have produced this?" means.
#   observed - the actual spread of the 34 trial Sharpes. This is the Bailey and
#              Lopez de Prado recommendation, and it is the conservative choice when
#              trials are correlated. But it over-corrects when the family contains a
#              real effect, because then the spread reflects genuine variation between
#              signals rather than sampling noise.
typical_length = float(frame["n_periods"].median())
iid_spread = 1.0 / np.sqrt(typical_length)
observed_spread = float(frame["sharpe_per_period"].std(ddof=1))

thresholds = {
    "iid": expected_max_sharpe_under_null(n_trials, iid_spread),
    "observed": expected_max_sharpe_under_null(n_trials, observed_spread),
}

for label, spread in (("iid", iid_spread), ("observed", observed_spread)):
    frame[f"deflated_{label}"] = [
        deflated_sharpe_ratio(
            observed_sharpe=row.sharpe_per_period,
            n_trials=n_trials,
            n_periods=int(row.n_periods),
            sharpe_std=spread,
            skew=row.skew,
            excess_kurtosis=row.excess_kurtosis,
        )
        for row in frame.itertuples()
    ]
    frame[f"survives_{label}"] = frame[f"deflated_{label}"] > 0.95

frame["survives_bh"] = benjamini_hochberg(frame["naive_p_value"].to_numpy(), alpha=0.05)
frame.to_csv(RESULTS / "real_data_signals.csv", index=False)

print(f"Best in-sample annualised Sharpe: {frame['sharpe'].iloc[0]:.3f} "
      f"({frame['signal'].iloc[0]})")
print(f"Signals passing a naive t > 2 test:   {(frame['ic_t_stat'].abs() > 2).sum()} / {n_trials}")
print(f"Signals surviving Benjamini-Hochberg: {frame['survives_bh'].sum()} / {n_trials}\n")
print("Deflated Sharpe, under two estimates of the null trial dispersion:")
for label in ("iid", "observed"):
    annual = thresholds[label] * np.sqrt(PERIODS_PER_YEAR)
    print(f"  {label:>8} spread: threshold {annual:.3f} annualised, "
          f"{frame[f'survives_{label}'].sum()} / {n_trials} survive")

print()
print(frame[["signal", "mean_ic", "ic_t_stat", "sharpe", "annual_turnover",
             "deflated_iid", "deflated_observed"]].head(10).to_string(index=False))

# Walk-forward check on the single best signal: does selecting in-sample survive
# contact with data that was never used to select it?
best_name = frame["signal"].iloc[0]
best_pnl = pnl_by_name[best_name]
oos_segments = [
    best_pnl.iloc[test] for _, test in walk_forward_splits(
        n_periods=len(best_pnl), n_test=60, embargo=36, min_train=240
    )
]
if oos_segments:
    stitched = pd.concat(oos_segments)
    print(f"\n{best_name}: in-sample Sharpe {frame['sharpe'].iloc[0]:.3f}, "
          f"stitched walk-forward Sharpe {sharpe_ratio(stitched, PERIODS_PER_YEAR):.3f}")

plt.figure(figsize=(8, 4.5))
plt.scatter(frame["annual_turnover"], frame["sharpe"],
            c=frame["survives_iid"].map({True: "tab:green", False: "tab:grey"}), s=28)
for label, style in (("iid", "--"), ("observed", ":")):
    plt.axhline(thresholds[label] * np.sqrt(PERIODS_PER_YEAR), linestyle=style,
                color="tab:red", label=f"best-of-{n_trials} null threshold ({label} spread)")
plt.xlabel("annualised turnover")
plt.ylabel("annualised Sharpe, net of costs")
plt.title("Candidate signals: green clears the multiple-testing correction")
plt.legend(fontsize=8)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(RESULTS / "real_data_signals.png", dpi=160)
print(f"\nWrote {RESULTS / 'real_data_signals.png'}")
