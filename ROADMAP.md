# Study guide: understanding this codebase

The code is written and the tests pass. What remains is being able to defend it. This
file maps each module to the specific thing you need to understand and the questions
you will be asked about it.

Work through it in order alongside `PREP_PLAN.md`. Roughly 3 hours a week for 9 weeks.
The order matters: each module uses the previous one.

**The self-test that counts:** open the file, read the function, close the file, and
explain out loud what it does and why. If you cannot, you have not learned it yet.

---

## 1. `src/metrics.py` — measurement and its error bars

**Read the code in this order:** `information_coefficient`, `ic_summary`,
`sharpe_ratio`, `sharpe_standard_error`, `newey_west_long_run_variance`.

**The one fact to internalise.** The standard error of a Sharpe ratio is about
`1/sqrt(T)`. With 120 monthly observations, an annualised Sharpe has a standard error
near 0.32. So "Sharpe 0.6 over ten years" is barely two standard errors from zero.
This single number reframes most backtests you will ever be shown.

**Questions you will be asked:**
- Where does `SE(SR) ≈ sqrt((1 + SR²/2)/T)` come from? (Delta method on the joint
  asymptotic distribution of the sample mean and variance.)
- Why Spearman rather than Pearson for the IC? (Fat-tailed return cross-sections; one
  outlier can otherwise dominate.)
- What does positive autocorrelation in P&L do to a naive Sharpe t-statistic? (Inflates
  it — the effective sample size is smaller than T. This is why Newey-West is here.)
- Why compute the IC within each period and then average, instead of pooling the whole
  panel? (A pooled correlation is dominated by time-series variation in the return
  level, not by the signal's ability to rank assets against each other.)

## 2. `src/synthetic.py` — the ground truth

**The key line** is in `make_panel`:
`r[t+1] = vol * (ic * x[t] + sqrt(1 - ic²) * e[t])`, with `x` and `e` each standardised
across assets. Because both have unit variance and are independent, the cross-sectional
correlation between `x[t]` and `r[t+1]` is exactly `ic`. Verify this algebraically once
by hand; it takes two lines and you will then never be unsure about it.

**Questions:**
- Why does that construction give a correlation of exactly `ic`?
- Why is the first row of `returns` generated separately? (Nothing precedes it, so
  nothing predicts it.)
- Why standardise cross-sectionally within each period rather than over the whole
  panel? (The IC is a within-period quantity.)

## 3. `src/portfolio.py` — from signal to P&L

**The one bug to understand** is the lag in `run_backtest`. Weights formed at the end
of period `t-1` earn the return over period `t`, hence `weights.shift(1)`. Remove the
shift and the backtest becomes spectacular and meaningless. `test_backtest_uses_lagged_weights`
exists solely to catch this.

**Questions:**
- Dollar neutral versus beta neutral: what is the difference and when does it bite?
  (Dollar neutral means weights sum to zero; if your longs have higher beta than your
  shorts you still carry market exposure.)
- A signal with IC 0.05 turning over 12 times a year, versus IC 0.03 turning over
  twice. Which do you want? (You cannot answer without the cost per unit of turnover;
  say so, then work it through.)
- Why rank before weighting? (Robustness to outliers; the real-data experiment does
  this, the synthetic one does not need to.)

## 4. `src/validation.py` — honest out-of-sample

Small file, big idea. Splits are chronological with an embargo gap.

**Questions:**
- Give two distinct reasons random k-fold is invalid on a financial panel. (Trains on
  the future; adjacent observations are near-duplicates, so train and test are not
  independent.)
- How wide should the embargo be for a signal with a 12-month lookback? (At least the
  lookback, so no test observation shares input data with a training observation. The
  real-data experiment uses 36 months.)
- Why does the function yield nothing rather than one degenerate split when the series
  is too short?

## 5. `src/selection.py` — the core

Spend the most time here. This is what the project is *about* and what an interviewer
will want to discuss.

**The central derivation.** For N independent standard normals, `E[max]` grows like
`sqrt(2 ln N)`. Sketch: `P(max ≤ z) = Φ(z)^N`, and with `Φ(z) ≈ 1 - φ(z)/z` for large
z, setting `N(1 - Φ(z)) ≈ 1` gives `z ≈ sqrt(2 ln N)`. Be able to produce this.

Then the consequence: a Sharpe estimated over T periods has dispersion `1/sqrt(T)`, so
the best of N worthless trials has expected Sharpe `≈ sqrt(2 ln N / T)`. Experiment 2
confirms this to within 1%.

**Questions:**
- Derive the `sqrt(2 ln N)` growth.
- False discovery rate versus family-wise error rate: define both, say which you chose
  and why. (FDR via Benjamini-Hochberg; in signal research tolerating a known fraction
  of false positives beats the crippling conservatism of controlling the probability of
  *any* false positive.)
- Why is BH a *step-up* procedure — why reject p-values above their own threshold?
- Why the `+1` in the permutation p-value? (Without it a finite permutation set can
  report p = 0, claiming more certainty than it has.)
- Why shuffle within each cross-section rather than across the whole panel? (Preserves
  the marginal distribution of the signal and the time-series structure of returns;
  destroys only the relationship being tested.)

## 6. The experiments

`01_power_study.py` — check the false-positive rate at true IC = 0 first. It comes out
at 0.047 to 0.063 against a nominal 0.05. Understand why that check has to come before
any other claim.

`02_selection_bias.py` — note that candidates are *nested*: each replication draws 200
once and configuration N uses the first N. Understand why that makes the curve smooth.

`03_real_data.py` — the interesting part is the disagreement between the two null
dispersions. Read that section of the README carefully; being able to argue both sides
is worth more than having a single answer.

---

## The hardest question, and your answer

> *"You tested 34 signals. But you also chose the universe, the cost model, the
> weighting scheme and the holding period. Shouldn't N be much larger than 34?"*

Yes. It should. The honest trial count includes every choice made during research, not
just the ones in the final sweep, and it is essentially unknowable. That is an argument
for treating deflation as a lower bound on the correction rather than an exact
adjustment — and for the discipline of holding out data that is never touched until
the end, which is the only defence that does not depend on counting.

Have this answer ready. It is the question that separates people who ran the procedure
from people who understand it.
