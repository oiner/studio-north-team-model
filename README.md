# Studio North 2026 forecast: tier-aware person-level multi-state model

Submission for the optional Studio North private-holdout challenge (Path B). `main.py` exposes the standard `predict(tables, prediction_grid, *, seed)` function and command from the challenge scaffold. The CLI and output validation are unchanged. The authoritative interface, limits and score are in [`submission-contract.json`](submission-contract.json).

## Run

```text
uv sync --frozen
uv run --frozen python main.py --data-dir INPUTS --prediction-grid prediction_grid.csv --output outputs/predictions.csv --seed 20261116
```

`INPUTS` is the folder with the eight released observed CSVs. The run fits everything from those tables. There are no precomputed artifacts, network calls or GPU. It finishes in about 15 seconds on a laptop CPU, against a 60-minute budget.

## What is forecast, and how it differs from the report's LTV

The lab report's LTV is contribution over each cohort's **first twelve monthly receipts**. This challenge asks for something different: **remaining** contribution in calendar **January–December 2026** for the 24 existing cohorts.

- Every cohort has exactly twelve future months, but at different ages:
  - Jan 2024 cohort: ages 24–35;
  - Dec 2025 cohort: ages 1–12.
- Younger cohorts have a larger forecast share of their trajectory, not more future months.
- An older cohort that has already finished its first twelve receipts can still be worth a lot in 2026.
- CAC is not deducted, because acquisition costs happened before the forecast window.

## Model

**The released reference:** cell-level hierarchical shifted-Beta-Geometric retention plus Gamma revenue. It treats every survivor in a channel/need/cohort cell as interchangeable. It also models revenue per member as a smooth function of age and segment.

**Why change it:** historical data shows that who survives matters for cash.

- **Survivors shift to premium.**
  - Premium contributes about $41 per paid month; standard about $23 (subscription prices are $75 and $50).
  - About 8% of standard members upgrade each month, and no downgrades are observed.
  - Premium members churn less.
  - Together, the premium share among survivors rises from about 44% to about 79% within a year.
- **Add-on spend rises with tenure in both tiers,** from about $4–5 to about $8–11 per member-month.
- **Subscription variable cost declines over calendar time** and rises slightly with tenure.

**Our alternative.** Each member active in December 2025 is followed through three states: **standard**, **premium** and **churned** (absorbing). Expected state occupancy is propagated exactly, with no simulation, and summed to the grid cells.

1. **Churn:** a monthly logistic hazard fit on all observed person-month renewals. Inputs:
   - current tier, log tenure, linear tenure, and a tier × log-tenure interaction;
   - channel and need;
   - recent behaviour, frozen at the origin: core actions in the last 1 and 3 months, lifetime core-action rate, support contacts, add-on spend, refunds, onboarding completion, and the latest ease and value ratings.

   The January renewal uses each member's known scheduled tier.
2. **Upgrades:** a logistic model for standard → premium at renewal, with the same inputs.
3. **Cash per active member-month:**
   - the tier subscription price;
   - plus an add-on curve by tier × tenure, scaled by a shrunken personal multiplier (6 months of prior weight);
   - minus subscription variable cost by tier, with linear calendar and tenure trends fit on the last 12 months;
   - plus add-on variable cost at its observed ratio;
   - plus a pooled residual per member-month for refunds and cash from members who left.
4. **Cohort trend blend.** Later cohorts retain better. A cohort-index term in the churn and upgrade models captures this. It helped at later backtest cutoffs, but over-forecast at the earliest full-horizon cutoff, where cohort and tenure are confounded. The final forecast is therefore a **50/50 average of the model without and with the cohort term**, chosen on historical validation only.

**Why it should extrapolate to ages 24–35.** Hazards fall smoothly with tenure, through log and linear tenure terms. No age-specific parameters exist beyond the data, and both tiers' hazards keep declining. A sensitivity run that freezes all tenure effects at age 23 changes the 2026 total by **−0.2%**. Freezing them at age 17 changes it by **−1.3%**. See `experiments/extrapolation_sensitivity.py`.

**Determinism:** all fits are deterministic (L-BFGS logistic regression and least squares). `seed` is accepted for the interface.

## Historical validation (released observations only)

**Method: rolling origins.**
- At each cutoff, every table is truncated to what was known then, in `experiments/replay.py`.
- Renewals due after the cutoff become `pending` scheduling records, exactly as January 2026 appears in the released data.
- Later transactions, events, feedback and spend are dropped.
- The rebuilt cohort tables match the lab's `cohort_age_summary.csv` exactly (1,200 rows).

**Score:** the challenge's primary metric, contribution WAPE after summing channel/need cells to cohort × calendar month.

| Cutoff | Horizon | Cohorts | **Final model WAPE** | Retention WAPE | Bias | Released-style sBG + per-person cash | Scaffold constant hazard |
|---|---|---|---|---|---|---|---|
| 2024-07 | 12 | 6 | 0.135 | 0.132 | −0.132 | 0.139 | 0.455 |
| **2025-01** | **12** | **12** | **0.058** | 0.099 | +0.027 | 0.059 | 0.379 |
| 2025-04 | 9 | 15 | 0.057 | 0.084 | −0.027 | 0.066 | 0.322 |
| 2025-07 | 6 | 18 | 0.041 | 0.066 | −0.028 | 0.074 | 0.270 |
| 2025-10 | 3 | 21 | 0.037 | 0.047 | −0.034 | 0.065 | 0.207 |

The sBG column is a non-hierarchical channel/need sBG proxy, refit at each cutoff. Its revenue side was upgraded to per-person cash, which already beats cell-average cash. The full released hierarchical NumPyro reference was **not** refit at past cutoffs.

Twelve candidates were screened; [`experiments/RESULTS.md`](experiments/RESULTS.md) has the full table and the reasoning. In summary:

- Person-level cash roughly halves the error of every retention model.
- A simple horizon-specific person hazard under-predicted long-horizon survival.
- Behaviour features help mainly when history is short.

**How closely the backtests match the final task.**
- Only the 2025-01 origin has a full 12-month horizon. It covers 12 cohorts at forecast ages 1–23.
- The final task has 24 cohorts and ages 1–35.
- **No backtest covers ages 24–35.** That rests on the extrapolation argument and the sensitivity run above.
- The 2024-07 origin has only six young cohorts, and every model is weak there.
- The final model still leans slightly toward under-forecasting, by −3% at recent origins.

**Context, not accuracy.** On the released data, the 2026 total contribution forecast is about $1.04M. The released sBG reference forecasts $1.01M. Our model puts relatively more value in the 2024 cohorts, which have more premium members, and slightly less in the newest cohorts. Validity of format is not private-holdout accuracy.

## Reproducibility

- **Locked environment:** Python 3.12; `pyproject.toml` plus `uv.lock` (pandas, numpy, scikit-learn, scipy).
- **Required files:** `main.py`, `models.py`, `history.py`, `prediction_grid.csv`, `submission-contract.json`.
- **Validation code** in `experiments/` (not needed for prediction):
  - `uv run --frozen python experiments/run_backtests.py --data-dir <released data>`, run with `experiments/` and the repo root on `PYTHONPATH`.
  - `experiments/constant_hazard_example.py` keeps the scaffold's original example for comparison.
- **Saved outputs** (records only; prediction never reads them):
  - [`evidence/predictions.csv`](evidence/predictions.csv): the submitted forecast, seed 20261116, SHA-256 `7a0ec04a…`;
  - [`evidence/fitted_parameters.json`](evidence/fitted_parameters.json): all fitted coefficients and rates for both blend components, plus input hashes and the seed.
- **Runtime evidence:** see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). It records two clean locked runs with identical prediction bytes, receipts, and the instructor validator result.

No private outcomes, credentials, transcripts or data copies are in this repository. `evidence/fitted_parameters.json` is a record of the fit, not an input: `main.py` refits from the supplied tables on every run.
