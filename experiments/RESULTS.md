# Quick model screen: rolling-origin backtests (observed data only)

**What was done.** At each historical cutoff, every table was truncated to what was known then. Renewals due after the cutoff became "pending", as January 2026 is in the released data. Each model was fit on the truncated data and scored against the realized months that followed, using the challenge metric: contribution WAPE after summing channel/need cells to cohort × calendar month.

**Checks on the rebuild.** The rebuilt cohort tables match the lab's `cohort_age_summary.csv` exactly: 1,200 rows, zero differences. The rebuilt origin survivors match the public grid.

**Code.**
- `history.py`: the lab's definitions, rebuilt from the observed tables.
- `experiments/replay.py`: cutoff masking, truth and scoring.
- `models.py`: the candidate models.
- `experiments/run_backtests.py`: runs the backtests.

**Cutoffs.**

| Cutoff | Horizon | Cohorts |
|---|---|---|
| 2024-07 | 12 months | 6 |
| 2025-01 | 12 months | 12 (the main origin) |
| 2025-04 | 9 months | 15 |
| 2025-07 | 6 months | 18 |
| 2025-10 | 3 months | 21 |

## Contribution WAPE (lower is better)

| Model | 2024-07 | 2025-01 | 2025-04 | 2025-07 | 2025-10 |
|---|---|---|---|---|---|
| Template constant hazard (example) | 0.455 | 0.379 | 0.322 | 0.270 | 0.207 |
| Constant hazard + cell cash | 0.448 | 0.352 | 0.285 | 0.222 | 0.154 |
| sBG (cell, non-hierarchical) + cell cash | 0.262 | 0.161 | 0.146 | 0.140 | 0.115 |
| Person hazard (logit) + cell cash | 0.227 | 0.212 | 0.186 | 0.145 | 0.116 |
| sBG + person cash (tier mix) | 0.139 | 0.059 | 0.066 | 0.074 | 0.065 |
| Person hazard (logit) + person cash | 0.127 | 0.109 | 0.112 | 0.077 | 0.066 |
| Person hazard (GBM) + person cash | 0.113 | 0.115 | 0.107 | 0.070 | 0.065 |
| **Multi-state** (tier-aware churn + upgrades + add-on curve) | 0.123 | 0.058 | 0.065 | 0.053 | 0.047 |
| Multi-state without behaviour features | 0.184 | 0.054 | 0.067 | 0.053 | 0.048 |
| Multi-state, cohort trend | 0.151 | 0.085 | 0.058 | 0.035 | 0.030 |
| **Multi-state, 50/50 base + cohort trend** | 0.135 | 0.058 | 0.057 | 0.041 | 0.037 |
| Ensemble: sBG + person cash and multi-state | 0.129 | 0.056 | 0.064 | 0.062 | 0.056 |

## What drives accuracy

1. **The cash side matters more than retention.** Moving from cell-average cash to person-level cash roughly halves the error for every retention model.
   - Survivors shift sharply toward premium, from 44% to 79% of active members over 12 months. About 8% of standard members upgrade each month, nobody downgrades, and premium members churn less.
   - Add-on spend rises with tenure in both tiers, from about $4 to $10 per month.
   - Variable cost per subscription is falling over calendar time.
2. **Retention heterogeneity matters.** The simple person-level hazard, fit by forecast horizon, under-predicted long-horizon survival. The sBG and the multi-state model both pass the conditioned-on-tenure retention check.
3. **Behaviour features matter only when history is thin.** These are usage, support and ratings. They help at the 2024-07 cutoff (0.123 versus 0.184) and are neutral after that.
4. **A cohort trend has a real but uncertain effect.**
   - Later cohorts retain better. The base model under-forecasts young cohorts by 3–7% at the 2025-07 and 2025-10 cutoffs.
   - A cohort-index term fixes that, but over-forecasts at 2025-01, where cohort and tenure are confounded with only 12 cohorts.
   - The 50/50 blend is the most stable from 2025-01 onward: 0.058 / 0.057 / 0.041 / 0.037.
5. **Remaining bias:** all candidates still under-forecast by about 2–5%.

## How well the backtests match the final task

- The final task has 24 cohorts, a 12-month horizon, and forecast ages 1–35.
- Only the 2025-01 cutoff has a full 12-month horizon. It covers ages 1–23 and has 12 cohorts.
- **No backtest covers ages 24–35.** That extrapolation rests on the tenure terms (log tenure plus linear tenure) and on the assumption that hazards keep declining with tenure. A sensitivity analysis is still to do.

## Comparison with the released reference (sanity check, not accuracy)

Total 2026 contribution forecasts, using no outcomes:

| Forecast | Total 2026 contribution |
|---|---|
| Released hierarchical sBG + Gamma | $1.008M |
| Multi-state | $1.001M |
| Multi-state with cohort trend | $1.078M |
| Template example | $0.689M |

The multi-state model shifts value toward older cohorts, which are richer in premium members, and away from the newest cohorts. The released sBG, which has a cohort trend, sits close to the cohort-trend variant on young cohorts.
