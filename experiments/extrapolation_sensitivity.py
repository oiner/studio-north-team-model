"""Sensitivity of the 2026 forecast to tenure effects beyond the largest observed age (23)."""
import sys
import pandas as pd
from replay import load
import models as M
full = load(sys.argv[1]); grid = pd.read_csv('prediction_grid.csv')
res = {}
for label, cap in [('extrapolate (final)', None), ('freeze tenure at 23', 23), ('freeze tenure at 17', 17)]:
    preds = [M.multistate(M.Context(full, grid), cohort_trend=ct, tenure_cap=cap) for ct in (False, True)]
    p = preds[0].copy(); p[M.PRED] = (preds[0][M.PRED].to_numpy() + preds[1][M.PRED].to_numpy()) / 2
    old = p[p.cohort_month < '2025-01-01']
    res[label] = {'total_2026_contribution': p.predicted_contribution.sum(),
                  '2024_cohorts_contribution': old.predicted_contribution.sum(),
                  'rows_age_24_plus_contribution': p[p.age_month >= 24].predicted_contribution.sum()}
print(pd.DataFrame(res).round(0).to_string())
