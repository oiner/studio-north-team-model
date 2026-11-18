"""Compare candidate 2026 forecasts with the released sBG reference (no outcomes involved)."""
import sys
import pandas as pd
from replay import load
import models as M
import main
full = load(sys.argv[1]); grid = pd.read_csv('prediction_grid.csv')
ref = pd.read_csv(sys.argv[2])
out = {'released_sBG_reference': ref}
for name, kw in [('multistate', {}), ('multistate_cohorttrend', {'cohort_trend': True}), ('multistate_recent12', {'train_window': 12})]:
    out[name] = M.multistate(M.Context(full, grid), **kw)
out['template'] = main.predict(full, grid.copy(), seed=20261116)
tab = pd.DataFrame({k: v.groupby('cohort_month').predicted_contribution.sum() for k, v in out.items()})
print(tab.round(0).to_string()); print(tab.sum().round(0).to_string())
ret = pd.DataFrame({k: v[v.calendar_month == '2026-12-01'].groupby('cohort_month').predicted_retained.sum() for k, v in out.items()})
print((ret.join(grid.drop_duplicates(M.CELL).groupby('cohort_month').origin_retained.sum())).round(1).iloc[::3].to_string())
