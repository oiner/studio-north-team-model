"""Diagnostic: decompose realized cash per retained member after a cutoff vs the cash model."""
import sys
import pandas as pd, numpy as np
from replay import load, mask, grid_and_truth
import models as M
from history import person_months, month_index
full = load(sys.argv[1]); cut = int(sys.argv[2])
P = person_months(full, 24)
per = full['subscription_periods'].copy(); per['m'] = month_index(per.period_start)
tier = per[per.renewal_outcome == 'paid'][['person_id', 'm', 'plan_tier']]
pass
origin = set(P[(P.m == cut - 1) & (P.retained == 1)].person_id)
F = P[P.person_id.isin(origin) & (P.m >= cut) & (P.m < cut + 12)]
R = F[F.retained == 1]
a = R.groupby('m').agg(n=('person_id', 'size'), prem=('plan_tier', lambda s: (s == 'premium').mean()),
                       addon=('addon_net', 'mean'), subvc=('sub_vc', 'mean'), addvc=('addon_vc', 'mean'))
a['resid_net'] = (F.groupby('m').net.sum() - (R.net + R.refund).groupby(R.m).sum()) / a.n
a['net_pm'] = F.groupby('m').net.sum() / a.n
tables = mask(full, cut); grid, _ = grid_and_truth(full, cut)
ctx = M.Context(tables, grid)
cash = M.revenue_person(ctx)
surv = M.retention_person(ctx, 'logit')
x = cash.merge(surv, on=['person_id', 'k'])
o = ctx.periods[ctx.periods.m == cut][['person_id', 'plan_tier']].drop_duplicates('person_id')
print('origin premium share (next tier)', (o[o.person_id.isin(origin)].plan_tier == 'premium').mean())
pw = x.groupby('k').apply(lambda d: np.average(d.net_pm, weights=d.surv))
a['pred_net_pm'] = pw.to_numpy()[:len(a)]
print(a.round(3).to_string())
