"""Diagnostic: monthly churn, upgrade and add-on rates by tier and tenure (observed data only)."""
import sys
import pandas as pd, numpy as np
from replay import load
from history import person_months
full = load(sys.argv[1])
P = person_months(full, 24).sort_values(['person_id', 'm'])
P['tier'] = P.plan_tier
g = P.groupby('person_id')
P['next_ret'] = g.retained.shift(-1); P['next_tier'] = g.tier.shift(-1)
A = P[(P.retained == 1) & P.next_ret.notna()].copy()
A['tb'] = pd.cut(A.age, [-1, 0, 2, 5, 11, 17, 23])
A['churn'] = 1 - A.next_ret
A['up'] = ((A.tier == 'standard') & (A.next_tier == 'premium')).astype(float)
A['down'] = ((A.tier == 'premium') & (A.next_tier == 'standard')).astype(float)
print(A.groupby(['tier', 'tb'], observed=True).agg(n=('churn', 'size'), churn=('churn', 'mean'), up=('up', 'mean'), down=('down', 'mean')).round(3).to_string())
R = P[P.retained == 1]
R['tb'] = pd.cut(R.age, [-1, 0, 2, 5, 11, 17, 23])
print(R.groupby(['tier', 'tb'], observed=True)[['addon_net', 'addon_vc']].mean().round(2).to_string())
R['half'] = (R.m // 6)
print(R.groupby(['tier', 'half'])[['addon_net', 'sub_vc']].mean().round(2).unstack(0).to_string())
