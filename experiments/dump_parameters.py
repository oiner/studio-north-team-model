"""Write the fitted parameters of the final model (both blend components) to JSON.

uv run --frozen python experiments/dump_parameters.py --data-dir INPUTS --output evidence/fitted_parameters.json
Refits exactly what main.predict fits; prediction never reads this file.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main  # noqa: E402
import models as M  # noqa: E402


def names(cats, behaviour, cohort_trend):
    n = ['premium', 'log1p_tenure', 'tenure_div_12', 'premium_x_log1p_tenure']
    if cohort_trend:
        n += ['cohort_index_div_12']
    if behaviour:
        n += list(M.STATE_FEATS)
    for col, levels in cats.items():
        n += [f'{col}={lv}' for lv in levels[1:]]
    return n


def logit(pipe, feature_names, rows):
    scaler, lr = pipe.steps[0][1], pipe.steps[1][1]
    coef_std = lr.coef_[0]
    raw = coef_std / scaler.scale_
    return {'training_rows': rows, 'regularization_C': float(lr.C),
            'intercept_standardized': float(lr.intercept_[0]),
            'intercept_raw_units': float(lr.intercept_[0] - (raw * scaler.mean_).sum()),
            'coefficients': {f: {'standardized': float(cs), 'raw_units': float(r),
                                 'feature_mean': float(m), 'feature_sd': float(sd)}
                             for f, cs, r, m, sd in zip(feature_names, coef_std, raw, scaler.mean_, scaler.scale_)},
            'reference_levels': 'first sorted level of channel and primary_need'}


ap = argparse.ArgumentParser()
ap.add_argument('--data-dir', type=Path, required=True)
ap.add_argument('--prediction-grid', type=Path, default=ROOT / 'prediction_grid.csv')
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()
contract = json.loads((ROOT / 'submission-contract.json').read_text())
tables = main.load_tables(a.data_dir, contract)
grid = pd.read_csv(a.prediction_grid)
out = {'model': 'Tier-aware person-level multi-state; final forecast = 50/50 average of the two components below',
       'seed': 20261116, 'deterministic': True,
       'input_sha256': {n: hashlib.sha256((a.data_dir / (n + '.csv')).read_bytes()).hexdigest() for n in tables},
       'components': {}}
for label, ct in [('base', False), ('cohort_trend', True)]:
    ctx = M.Context(tables, grid)
    M.multistate(ctx, cohort_trend=ct)
    f = ctx.multistate
    fn = names(f['cats'], f['behaviour'], ct)
    eff = f['addon_effect']
    out['components'][label] = {
        'churn_logit (P[churn at next renewal])': logit(f['churn'], fn, f['churn_rows']),
        'upgrade_logit (P[standard -> premium at renewal])': logit(f['upgrade'], fn, f['upgrade_rows']),
        'subscription_by_tier': {t: {'net_price_per_month': v[0],
                                     'variable_cost': {'intercept_at_origin': float(v[1][0]),
                                                       'per_month_ahead': float(v[1][1]),
                                                       'per_month_tenure': float(v[1][2])}}
                                 for t, v in f['tiers'].items()},
        'addon_net_per_member_month': {'intercept': float(f['addon'][0]), 'premium_shift': float(f['addon'][1]),
                                       'per_month_tenure': float(f['addon'][2]),
                                       'premium_x_tenure': float(f['addon'][3]),
                                       'person_multiplier_shrinkage_months': 6.0,
                                       'person_multiplier_summary': {k: float(v) for k, v in eff.describe().items()}},
        'addon_variable_cost_ratio': f['addon_vc_ratio'],
        'residual_net_per_member_month (refunds, cash from leavers)': f['resid_net'],
        'residual_variable_cost_per_member_month': f['resid_vc'],
    }
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'written', 'output': str(a.output)}))
