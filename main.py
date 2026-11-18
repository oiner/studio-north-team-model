"""Studio North 2026 forecast: tier-aware person-level multi-state model.

An alternative to the released hierarchical sBG + Gamma reference. Standard CLI and
output validation from the challenge scaffold are unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models  # noqa: E402

KEYS = ['cohort_month','channel','primary_need','calendar_month','age_month']
EXPOSURES = ['n_acquired','origin_retained']
PREDICTIONS = ['predicted_retained','predicted_net_revenue','predicted_contribution']


def predict(tables: dict[str,pd.DataFrame], prediction_grid: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Tier-aware person-level multi-state forecast (see README).

    Each member active at the origin moves monthly between standard, premium and churned.
    Churn depends on tier, tenure, channel, need and recent behaviour; standard members may
    upgrade at renewal. Cash per member depends on tier, tenure and calendar month. The final
    forecast is a 50/50 average of the model without and with a cohort-index churn term.
    Uses only the supplied observed tables. All fits are deterministic; `seed` is accepted
    for the common interface.
    """
    grid = prediction_grid.copy(deep=True)
    ctx = models.Context(tables, grid)
    base = models.multistate(ctx, cohort_trend=False)
    trend = models.multistate(ctx, cohort_trend=True)
    base = base.sort_values(KEYS).reset_index(drop=True)
    trend = trend.sort_values(KEYS).reset_index(drop=True)
    if not base[KEYS].equals(trend[KEYS]):
        raise ValueError('Component forecasts do not share the same grid')
    out = base[KEYS+EXPOSURES].copy()
    out[PREDICTIONS] = (base[PREDICTIONS].to_numpy(float)+trend[PREDICTIONS].to_numpy(float))/2
    return out


def load_tables(data_dir: Path, contract: dict) -> dict[str,pd.DataFrame]:
    expected = contract['input_tables']
    if {p.name for p in data_dir.glob('*.csv')} != {name+'.csv' for name in expected}:
        raise ValueError('Supply exactly the eight permitted observed CSVs')
    tables = {}
    for name, fields in expected.items():
        dtypes = {field:('string' if kind in {'VARCHAR','DATE','TIMESTAMP'} else
                         'Int64' if kind=='INTEGER' else 'float64') for field,kind in fields.items()}
        table = pd.read_csv(data_dir/(name+'.csv'),dtype=dtypes)
        if list(table.columns)!=list(fields):
            raise ValueError('Unexpected source schema: '+name)
        tables[name] = table
    return tables


def validate_output(predictions: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(predictions,pd.DataFrame) or set(predictions)!=set(KEYS+EXPOSURES+PREDICTIONS):
        raise ValueError('Return exactly the required grid and prediction columns')
    if predictions.duplicated(KEYS).any():
        raise ValueError('Duplicate prediction keys')
    ordered = predictions.sort_values(KEYS).reset_index(drop=True)
    expected = grid.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered[KEYS+EXPOSURES],expected[KEYS+EXPOSURES],check_dtype=False)
    values = ordered[PREDICTIONS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Predictions must be finite and nonnegative')
    if (ordered.predicted_retained>ordered.origin_retained+1e-8).any():
        raise ValueError('Predicted retained count exceeds origin survivors')
    if (ordered.groupby(KEYS[:3]).predicted_retained.diff().dropna()>1e-8).any():
        raise ValueError('Expected membership must not increase')
    if ((ordered.predicted_retained==0) & ((ordered.predicted_net_revenue>0)|(ordered.predicted_contribution>0))).any():
        raise ValueError('Zero expected members must have zero expected receipts')
    if (ordered.predicted_contribution>ordered.predicted_net_revenue+1e-8).any():
        raise ValueError('Contribution exceeds net revenue')
    return ordered[KEYS+EXPOSURES+PREDICTIONS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--prediction-grid',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261116)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.receipt.json').exists():
        parser.error('Use a new output path; existing forecasts are preserved')
    contract_path = Path(__file__).with_name('submission-contract.json')
    contract = json.loads(contract_path.read_text())
    tables = load_tables(args.data_dir,contract)
    grid = pd.read_csv(args.prediction_grid)
    started = time.monotonic()
    predictions = validate_output(predict(tables,grid.copy(deep=True),seed=args.seed),grid)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(args.output,index=False,float_format='%.10f',lineterminator='\n')
    sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt = {'model':'Tier-aware person-level multi-state (50/50 base + cohort-trend churn); not the released sBG reference',
               'rows':len(predictions),'seed':args.seed,'elapsed_seconds':time.monotonic()-started,
               'source_sha256':sha(__file__),'models_sha256':sha(Path(__file__).with_name('models.py')),'history_sha256':sha(Path(__file__).with_name('history.py')),'uv_lock_sha256':sha(Path(__file__).with_name('uv.lock')),
               'contract_sha256':sha(contract_path),'prediction_grid_sha256':sha(args.prediction_grid),
               'input_sha256':{name:sha(args.data_dir/(name+'.csv')) for name in tables},
               'prediction_sha256':sha(args.output),'future_outcomes_read':False}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'validated','rows':len(predictions),'output':str(args.output)}))


if __name__ == '__main__':
    main()
