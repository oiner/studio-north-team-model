"""Rolling-origin historical validation of candidate models.

uv run --frozen python experiments/run_backtests.py --data-dir ../customer-economics-lab/data
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from replay import ROOT, grid_and_truth, load, mask, score
import models as M
import main
from constant_hazard_example import predict as constant_hazard_predict

CUTOFFS = {'2024-07': 6, '2025-01': 12, '2025-04': 15, '2025-07': 18, '2025-10': 21}

CANDIDATES = {
    'template_constant_hazard': None,
    'constH+cellcash': ('constant', 'cell'),
    'sBG+cellcash': ('sbg', 'cell'),
    'sBG+personcash': ('sbg', 'person'),
    'person_logit+personcash': ('logit', 'person'),
    'person_gbm+personcash': ('gbm', 'person'),
    'person_logit+cellcash': ('logit', 'cell'),
    'person_logit+personcash_notrend': ('logit', 'person_notrend'),
    'multistate': 'ms',
    'multistate_nobehaviour': 'ms_nob',
    'ens_sBG+multistate': 'ens',
    'FINAL_main_predict': 'final',
    'multistate_cohorttrend': 'ms_ct',
    'multistate_recent12': 'ms_w12',
    'multistate_nob_cohorttrend': 'ms_nob_ct',
}


def run(tables, grid, name, seed=20261116):
    if name == 'template_constant_hazard':
        return constant_hazard_predict(tables, grid.copy(), seed=seed)
    ctx = M.Context(tables, grid)
    if CANDIDATES[name] == 'final':
        return main.predict(tables, grid.copy(), seed=seed)
    if CANDIDATES[name] == 'ms':
        return M.multistate(ctx)
    if CANDIDATES[name] == 'ms_nob':
        return M.multistate(ctx, behaviour=False)
    if CANDIDATES[name] == 'ms_ct':
        return M.multistate(ctx, cohort_trend=True)
    if CANDIDATES[name] == 'ms_w12':
        return M.multistate(ctx, train_window=12)
    if CANDIDATES[name] == 'ms_nob_ct':
        return M.multistate(ctx, behaviour=False, cohort_trend=True)
    if CANDIDATES[name] == 'ens':
        a = M.multistate(ctx)
        b = M.assemble(ctx, M.retention_sbg(ctx), M.revenue_person(ctx))
        b = a[M.CELL + ['calendar_month']].merge(b, on=M.CELL + ['calendar_month'])
        a[M.PRED] = (a[M.PRED].to_numpy() + b[M.PRED].to_numpy()) / 2
        return a
    r, c = CANDIDATES[name]
    surv = {'constant': M.retention_constant, 'sbg': M.retention_sbg,
            'logit': lambda x: M.retention_person(x, 'logit', seed),
            'gbm': lambda x: M.retention_person(x, 'gbm', seed)}[r](ctx)
    cash = {'cell': M.revenue_cell_recent, 'person': M.revenue_person,
            'person_notrend': lambda x: M.revenue_person(x, trend=False)}[c](ctx)
    return M.assemble(ctx, surv, cash)


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'experiments' / 'results')
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    full = load(a.data_dir)
    rows = []
    for label, cut in CUTOFFS.items():
        tables = mask(full, cut)
        grid, truth = grid_and_truth(full, cut)
        for name in (a.only or CANDIDATES):
            t0 = time.time()
            try:
                pred = run(tables, grid, name)
                s = score(pred, truth)
                s.update(cutoff=label, model=name, seconds=round(time.time() - t0, 1),
                         horizon=int(grid.calendar_month.nunique()), cohorts=int(grid.cohort_month.nunique()))
                pred.merge(truth, on=M.CELL + ['calendar_month', 'age_month', 'n_acquired', 'origin_retained']
                           ).to_csv(a.out / f'{label}__{name}.csv', index=False)
            except Exception as e:  # keep failures visible
                s = dict(cutoff=label, model=name, error=repr(e))
            print(json.dumps(s), flush=True)
            rows.append(s)
    res = pd.DataFrame(rows)
    res.to_csv(a.out / 'summary.csv', index=False)
    if 'contribution_wape' in res:
        print(res.pivot(index='model', columns='cutoff', values='contribution_wape').round(3).to_string())


if __name__ == '__main__':
    main_()
