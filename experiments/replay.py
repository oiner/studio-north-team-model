"""Rebuild what was observable at an earlier cutoff, and the realized outcomes after it.

Only used for historical validation. Every table is truncated to information known before
the cutoff; renewals due on/after the cutoff become 'pending' scheduling records, exactly
as the released data treats January 2026.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from history import CELL, index_to_month, month_index, person_months, cell_months  # noqa: E402
import main  # noqa: E402

KEYS = CELL + ['calendar_month', 'age_month']
FULL_CUTOFF = 24  # 2026-01


def load(data_dir: Path) -> dict[str, pd.DataFrame]:
    contract = json.loads((ROOT / 'submission-contract.json').read_text())
    return main.load_tables(Path(data_dir), contract)


def mask(tables: dict[str, pd.DataFrame], cutoff: int) -> dict[str, pd.DataFrame]:
    def before(df, col):
        return df[month_index(df[col]) < cutoff].reset_index(drop=True)

    out = {}
    out['people'] = before(tables['people'], 'record_created_at')
    out['transactions'] = before(tables['transactions'], 'occurred_at')
    out['product_events'] = before(tables['product_events'], 'occurred_at')
    out['feedback'] = before(tables['feedback'], 'submitted_at')
    out['campaign_spend'] = before(tables['campaign_spend'], 'spend_date')
    out['acquisition_attribution'] = before(tables['acquisition_attribution'], 'credited_at')
    out['channel_holdouts'] = before(tables['channel_holdouts'], 'assigned_at')
    per = tables['subscription_periods'].copy()
    due = month_index(per.renewal_due_at)
    known = per[due < cutoff]
    later = per[due >= cutoff].sort_values('renewal_due_at').drop_duplicates('subscription_id').copy()
    # Only subscriptions that already exist get a scheduled next period (the released data
    # contains acquired customers only).
    later = later[later.subscription_id.isin(known.subscription_id)]
    later['renewal_outcome'] = 'pending'
    later['cancellation_effective_at'] = np.nan
    out['subscription_periods'] = pd.concat([known, later]).sort_index().reset_index(drop=True)
    buyers = set(out['acquisition_attribution'].person_id)
    for name in ['people', 'product_events', 'feedback']:
        out[name] = out[name][out[name].person_id.isin(buyers)].reset_index(drop=True)
    for name, df in out.items():
        assert list(df.columns) == list(tables[name].columns), name
    return out


def grid_and_truth(tables: dict[str, pd.DataFrame], cutoff: int, horizon: int = 12):
    """Prediction grid (keys + exposures at cutoff) and realized outcomes from full data."""
    horizon = min(horizon, FULL_CUTOFF - cutoff)
    full = cell_months(person_months(tables, FULL_CUTOFF))
    full = full[full.cohort_month.map(lambda s: month_index([s])[0]) < cutoff]
    origin = full[full.m == cutoff - 1][CELL + ['n_acquired', 'n_retained']].rename(
        columns={'n_retained': 'origin_retained'})
    fut = full[(full.m >= cutoff) & (full.m < cutoff + horizon)].copy()
    fut['calendar_month'] = index_to_month(fut.m)
    fut['age_month'] = fut.m - month_index(fut.cohort_month)
    fut = fut.merge(origin.drop(columns='n_acquired'), on=CELL)
    grid = fut[KEYS + ['n_acquired', 'origin_retained']].sort_values(KEYS).reset_index(drop=True)
    truth = fut[KEYS + ['n_acquired', 'origin_retained', 'n_retained', 'net_revenue', 'contribution']]
    return grid, truth.sort_values(KEYS).reset_index(drop=True)


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    j = pred.merge(truth, on=KEYS, validate='one_to_one')
    cm = j.groupby(['cohort_month', 'calendar_month'])[['predicted_contribution', 'contribution']].sum()
    d = cm.predicted_contribution - cm.contribution
    den = cm.contribution.abs().sum()
    return {'contribution_wape': float(d.abs().sum() / den),
            'bias': float(d.sum() / den),
            'retention_wape': float((j.predicted_retained - j.n_retained).abs().sum() / j.n_retained.sum()),
            'points': len(cm)}
