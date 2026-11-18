"""Customer histories rebuilt from the observed tables, mirroring the lab's SQL.

Definitions follow the lab's `sql/01_views.sql`:
- A customer's cohort is the month of their first payment with gross - discount > 0.
- Channel is the credited campaign's channel, falling back to the attribution source_type.
- A customer is "retained" in a calendar month when a period starting that month was paid
  and its renewal was due before the data cutoff.
- Cash is assigned by the calendar month of `occurred_at` (refunds included).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CELL = ['cohort_month', 'channel', 'primary_need']


def month_index(values) -> np.ndarray:
    """Months since 2024-01 for ISO date/timestamp strings or Timestamps."""
    ts = pd.to_datetime(pd.Series(values), utc=False, format='mixed')
    return (ts.dt.year.to_numpy() - 2024) * 12 + ts.dt.month.to_numpy() - 1


def index_to_month(idx) -> list[str]:
    idx = np.asarray(idx, dtype=int)
    return [f'{2024 + i // 12:04d}-{i % 12 + 1:02d}-01' for i in idx]


def customers(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per acquired customer with cohort, channel and need."""
    tx = tables['transactions']
    pay = tx[(tx.transaction_type == 'payment') & ((tx.gross_amount - tx.discount_amount) > 0)]
    first = pay.sort_values(['occurred_at', 'transaction_id']).drop_duplicates('person_id')
    first = first[['person_id', 'occurred_at']].rename(columns={'occurred_at': 'first_paid_at'})
    camp = tables['campaign_spend'].groupby('campaign_id', as_index=False).channel.min()
    attr = tables['acquisition_attribution'].merge(camp, left_on='credited_campaign_id',
                                                   right_on='campaign_id', how='left')
    attr['channel'] = attr.channel.fillna(attr.source_type)
    out = (first.merge(tables['people'][['person_id', 'primary_need']], on='person_id')
           .merge(attr[['person_id', 'channel']], on='person_id'))
    out['cohort_idx'] = month_index(out.first_paid_at)
    out['cohort_month'] = index_to_month(out.cohort_idx)
    return out.reset_index(drop=True)


def person_months(tables: dict[str, pd.DataFrame], cutoff_idx: int) -> pd.DataFrame:
    """Dense person x calendar-month panel from acquisition through cutoff_idx - 1."""
    cust = customers(tables)
    cust = cust[cust.cohort_idx < cutoff_idx]
    per = tables['subscription_periods']
    due = month_index(per.renewal_due_at)
    paid = per[(per.renewal_outcome == 'paid') & (due < cutoff_idx)].copy()
    paid['m'] = month_index(paid.period_start)
    ret = paid[['person_id', 'm']].drop_duplicates().assign(retained=1)
    tier = paid[['person_id', 'm', 'plan_tier']].drop_duplicates(['person_id', 'm'])

    tx = tables['transactions'].copy()
    tx['m'] = month_index(tx.occurred_at)
    tx = tx[tx.m < cutoff_idx]
    tx['net'] = tx.gross_amount - tx.discount_amount - tx.refund_amount
    tx['vc'] = tx.variable_cost
    tx['addon_net'] = np.where(tx.charge_type.eq('add_on'), tx.net, 0.0)
    tx['addon_vc'] = np.where(tx.charge_type.eq('add_on'), tx.vc, 0.0)
    tx['sub_vc'] = np.where(tx.charge_type.eq('subscription') & tx.transaction_type.eq('payment'), tx.vc, 0.0)
    tx['refund'] = tx.refund_amount
    cash = tx.groupby(['person_id', 'm'], as_index=False)[['net', 'vc', 'addon_net', 'addon_vc', 'sub_vc', 'refund']].sum()

    reps = (cutoff_idx - cust.cohort_idx).to_numpy()
    panel = cust.loc[cust.index.repeat(reps)].copy()
    panel['m'] = panel.cohort_idx + panel.groupby('person_id').cumcount()
    panel['age'] = panel.m - panel.cohort_idx
    panel = (panel.merge(ret, on=['person_id', 'm'], how='left')
             .merge(tier, on=['person_id', 'm'], how='left')
             .merge(cash, on=['person_id', 'm'], how='left'))
    panel['retained'] = panel.retained.fillna(0).astype(int)
    for c in ['net', 'vc', 'addon_net', 'addon_vc', 'sub_vc', 'refund']:
        panel[c] = panel[c].fillna(0.0)
    panel['contribution'] = panel.net - panel.vc
    return panel.reset_index(drop=True)


def cell_months(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a person-month panel to the lab's cohort x channel x need x month table."""
    g = panel.groupby(CELL + ['m'], as_index=False).agg(
        n_retained=('retained', 'sum'), net_revenue=('net', 'sum'), contribution=('contribution', 'sum'))
    n = panel.drop_duplicates('person_id').groupby(CELL).size().rename('n_acquired').reset_index()
    return g.merge(n, on=CELL)
