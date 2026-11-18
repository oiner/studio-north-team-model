"""The scaffold's original constant-hazard interface example (kept for comparison only).

NOT the team model and NOT the released sBG reference.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

KEYS = ['cohort_month','channel','primary_need','calendar_month','age_month']
EXPOSURES = ['n_acquired','origin_retained']
PREDICTIONS = ['predicted_retained','predicted_net_revenue','predicted_contribution']


def predict(tables: dict[str,pd.DataFrame], prediction_grid: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Use only the supplied observed tables; return one forecast per grid row.

    Example assumptions: constant monthly retention within channel/need;
    stationary cash contribution per paid month; no customer heterogeneity.
    seed is accepted for the common interface; this example is deterministic.
    """
    grid = prediction_grid.copy(deep=True)
    origin = pd.Period(grid.calendar_month.min(),freq='M') - 1
    people = tables['people'][['person_id','primary_need']]
    campaigns = tables['campaign_spend'][['campaign_id','channel']].drop_duplicates()
    if campaigns.campaign_id.duplicated().any():
        raise ValueError('A campaign has conflicting channels')
    acquisition = tables['acquisition_attribution'].merge(people,on='person_id',validate='one_to_one')
    acquisition = acquisition.merge(campaigns,left_on='credited_campaign_id',right_on='campaign_id',how='left',validate='many_to_one')
    acquisition['channel'] = acquisition.channel.fillna(acquisition.source_type)
    subjects = acquisition[['person_id','channel','primary_need']]
    periods = tables['subscription_periods'].merge(subjects,on='person_id',validate='many_to_one')
    periods['month'] = pd.to_datetime(periods.period_start).dt.to_period('M')
    periods = periods[(periods.month<=origin) & periods.renewal_outcome.ne('pending')].copy()
    first = periods.loc[periods.renewal_outcome.eq('paid')].groupby('person_id').month.min()
    periods['first_paid_month'] = periods.person_id.map(first)
    renewals = periods[periods.month>periods.first_paid_month].copy()
    renewals['success'] = renewals.renewal_outcome.eq('paid').astype(int)
    segments = ['channel','primary_need']
    renewal_rates = renewals.groupby(segments).success.agg(['sum','count'])
    # Beta(1,1) smoothing is an explicit example choice, not private knowledge.
    renewal_rates['q'] = (renewal_rates['sum']+1)/(renewal_rates['count']+2)
    paid_months = periods[periods.renewal_outcome.eq('paid')].groupby(segments).size()
    cash = tables['transactions'].merge(subjects,on='person_id',validate='many_to_one').copy()
    if (pd.to_datetime(cash.occurred_at).dt.to_period('M')>origin).any():
        raise ValueError('Transactions after the forecast origin were supplied')
    cash['net'] = cash.gross_amount-cash.discount_amount-cash.refund_amount
    cash['contribution'] = cash.net-cash.variable_cost
    rates = cash.groupby(segments)[['net','contribution']].sum().div(paid_months,axis=0)
    result = []
    for row in grid.itertuples(index=False):
        segment = (row.channel,row.primary_need)
        if segment not in rates.index or segment not in renewal_rates.index:
            raise ValueError('Example model needs observed paid cycles and renewals in each segment')
        q = float(renewal_rates.loc[segment,'q'])
        ahead = pd.Period(row.calendar_month,freq='M').ordinal-origin.ordinal
        retained = float(row.origin_retained)*q**ahead
        net = max(0.0,float(rates.loc[segment,'net']))
        contribution = min(net,max(0.0,float(rates.loc[segment,'contribution'])))
        result.append((retained,retained*net,retained*contribution))
    grid[PREDICTIONS] = pd.DataFrame(result,index=grid.index,columns=PREDICTIONS)
    return grid[KEYS+EXPOSURES+PREDICTIONS]


