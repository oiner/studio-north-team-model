"""Candidate retention and cash models for the Studio North 2026 forecast.

Every candidate takes the observed tables and the prediction grid and returns expected
retained members, net revenue and contribution per grid row. Retention and revenue pieces
are separable so validation can show which side drives accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, expit

from history import CELL, month_index, person_months

PRED = ['predicted_retained', 'predicted_net_revenue', 'predicted_contribution']
H = 12


# ----------------------------------------------------------------------------- context
class Context:
    """Everything derived once from the observed tables at the forecast origin."""

    def __init__(self, tables, grid):
        self.tables = tables
        self.grid = grid.copy()
        self.cutoff = int(month_index(grid.calendar_month).min())
        self.grid['k'] = month_index(self.grid.calendar_month) - self.cutoff + 1
        self.panel = person_months(tables, self.cutoff)
        p = self.panel
        self.origin_people = p[(p.m == self.cutoff - 1) & (p.retained == 1)].copy()
        per = tables['subscription_periods'].copy()
        per['m'] = month_index(per.period_start)
        self.periods = per


# ----------------------------------------------------------------------------- retention
def retention_constant(ctx) -> pd.DataFrame:
    """Constant monthly renewal per channel x need (Beta(1,1) smoothed)."""
    p = ctx.panel.sort_values(['person_id', 'm'])
    prev = p.groupby('person_id').retained.shift()
    r = p[(prev == 1) & (p.age > 0)]
    q = r.groupby(['channel', 'primary_need']).retained.agg(['sum', 'count'])
    q = ((q['sum'] + 1) / (q['count'] + 2)).rename('q').reset_index()
    g = ctx.grid.merge(q, on=['channel', 'primary_need'], how='left')
    return g.assign(surv=g.q ** g.k)[CELL + ['k', 'surv']]


def _sbg_survival(a, b, t):
    """sBG survival to tenure t (t renewals survived)."""
    return np.exp(betaln(a, b + t) - betaln(a, b))


def retention_sbg(ctx) -> pd.DataFrame:
    """Shifted-beta-geometric per channel x need, fit by ML on renewal trials by tenure,
    then conditioned on each cohort's tenure at the origin (non-hierarchical proxy)."""
    p = ctx.panel.sort_values(['person_id', 'm'])
    prev = p.groupby('person_id').retained.shift()
    r = p[(prev == 1) & (p.age > 0)]
    trials = r.groupby(['channel', 'primary_need', 'age']).retained.agg(['sum', 'count']).reset_index()
    params = {}
    for key, g in trials.groupby(['channel', 'primary_need']):
        t = g.age.to_numpy(float); s = g['sum'].to_numpy(float); n = g['count'].to_numpy(float)

        def nll(x):
            a, b = np.exp(x)
            cond = np.clip(_sbg_survival(a, b, t) / _sbg_survival(a, b, t - 1), 1e-12, 1 - 1e-12)
            return -(s * np.log(cond) + (n - s) * np.log1p(-cond)).sum()
        params[key] = np.exp(minimize(nll, [0.0, 1.0], method='Nelder-Mead').x)
    g = ctx.grid.copy()
    a = np.array([params[(c, n)][0] for c, n in zip(g.channel, g.primary_need)])
    b = np.array([params[(c, n)][1] for c, n in zip(g.channel, g.primary_need)])
    origin_age = g.age_month.to_numpy() - g.k.to_numpy()
    g['surv'] = _sbg_survival(a, b, g.age_month.to_numpy()) / _sbg_survival(a, b, origin_age)
    return g[CELL + ['k', 'surv']]


# person-level direct-horizon hazard --------------------------------------------------------
def _person_features(ctx) -> pd.DataFrame:
    """Features for every person-month (anchor t), using only information dated <= month t."""
    p = ctx.panel[['person_id', 'm', 'age', 'retained', 'channel', 'primary_need', 'cohort_idx', 'cohort_month',
                   'addon_net', 'refund']].copy()
    ev = ctx.tables['product_events'].copy()
    ev['m'] = month_index(ev.occurred_at)
    cnt = ev.pivot_table(index=['person_id', 'm'], columns='event_type', values='event_id',
                         aggfunc='count', fill_value=0).reset_index()
    for c in ['core_action', 'support_contact', 'onboarding_completed']:
        if c not in cnt:
            cnt[c] = 0
    p = p.merge(cnt[['person_id', 'm', 'core_action', 'support_contact', 'onboarding_completed']],
                on=['person_id', 'm'], how='left').fillna({'core_action': 0, 'support_contact': 0,
                                                           'onboarding_completed': 0})
    p = p.sort_values(['person_id', 'm']).reset_index(drop=True)
    gb = p.groupby('person_id')
    roll = lambda col, w: gb[col].transform(lambda s: s.rolling(w, min_periods=1).sum())
    p['core_1'] = p.core_action
    p['core_3'] = roll('core_action', 3)
    p['support_3'] = roll('support_contact', 3)
    p['addon_3'] = roll('addon_net', 3)
    p['refund_3'] = roll('refund', 3)
    p['onboarded'] = gb.onboarding_completed.cummax()
    p['core_cum'] = gb.core_action.cumsum()
    p['core_rate'] = p.core_cum / (p.age + 1)
    # tier of the period starting next month (a scheduled / pending record at anchor t)
    per = ctx.periods[['person_id', 'm', 'plan_tier']].drop_duplicates(['person_id', 'm'])
    per = per.assign(m=per.m - 1).rename(columns={'plan_tier': 'tier_next'})
    p = p.merge(per, on=['person_id', 'm'], how='left')
    cur = ctx.periods[['person_id', 'm', 'plan_tier']].drop_duplicates(['person_id', 'm'])
    p = p.merge(cur, on=['person_id', 'm'], how='left')
    p['tier_next'] = p.tier_next.fillna(p.plan_tier)
    p['premium'] = p.tier_next.eq('premium').astype(float)
    p['switched'] = (p.plan_tier.notna() & p.tier_next.ne(p.plan_tier)).astype(float)
    p['n_switch'] = p.groupby('person_id').switched.cumsum()
    fb = ctx.tables['feedback'].copy()
    fb['m'] = month_index(fb.submitted_at)
    fb = fb.sort_values('submitted_at').groupby(['person_id', 'm'])[['value_rating', 'ease_rating']].last().reset_index()
    p = p.merge(fb, on=['person_id', 'm'], how='left')
    p[['value_rating', 'ease_rating']] = p.groupby('person_id')[['value_rating', 'ease_rating']].ffill()
    p['has_rating'] = p.value_rating.notna().astype(float)
    p['value_rating'] = p.value_rating.astype(float).fillna(0)
    p['ease_rating'] = p.ease_rating.astype(float).fillna(0)
    return p


BASE = ['premium', 'core_1', 'core_3', 'core_rate', 'support_3', 'addon_3', 'refund_3', 'onboarded',
        'n_switch', 'value_rating', 'ease_rating', 'has_rating']


def _design(f: pd.DataFrame, k: np.ndarray, cats) -> np.ndarray:
    ten = f.age.to_numpy(float) + k
    cols = [f[c].to_numpy(float) for c in BASE]
    cols += [np.log1p(ten), ten / 12, np.log(k), (k == 1).astype(float)]
    cols += [np.log1p(f.core_3.to_numpy(float)) * np.log(k), f.premium.to_numpy(float) * np.log1p(ten)]
    for name, levels in cats.items():
        cols += [(f[name] == lv).to_numpy(float) for lv in levels[1:]]
    return np.column_stack(cols)


def _hazard_training(ctx, feats):
    alive = feats.set_index(['person_id', 'm']).retained
    rows = []
    anchors = feats[feats.retained == 1]
    for k in range(1, H + 1):
        a = anchors[anchors.m + k < ctx.cutoff]
        prev = alive.reindex(list(zip(a.person_id, a.m + k - 1))).to_numpy()
        now = alive.reindex(list(zip(a.person_id, a.m + k))).to_numpy()
        keep = prev == 1
        rows.append(a[keep].assign(k=k, y=now[keep]))
    return pd.concat(rows, ignore_index=True)


def retention_person(ctx, learner='logit', seed=0) -> pd.DataFrame:
    feats = _person_features(ctx)
    cats = {'channel': sorted(feats.channel.unique()), 'primary_need': sorted(feats.primary_need.unique())}
    train = _hazard_training(ctx, feats)
    X = _design(train, train.k.to_numpy(float), cats)
    y = train.y.to_numpy()
    if learner == 'logit':
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                               min_samples_leaf=100, l2_regularization=1.0,
                                               early_stopping=False, random_state=seed)
    model.fit(X, y)
    origin = feats[(feats.m == ctx.cutoff - 1) & (feats.retained == 1)]
    out = []
    surv = np.ones(len(origin))
    for k in range(1, H + 1):
        q = model.predict_proba(_design(origin, np.full(len(origin), float(k)), cats))[:, 1]
        surv = surv * q
        out.append(origin[['person_id']].assign(k=k, surv=surv.copy()))
    ctx.model = model
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- cash
def revenue_cell_recent(ctx, months=3) -> pd.DataFrame:
    """Flat cash per retained member: each cell's last `months` of cash / retained-months."""
    p = ctx.panel[ctx.panel.m >= ctx.cutoff - months]
    g = p.groupby(CELL)[['net', 'contribution', 'retained']].sum()
    pooled = g.groupby(level=['channel', 'primary_need']).sum()
    rate = g[['net', 'contribution']].div(g.retained.where(g.retained > 0), axis=0)
    fallback = pooled[['net', 'contribution']].div(pooled.retained, axis=0)
    rate = rate.fillna(fallback.reindex(rate.index.droplevel(0)).set_axis(rate.index))
    return rate.reset_index().rename(columns={'net': 'net_pm', 'contribution': 'con_pm'})


def revenue_person(ctx, trend=True, window=12, shrink=6.0) -> pd.DataFrame:
    """Per-person expected cash per retained month for k = 1..H.

    Subscription price by tier (with Markov tier switching from the known next-period tier),
    subscription variable cost by tier with calendar and tenure trends, a shrunken
    person-level add-on rate, and a pooled residual for refunds and other cash.
    """
    p = ctx.panel
    rec = p[(p.m >= ctx.cutoff - window) & (p.retained == 1)]
    # subscription net & variable cost per tier
    tx = ctx.tables['transactions']
    sub = tx[(tx.charge_type == 'subscription') & (tx.transaction_type == 'payment')].merge(
        ctx.periods[['period_id', 'plan_tier', 'm']], on='period_id')
    sub = sub[(sub.m >= ctx.cutoff - window) & (sub.m < ctx.cutoff)]
    sub = sub.merge(p[['person_id', 'm', 'age']], on=['person_id', 'm'])
    sub = sub[sub.age > 0]
    tiers = {}
    for tier, g in sub.groupby('plan_tier'):
        net = float((g.gross_amount - g.discount_amount).mean())
        if trend:
            A = np.column_stack([np.ones(len(g)), g.m - ctx.cutoff, g.age])
            coef = np.linalg.lstsq(A, g.variable_cost.to_numpy(float), rcond=None)[0]
        else:
            coef = np.array([float(g.variable_cost.mean()), 0.0, 0.0])
        tiers[tier] = (net, coef)
    # tier switching among consecutive retained months
    per = ctx.periods[(ctx.periods.renewal_outcome == 'paid') & (ctx.periods.m >= ctx.cutoff - window)]
    per = per.sort_values(['person_id', 'm'])
    nxt = per.groupby('person_id').plan_tier.shift(-1)
    pairs = per.assign(nxt=nxt).dropna(subset=['nxt'])
    s2p = float((pairs[pairs.plan_tier == 'standard'].nxt == 'premium').mean())
    p2s = float((pairs[pairs.plan_tier == 'premium'].nxt == 'standard').mean())
    # add-ons (person-level shrinkage) and residual cash per retained month
    ret_m = rec.groupby('person_id').size()
    add_net = rec.groupby('person_id').addon_net.sum()
    add_vc = rec.groupby('person_id').addon_vc.sum()
    g_net = add_net.sum() / ret_m.sum(); g_vc = add_vc.sum() / ret_m.sum()
    win = p[p.m >= ctx.cutoff - window]
    total_ret = win.retained.sum()
    resid_net = (win.net.sum() - (win.net[win.retained == 1] - win.refund[win.retained == 1]).sum()
                 + 0) / total_ret
    # residual = refunds + any cash from members not retained that month, per retained month
    resid_net = (win.net.sum() - (win.net + win.refund)[win.retained == 1].sum()) / total_ret
    resid_vc = (win.vc.sum() - win.vc[win.retained == 1].sum()) / total_ret

    o = ctx.origin_people[['person_id', 'age'] + CELL].copy()
    nxt_tier = ctx.periods[ctx.periods.m == ctx.cutoff][['person_id', 'plan_tier']].drop_duplicates('person_id')
    o = o.merge(nxt_tier, on='person_id', how='left')
    o['plan_tier'] = o.plan_tier.fillna('standard')
    n = o.person_id.map(ret_m).fillna(0).to_numpy()
    o['add_net'] = (o.person_id.map(add_net).fillna(0).to_numpy() + shrink * g_net) / (n + shrink)
    o['add_vc'] = (o.person_id.map(add_vc).fillna(0).to_numpy() + shrink * g_vc) / (n + shrink)
    prem = o.plan_tier.eq('premium').to_numpy(float)
    out = []
    for k in range(1, H + 1):
        if k > 1:
            prem = prem * (1 - p2s) + (1 - prem) * s2p
        age = o.age.to_numpy(float) + k
        sub_net = prem * tiers['premium'][0] + (1 - prem) * tiers['standard'][0]
        vc = sum(w * (c[0] + c[1] * k + c[2] * age) for w, (_, c) in
                 [(prem, tiers['premium']), (1 - prem, tiers['standard'])])
        net = sub_net + o.add_net.to_numpy() + resid_net
        vct = vc + o.add_vc.to_numpy() + resid_vc
        out.append(o[['person_id'] + CELL].assign(k=k, net_pm=net, con_pm=net - vct))
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- assemble
def assemble(ctx, surv: pd.DataFrame, cash: pd.DataFrame) -> pd.DataFrame:
    """Combine retention and cash; person-level pieces are summed to grid cells."""
    g = ctx.grid
    if 'person_id' in surv and 'person_id' in cash:
        x = surv.merge(cash, on=['person_id', 'k'])
        x = x.assign(r=x.surv, n=x.surv * x.net_pm, c=x.surv * x.con_pm)
        agg = x.groupby(CELL + ['k'])[['r', 'n', 'c']].sum().reset_index()
    elif 'person_id' in cash:  # cell survival x mean person cash
        cm = cash.groupby(CELL + ['k'])[['net_pm', 'con_pm']].mean().reset_index()
        x = surv.merge(cm, on=CELL + ['k'])
        o = g[CELL + ['k', 'origin_retained']]
        x = x.merge(o, on=CELL + ['k'])
        agg = x.assign(r=x.surv * x.origin_retained, n=x.surv * x.origin_retained * x.net_pm,
                       c=x.surv * x.origin_retained * x.con_pm)[CELL + ['k', 'r', 'n', 'c']]
    else:
        if 'person_id' in surv:
            surv = surv.merge(ctx.origin_people[['person_id'] + CELL], on='person_id')
            surv = surv.groupby(CELL + ['k']).surv.sum().reset_index().rename(columns={'surv': 'r'})
        else:
            surv = surv.merge(g[CELL + ['k', 'origin_retained']], on=CELL + ['k'])
            surv = surv.assign(r=surv.surv * surv.origin_retained)
        x = surv.merge(cash, on=CELL)
        agg = x.assign(n=x.r * x.net_pm, c=x.r * x.con_pm)[CELL + ['k', 'r', 'n', 'c']]
    out = g.merge(agg, on=CELL + ['k'], how='left').fillna({'r': 0.0, 'n': 0.0, 'c': 0.0})
    out['predicted_retained'] = np.minimum(out.r, out.origin_retained)
    out['predicted_net_revenue'] = np.maximum(out.n, 0.0)
    out['predicted_contribution'] = np.minimum(np.maximum(out.c, 0.0), out.predicted_net_revenue)
    return out.drop(columns=['k', 'r', 'n', 'c'])


# ============================================================================ multi-state model
# Monthly transitions per active member: churn (absorbing) or renew; a standard member may
# upgrade to premium at renewal (no downgrades are observed). Expected state occupancy is
# propagated exactly; cash depends on tier, tenure and calendar month.
STATE_FEATS = ['core_1', 'core_3', 'core_rate', 'support_3', 'addon_3', 'refund_3', 'onboarded',
               'value_rating', 'ease_rating', 'has_rating']


def _ms_design(f, ten, prem, cats, behaviour=True, cohort_trend=False):
    cols = [prem, np.log1p(ten), ten / 12, prem * np.log1p(ten)]
    if cohort_trend:
        cols += [f.cohort_idx.to_numpy(float) / 12]
    if behaviour:
        cols += [f[c].to_numpy(float) for c in STATE_FEATS]
    for name, levels in cats.items():
        cols += [(f[name] == lv).to_numpy(float) for lv in levels[1:]]
    return np.column_stack(cols)


def _fit_logit(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X, y)


def _addon_model(ctx, shrink=6.0):
    """Add-on net and variable cost per retained month: tier x tenure curve x person effect."""
    p = ctx.panel
    r = p[p.retained == 1].copy()
    prem = r.plan_tier.eq('premium').to_numpy(float)
    A = np.column_stack([np.ones(len(r)), prem, r.age, prem * r.age])
    coef = np.linalg.lstsq(A, r.addon_net.to_numpy(float), rcond=None)[0]
    vc_ratio = r.addon_vc.sum() / max(r.addon_net.sum(), 1e-9)
    r['expected'] = np.maximum(A @ coef, 0.01)
    s = r.groupby('person_id')[['addon_net', 'expected']].sum()
    mean_e = r.expected.mean()
    effect = (s.addon_net + shrink * mean_e) / (s.expected + shrink * mean_e)
    return coef, vc_ratio, effect


def multistate(ctx, behaviour=True, trend=True, window=12, cohort_trend=False, train_window=None, tenure_cap=None):
    feats = _person_features(ctx)
    cats = {'channel': sorted(feats.channel.unique()), 'primary_need': sorted(feats.primary_need.unique())}
    f = feats.sort_values(['person_id', 'm']).reset_index(drop=True)
    gb = f.groupby('person_id')
    f['next_ret'] = gb.retained.shift(-1)
    f['tier_now'] = f.plan_tier
    f['tier_after'] = gb.plan_tier.shift(-1)
    tr = f[(f.retained == 1) & f.next_ret.notna() & f.tier_now.notna()]
    if train_window:
        tr = tr[tr.m >= ctx.cutoff - 1 - train_window]
    prem = tr.tier_now.eq('premium').to_numpy(float)
    churn = _fit_logit(_ms_design(tr, tr.age.to_numpy(float), prem, cats, behaviour, cohort_trend), 1 - tr.next_ret.to_numpy())
    st = tr[(tr.tier_now == 'standard') & (tr.next_ret == 1)]
    upgrade = _fit_logit(_ms_design(st, st.age.to_numpy(float), np.zeros(len(st)), cats, behaviour, cohort_trend),
                         st.tier_after.eq('premium').to_numpy(float))
    # cash pieces
    tx = ctx.tables['transactions']
    sub = tx[(tx.charge_type == 'subscription') & (tx.transaction_type == 'payment')].merge(
        ctx.periods[['period_id', 'plan_tier', 'm']], on='period_id')
    sub = sub[(sub.m >= ctx.cutoff - window) & (sub.m < ctx.cutoff)].merge(
        ctx.panel[['person_id', 'm', 'age']], on=['person_id', 'm'])
    sub = sub[sub.age > 0]
    tiers = {}
    for tier, g in sub.groupby('plan_tier'):
        A = np.column_stack([np.ones(len(g)), g.m - ctx.cutoff, g.age]) if trend else np.ones((len(g), 1))
        c = np.linalg.lstsq(A, g.variable_cost.to_numpy(float), rcond=None)[0]
        tiers[tier] = (float((g.gross_amount - g.discount_amount).mean()), np.pad(c, (0, 3 - len(c))))
    add_coef, add_vc_ratio, effect = _addon_model(ctx)
    win = ctx.panel[ctx.panel.m >= ctx.cutoff - window]
    ret_n = win.retained.sum()
    resid_net = (win.net.sum() - (win.net + win.refund)[win.retained == 1].sum()) / ret_n
    resid_vc = (win.vc.sum() - win.vc[win.retained == 1].sum()) / ret_n

    o = f[(f.m == ctx.cutoff - 1) & (f.retained == 1)].copy()
    pend = ctx.periods[ctx.periods.m == ctx.cutoff][['person_id', 'plan_tier']].drop_duplicates('person_id')
    o = o.merge(pend.rename(columns={'plan_tier': 'pending_tier'}), on='person_id', how='left')
    o['pending_tier'] = o.pending_tier.fillna(o.plan_tier)
    eff = o.person_id.map(effect).fillna(1.0).to_numpy()
    ten0 = o.age.to_numpy(float)
    cur_prem = o.plan_tier.eq('premium').to_numpy(float)
    # k = 1: renewal decided under current tier; the scheduled tier for the new period is known
    alive_p_prev, alive_s_prev = cur_prem, 1 - cur_prem
    out = []
    for k in range(1, H + 1):
        ten_prev = ten0 + k - 1
        if tenure_cap is not None:  # sensitivity only: freeze tenure effects beyond the cap
            ten_prev = np.minimum(ten_prev, tenure_cap)
        hp = churn.predict_proba(_ms_design(o, ten_prev, np.ones(len(o)), cats, behaviour, cohort_trend))[:, 1]
        hs = churn.predict_proba(_ms_design(o, ten_prev, np.zeros(len(o)), cats, behaviour, cohort_trend))[:, 1]
        surv_p, surv_s = alive_p_prev * (1 - hp), alive_s_prev * (1 - hs)
        if k == 1:
            to_p = o.pending_tier.eq('premium').to_numpy(float)
            alive_p = (surv_p + surv_s) * to_p
            alive_s = (surv_p + surv_s) * (1 - to_p)
        else:
            u = upgrade.predict_proba(_ms_design(o, ten_prev, np.zeros(len(o)), cats, behaviour, cohort_trend))[:, 1]
            alive_p, alive_s = surv_p + surv_s * u, surv_s * (1 - u)
        ten = ten0 + k if tenure_cap is None else np.minimum(ten0 + k, tenure_cap)
        cash = {}
        for name, w in [('premium', alive_p), ('standard', alive_s)]:
            net_sub, c = tiers[name]
            pr = 1.0 if name == 'premium' else 0.0
            addon = np.maximum(add_coef[0] + add_coef[1] * pr + (add_coef[2] + add_coef[3] * pr) * ten, 0) * eff
            vc = c[0] + c[1] * k + c[2] * ten + add_vc_ratio * addon + resid_vc
            net = net_sub + addon + resid_net
            cash[name] = (w * net, w * (net - vc))
        alive = alive_p + alive_s
        out.append(o[['person_id'] + CELL].assign(
            k=k, r=alive, n=cash['premium'][0] + cash['standard'][0], c=cash['premium'][1] + cash['standard'][1]))
        alive_p_prev, alive_s_prev = alive_p, alive_s
    agg = pd.concat(out).groupby(CELL + ['k'])[['r', 'n', 'c']].sum().reset_index()
    g = ctx.grid.merge(agg, on=CELL + ['k'], how='left').fillna({'r': 0.0, 'n': 0.0, 'c': 0.0})
    g['predicted_retained'] = g.r
    g['predicted_net_revenue'] = g.n
    g['predicted_contribution'] = g.c
    ctx.multistate = {'churn': churn, 'upgrade': upgrade, 'tiers': tiers, 'addon': add_coef,
                      'addon_vc_ratio': float(add_vc_ratio), 'resid_net': float(resid_net),
                      'resid_vc': float(resid_vc), 'cats': cats, 'behaviour': behaviour,
                      'cohort_trend': cohort_trend, 'churn_rows': int(len(tr)), 'upgrade_rows': int(len(st)),
                      'addon_effect': effect}
    return g.drop(columns=['k', 'r', 'n', 'c'])
