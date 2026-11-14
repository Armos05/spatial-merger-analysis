"""T1: entry analysis on the twin (Pillar 2 step 3; DESIGN §4.3). Two estimators of how much nearby stores lose when
a discounter opens, by distance band and incumbent group:

Traditional — store-month two-way fixed effects (TWFE):
    ln R_km = a_k + b_m + Σ_bg β_bg × 1[k in band b, group g of its nearest entrant] × 1[month ≥ that entrant's opening]
    Problems: monthly aggregation, already-treated stores act as controls, stores hit by several entrants.

Data-driven — pooled ("stacked") event study (Cengiz et al. 2019 style), the way Keystone pools many store openings:
    for each entrant e build a stack: weeks [open − W, open + W), treated = stores within the bands of e,
    clean controls = stores far from e and from every other entrant opening inside the window.
    ln q_kte = a_ke + b_te + Σ_bg β_bg × 1[k in band b, group g in stack e] × 1[t ≥ open_e]
    Stack-specific store and week effects compare each opening only with its own clean controls.

Pooled exposure regression — every store-week counts the entrants already open in each distance band:
    ln q_kt = a_k + b_t + Σ_bg β_bg × N_kt(b) × 1[k in group g],   N_kt(b) = # entrants open by t within band b of k
    Overlapping openings add up instead of contaminating each other; bands reach 30 km to absorb wide spillovers.
    Limit: week effects absorb what every store suffers, so effects are relative to the farthest band.

All: unweighted OLS after removing fixed effects by alternating projections; cluster-robust standard errors
(by store for TWFE, by entrant for the stacked design). Loss in % = −100 × (exp(β) − 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def demean(X: np.ndarray, groups: list, iters: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Remove several sets of fixed effects (alternating projections)."""
    X = np.array(X, float, copy=True)
    if X.ndim == 1:
        X = X[:, None]
    sizes = [np.bincount(g) for g in groups]
    for _ in range(iters):
        delta = 0.0
        for g, n in zip(groups, sizes):
            m = np.stack([np.bincount(g, X[:, k], len(n)) for k in range(X.shape[1])], 1) / np.maximum(n, 1)[:, None]
            X -= m[g]
            delta = max(delta, float(np.abs(m).max()))
        if delta < tol:
            break
    return X


def ols_cluster(y: np.ndarray, X: np.ndarray, cluster: np.ndarray):
    """OLS on already-demeaned data with cluster-robust (CR1) standard errors."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    G = int(cluster.max()) + 1
    S = np.stack([np.bincount(cluster, X[:, k] * u, G) for k in range(X.shape[1])], 1)
    n, k = X.shape
    Gn = len(np.unique(cluster))
    c = Gn / (Gn - 1) * (n - 1) / (n - k)
    V = c * XtX_inv @ (S.T @ S) @ XtX_inv
    return b, np.sqrt(np.clip(np.diag(V), 0, None))


def _cells(bands, groups):
    return [(b, g) for b in range(len(bands) - 1) for g in groups]


def stacked_event_study(panel, entrants, xy_km, group_of_store, bands=(0, 1, 2, 3, 5, 10), groups=None,
                        window=26, control_min_km=30.0, n_controls=150, exclude=(), seed=0, dynamic_km=3.0,
                        clean_treated=True):
    """panel: store, week, baskets (observed). entrants: DataFrame store, week.
    clean_treated: drop treated stores that are also within the outer band of another opening inside the window
    (otherwise they mix two openings' effects).
    Returns (static table, dynamic table, stacked data size, per-entrant DiD within dynamic_km)."""
    rng = np.random.default_rng(seed)
    groups = groups or sorted(set(group_of_store))
    cells = _cells(bands, groups)
    ent = entrants.store.to_numpy()
    ow = dict(zip(entrants.store, entrants.week))
    panel = panel[~panel.store.isin(set(ent) | set(exclude))]
    T = int(panel.week.max()) + 1
    J = len(xy_km)
    Y = np.full((T, J), np.nan)
    Y[panel.week.to_numpy(), panel.store.to_numpy()] = np.log(panel.baskets.to_numpy())
    usable = np.zeros(J, bool)
    usable[panel.store.unique()] = True
    pieces, per_entrant = [], []
    for s, (e, w0) in enumerate(ow.items()):
        lo, hi = max(0, w0 - window), min(T, w0 + window)
        d = np.hypot(*(xy_km - xy_km[e]).T)
        band = np.digitize(d, bands, right=True) - 1                            # 0 .. len(bands)-2 inside
        treated = usable & (d <= bands[-1]) & (band >= 0)
        others = [o for o in ent if o != e and lo - window <= ow[o] <= hi + window]
        dirty = np.zeros(J, bool)
        exposed = np.zeros(J, bool)
        for o in others:
            do = np.hypot(*(xy_km - xy_km[o]).T)
            dirty |= do <= bands[-1] + 5
            if lo <= ow[o] < hi:
                exposed |= do <= bands[-1]
        if clean_treated:
            treated &= ~exposed
        pool = np.where(usable & (d > control_min_km) & ~dirty)[0]
        ctrl = rng.choice(pool, min(n_controls, len(pool)), replace=False)
        ks = np.r_[np.where(treated)[0], ctrl]
        wk = np.arange(lo, hi)
        kk, tt = np.meshgrid(ks, wk, indexing="ij")
        y = Y[tt, kk]
        ok = ~np.isnan(y)
        is_t = np.r_[np.ones(treated.sum(), bool), np.zeros(len(ctrl), bool)]
        df = pd.DataFrame({"stack": s, "entrant": e, "store": kk[ok], "week": tt[ok], "y": y[ok],
                           "post": (tt[ok] >= w0), "rel": tt[ok] - w0,
                           "treated": np.repeat(is_t, len(wk)).reshape(kk.shape)[ok],
                           "band": np.repeat(np.where(is_t, band[ks], -1), len(wk)).reshape(kk.shape)[ok]})
        pieces.append(df)
        near = is_t & (d[ks] <= dynamic_km)
        if near.any():
            yt, yc = y[near], y[~is_t]
            pre, post = wk < w0, wk >= w0
            did = (np.nanmean(yt[:, post]) - np.nanmean(yt[:, pre])) - (np.nanmean(yc[:, post]) - np.nanmean(yc[:, pre]))
            per_entrant.append({"entrant": e, "near_stores": int(near.sum()), "did_loss_pct": -100 * (np.exp(did) - 1)})
    st = pd.concat(pieces, ignore_index=True)
    st["group"] = np.where(st.treated, np.asarray(group_of_store)[st.store], "control")
    sk = st["stack"].to_numpy().astype(np.int64)
    fe1 = pd.factorize(sk * J + st.store.to_numpy())[0]
    fe2 = pd.factorize(sk * T + st.week.to_numpy())[0]
    X = np.stack([(st.band.to_numpy() == b) & (st.group.to_numpy() == g) & st.post.to_numpy() for b, g in cells], 1)
    Z = demean(np.c_[st.y.to_numpy(), X.astype(float)], [fe1, fe2])
    b, se = ols_cluster(Z[:, 0], Z[:, 1:], sk)
    static = pd.DataFrame([{"band": f"{bands[bi]}-{bands[bi + 1]} km", "group": g, "beta": b[i], "se": se[i],
                            "n_stores": int(st[(st.band == bi) & (st.group == g)].drop_duplicates(["stack", "store"]).shape[0])}
                           for i, (bi, g) in enumerate(cells)])
    # dynamic: stores within dynamic_km (all groups) vs controls, event time −window..window−1, reference −1
    dyn = st[(~st.treated) | (st.band <= np.digitize(dynamic_km, bands, right=True) - 1)]
    rels = [r for r in range(-window, window) if r != -1]
    Xd = np.stack([(dyn.treated & (dyn.rel == r)).to_numpy() for r in rels], 1).astype(float)
    dk = dyn["stack"].to_numpy().astype(np.int64)
    f1 = pd.factorize(dk * J + dyn.store.to_numpy())[0]
    f2 = pd.factorize(dk * T + dyn.week.to_numpy())[0]
    Zd = demean(np.c_[dyn.y.to_numpy(), Xd], [f1, f2])
    bd, sed = ols_cluster(Zd[:, 0], Zd[:, 1:], dk)
    dynamic = pd.DataFrame({"rel_week": rels, "beta": bd, "se": sed})
    return static, dynamic, len(st), pd.DataFrame(per_entrant)


def twfe_monthly(panel, entrants, xy_km, group_of_store, bands=(0, 1, 2, 3, 5, 10), groups=None, exclude=()):
    """Traditional: store-month mean weekly revenue, treatment from the nearest entrant, store + month FE."""
    groups = groups or sorted(set(group_of_store))
    cells = _cells(bands, groups)
    ent = entrants.store.to_numpy()
    p = panel[~panel.store.isin(set(ent) | set(exclude))].copy()
    p["month"] = pd.DatetimeIndex(p.date).to_period("M").astype(str)
    m = p.groupby(["store", "month"], as_index=False).agg(rev=("revenue", "mean"))
    stores = m.store.unique()
    D = np.hypot(*(xy_km[stores][:, None, :] - xy_km[ent][None, :, :]).transpose(2, 0, 1))  # (stores, entrants)
    near = D.argmin(1)
    dist = D.min(1)
    open_month = pd.Series(pd.DatetimeIndex(entrants.date).to_period("M").astype(str).to_numpy(), index=range(len(ent)))
    info = pd.DataFrame({"store": stores, "dist": dist, "band": np.digitize(dist, bands, right=True) - 1,
                         "open_month": open_month.loc[near].to_numpy()})
    m = m.merge(info, on="store")
    m["post"] = m.month >= m.open_month
    m["group"] = np.asarray(group_of_store)[m.store]
    m["band"] = np.where(m.dist <= bands[-1], m.band, -1)
    X = np.stack([(m.band.to_numpy() == b) & (m.group.to_numpy() == g) & m.post.to_numpy() for b, g in cells], 1)
    fe1, fe2 = pd.factorize(m.store)[0], pd.factorize(m.month)[0]
    Z = demean(np.c_[np.log(m.rev.to_numpy()), X.astype(float)], [fe1, fe2])
    b, se = ols_cluster(Z[:, 0], Z[:, 1:], fe1)
    return pd.DataFrame([{"band": f"{bands[bi]}-{bands[bi + 1]} km", "group": g, "beta": b[i], "se": se[i],
                          "n_stores": int(info[(info.band == bi) & (info.dist <= bands[-1])].store
                                          .map(lambda k: group_of_store[k] == g).sum())}
                         for i, (bi, g) in enumerate(cells)])


def to_loss(df):
    """β (log points) → loss % with a 95% interval."""
    out = df.copy()
    out["loss_pct"] = -100 * (np.exp(out.beta) - 1)
    out["lo95"] = -100 * (np.exp(out.beta + 1.96 * out.se) - 1)
    out["hi95"] = -100 * (np.exp(out.beta - 1.96 * out.se) - 1)
    return out


def exposure_regression(panel, entrants, xy_km, group_of_store, bands=(0, 1, 2, 3, 5, 10, 15, 20, 30), groups=None,
                        exclude=(), y="baskets"):
    """Weekly store panel, store + week FE, counts of open entrants per distance band × incumbent group."""
    groups = groups or sorted(set(group_of_store))
    ent, ow = entrants.store.to_numpy(), entrants.week.to_numpy()
    p = panel[~panel.store.isin(set(ent) | set(exclude))]
    k, t = p.store.to_numpy(), p.week.to_numpy()
    T = int(t.max()) + 1
    D = np.hypot(*(xy_km[:, None, :] - xy_km[ent][None, :, :]).transpose(2, 0, 1))      # (J, E)
    band = np.digitize(D, bands, right=True) - 1
    band[D > bands[-1]] = -1
    is_open = (ow[None, :] <= np.arange(T)[:, None]).astype(float)                     # (T, E)
    g_of = np.asarray(group_of_store)[k]
    cols, names = [], []
    for b in range(len(bands) - 1):
        N = is_open @ (band == b).T.astype(float)                                       # (T, J)
        for g in groups:
            cols.append(N[t, k] * (g_of == g))
            names.append((f"{bands[b]}-{bands[b + 1]} km", g))
    X = np.stack(cols, 1)
    Z = demean(np.c_[np.log(p[y].to_numpy()), X], [pd.factorize(k)[0], t])
    b, se = ols_cluster(Z[:, 0], Z[:, 1:], pd.factorize(k)[0])
    return pd.DataFrame({"band": [n[0] for n in names], "group": [n[1] for n in names], "beta": b, "se": se})
