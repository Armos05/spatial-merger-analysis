"""T3: local (store-level) elasticities and local GUPPIs (Pillar 2 step 5; DESIGN §4.3). [sim]

Input: the cross-fitted DML residuals of 2.4 (ỹ = log baskets, d̃ = log price, z̃ = cost index), one row per
store-week, plus store features an analyst can build from public data (no hidden twin quantities).

Five ways to give every store its own elasticity θ_j:
    national        one θ for all stores (DML-PLIV, 2.4)
    banner          one θ per banner (DML-PLIV, 2.4)
    store-by-store  θ_j = Σ_t z̃ ỹ / Σ_t z̃ d̃ using only store j's ~154 weeks (unbiased but noisy)
    linear IV       θ(x) = a + b'x: instruments z̃·[1, x] for d̃·[1, x] (parametric heterogeneity)
    causal forest   generalised random forest for IV (Athey, Tibshirani & Wager 2019): θ(x) solves the local
                    moment Σ_i w_i(x) z̃_i (ỹ_i − θ d̃_i) = 0 with forest weights w_i(x); honest splitting, so
                    intervals are valid.

Local GUPPI (screen, Pillar 1 §1.5), with diversion from the twin's true model and margins from each method:
    GUPPI_j = Σ_{k in the other merging party} D(j→k) × m_k / p_j,     m_k = p_k / |θ_k|   (Lerner, single store)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

GROUPS = {"Migros": "migros", "Migros Partner": "migros", "Migrolino": "migros", "Denner": "migros",
          "Coop": "coop", "Coop Pronto": "coop", "Aldi": "discount", "Lidl": "discount"}


def store_features(xy_store, banner, nest, loc_xy, loc_hh, loc_income, loc_border):
    """Public-data features per store (all distances in km, straight line)."""
    grp = np.array([GROUPS.get(b, "other") for b in banner])
    tree = cKDTree(xy_store)
    f = {}
    for g in ("migros", "coop", "discount", "other"):
        pts = xy_store[grp == g]
        tg = cKDTree(pts)
        for r in (1, 3, 5):
            f[f"n_{g}_{r}km"] = np.array([len(x) for x in tg.query_ball_point(xy_store, r)]) - (grp == g) * 1
        dd, _ = tg.query(xy_store, k=2)
        f[f"dist_{g}"] = np.where(grp == g, dd[:, 1], dd[:, 0])
    f["n_all_3km"] = np.array([len(x) for x in tree.query_ball_point(xy_store, 3)]) - 1
    lt = cKDTree(loc_xy)
    near = lt.query_ball_point(xy_store, 5)
    f["hh_5km"] = np.array([loc_hh[i].sum() for i in near])
    f["income_5km"] = np.array([np.average(loc_income[i], weights=loc_hh[i]) if len(i) else np.nan for i in near])
    f["border_min_5km"] = np.array([np.average(loc_border[i], weights=loc_hh[i]) if len(i) else np.nan for i in near])
    f["nest_code"] = nest
    f["banner_code"] = pd.factorize(pd.Series(banner), sort=True)[0]
    return pd.DataFrame(f)


def store_by_store(yr, dr, zr, store, J):
    num, den = np.bincount(store, zr * yr, J), np.bincount(store, zr * dr, J)
    th = num / den
    u = yr - th[store] * dr
    se = np.sqrt(np.bincount(store, (zr * u) ** 2, J)) / np.abs(den)
    return th, se


def linear_iv(yr, dr, zr, Xs):
    """θ(x) = [1, x]·b by 2SLS with instruments z̃·[1, x]; returns b and θ for any X."""
    W = np.c_[np.ones(len(Xs)), Xs]
    D, Z = dr[:, None] * W, zr[:, None] * W
    b = np.linalg.solve(Z.T @ D, Z.T @ yr)
    return b


def local_guppi(Jac, price, theta, party):
    """party: 'migros' / 'coop' / '' per store. Margins m_k = p_k / |θ_k|."""
    m = price / np.abs(theta)
    wM = np.where(party == "migros", m, 0.0)
    wC = np.where(party == "coop", m, 0.0)
    own = -Jac.diagonal()
    to_m, to_c = Jac @ wM, Jac @ wC
    g = np.where(party == "migros", to_c, np.where(party == "coop", to_m, np.nan)) / own / price
    return g
