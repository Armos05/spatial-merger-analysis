"""T2: own-price elasticities on the twin — OLS vs IV vs double machine learning (Pillar 2 step 4; DESIGN §4.3).

Model (partially linear, Chernozhukov et al. 2018):   ln q_jt = θ_c × ln p_jt + g(X_jt) + ε_jt
θ_c is the own-price elasticity in cell c (banner or canton). ln p is endogenous: managers raise prices when
demand is strong, so E[ε | ln p] ≠ 0. The cost index z (observed, moves price, not demand) is the instrument.

Every method is the same residual-on-residual estimator; they differ only in how residuals are made:
    ỹ = y − E[y | X],  d̃ = d − E[d | X],  z̃ = z − E[z | X]
    PLR  (no instrument):  θ_c = Σ_c d̃ ỹ / Σ_c d̃ d̃
    PLIV (instrument z):   θ_c = Σ_c z̃ ỹ / Σ_c z̃ d̃
    OLS-FE  = PLR  with E[· | X] = store + week fixed effects only (two-way demeaning)
    IV-FE   = PLIV with the same fixed effects
    DML     = fixed effects first, then E[· | X] learned by gradient boosting (LightGBM) with 5-fold
              cross-fitting by store, so no store's own data predicts its own residual.
Standard errors: cluster-robust by store,  Var θ_c = Σ_g (Σ_{i∈g, c} z̃ u)² / (Σ_c z̃ d̃)²,  u = ỹ − θ_c d̃.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .entry_pooled import demean


def add_features(sw: pd.DataFrame, lags=(1, 2, 3, 4)) -> pd.DataFrame:
    """sw: store, week, banner, canton, baskets, price, cost_index, flyer, local_promo (one row per store-week).
    Adds y, d, z, two-way demeaned versions, lags, and the leave-one-out cost of sister stores (same banner, canton)."""
    df = sw.sort_values(["store", "week"]).reset_index(drop=True).copy()
    df["y"], df["d"], df["z"] = np.log(df.baskets), np.log(df.price), df.cost_index
    cell = df.banner.astype(str) + "|" + df.canton.astype(str) + "|" + df.week.astype(str)
    s, n = df.groupby(cell).z.transform("sum"), df.groupby(cell).z.transform("size")
    df["sister_cost"] = np.where(n > 1, (s - df.z) / (n - 1).clip(lower=1), np.nan)
    df["sister_n"] = n - 1
    fe = [pd.factorize(df.store)[0], df.week.to_numpy()]
    cols = ["y", "d", "z", "flyer", "local_promo"]
    D = demean(df[cols].astype(float).to_numpy(), fe)
    for i, c in enumerate(cols):
        df[f"{c}_dm"] = D[:, i]
    ok = df.sister_cost.notna()
    sc = np.full(len(df), np.nan)
    sc[ok.to_numpy()] = demean(df.loc[ok, ["sister_cost"]].to_numpy(),
                               [pd.factorize(df.store[ok])[0], df.week[ok].to_numpy()])[:, 0]
    df["sister_cost_dm"] = sc
    g = df.groupby("store")
    full = df.set_index(["store", "week"])
    for L in lags:                                     # lags by calendar week (NaN across gaps)
        idx = pd.MultiIndex.from_arrays([df.store, df.week - L])
        for c in ("y_dm", "d_dm", "z_dm"):
            df[f"{c}_lag{L}"] = full[c].reindex(idx).to_numpy()
    df["woy"] = pd.DatetimeIndex(df.date).isocalendar().week.to_numpy().astype(int) if "date" in df else df.week % 52
    for c in ("banner", "canton"):
        df[f"{c}_code"] = pd.factorize(df[c])[0]
    del g
    return df


ML_FEATURES = ["flyer_dm", "local_promo_dm", "sister_cost_dm", "sister_n", "woy", "banner_code", "canton_code"] + \
              [f"{c}_lag{L}" for L in (1, 2, 3, 4) for c in ("y_dm", "d_dm", "z_dm")]


def crossfit(df: pd.DataFrame, targets, features=ML_FEATURES, folds=5, seed=0, params=None):
    """Out-of-fold predictions of each target from the features (LightGBM), folds by store."""
    import lightgbm as lgb
    params = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 200,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1, "random_state": seed,
              **(params or {})}
    rng = np.random.default_rng(seed)
    stores = df.store.unique()
    fold_of = pd.Series(rng.integers(0, folds, len(stores)), index=stores)
    f = fold_of.loc[df.store].to_numpy()
    X = df[features].to_numpy(float)
    out, r2 = {}, {}
    for t in targets:
        y = df[t].to_numpy(float)
        pred = np.empty(len(df))
        for k in range(folds):
            tr, te = f != k, f == k
            m = lgb.LGBMRegressor(**params).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        out[t] = pred
        r2[t] = 1 - np.var(y - pred) / np.var(y)
    return out, r2


def theta_by_cell(yr, dr, zr, cell, cluster):
    """Residual-on-residual (IV if zr is not dr) estimate per cell with store-clustered standard errors."""
    cells = pd.factorize(cell, sort=True)
    codes, names = cells
    rows = []
    for c, name in enumerate(names):
        m = codes == c
        den = (zr[m] * dr[m]).sum()
        th = (zr[m] * yr[m]).sum() / den
        u = yr[m] - th * dr[m]
        s = np.bincount(pd.factorize(cluster[m])[0], zr[m] * u)
        rows.append({"cell": name, "theta": th, "se": np.sqrt((s ** 2).sum()) / abs(den), "n": int(m.sum()),
                     "first_stage_corr": float(np.corrcoef(zr[m], dr[m])[0, 1])})
    return pd.DataFrame(rows)
