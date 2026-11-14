"""Pillar 3 step 1: randomised coupon experiments → each household's price sensitivity α̂_i. [sim]

Each chain runs an A/B test on its own card holders (after the 156 weeks of the 2.2 panel):
    treatment: a personal coupon, 10% off all card banners of the chain for 8 weeks (half of holders, random)
    outcome:   card-recorded trips at the chain in those 8 weeks
    Y_i ~ Poisson(8 × trip_rate_i × swipe 0.9 × s_i,chain(prices)),  s from the household's true model.
True uplift τ_i = ln s_i(coupon) − ln s_i(no coupon) is known from the twin.

What each data regime can use as features X_i (all built from what that firm would really hold):
    location only   home location: drive minutes to the nearest Migros-group / Coop-group / discount store,
                    stores within 10 min, minutes to the border, municipal mean income
    Cumulus only    + 156 weeks of the household's Cumulus trips: trips/week, spend, basket mix, convenience share,
                    stores used, flyer-week share, and the price it paid relative to the store's usual price
                    (sensitive shoppers come when prices are low)
    Supercard only  the same from Supercard
    pooled          location + both cards + share of the household's card trips at Migros vs Coop
Estimator: cross-fitted T-learner (LightGBM Poisson; 5 folds): τ̂(x) = ln μ̂1(x) − ln μ̂0(x).
Then the firm inverts its own demand model (no tastes, average travel costs): α̂_i solves
    ln s_i(α, coupon) − ln s_i(α, none) = τ̂_i               (bisection, α in [0.002, 0.25])
Pooled α̂ for a household with both cards = mean of the two inversions.

Run:  conda activate keystone && python scripts/30_experiments.py      (needs 20, 21)
Data (data/processed/p3/, not in git): features.parquet, alpha_hat.parquet
Outputs (assets/p3/m1/):
  3_1_experiment.csv       ATE per chain: estimate (difference in means) vs truth, by segment
  3_1_uplift_scorecard.csv  τ̂ vs true τ: correlation, RMSE, by regime
  3_1_uplift_segment.png   true vs estimated uplift by hidden segment and regime
  3_1_uplift_map.png       estimated uplift of the Migros coupon by home canton (pooled) vs truth
"""
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from mxc import viz  # noqa: E402
from mxc.datamerger import personal as pe  # noqa: E402
from mxc.structural import demand as dm  # noqa: E402
from mxc.twin import loyalty as ly  # noqa: E402

PROCESSED, INTERIM = ROOT / "data/processed", ROOT / "data/interim"
EV = PROCESSED / "twin/events"
P3 = PROCESSED / "p3"
OUT = ROOT / "assets/p3/m1"
T20 = runpy.run_path(str(ROOT / "scripts/20_twin.py"), run_name="lib")
BLUES = ["#d6e6f8", "#a9cbf0", "#6fa6e6", "#2a78d6", "#1d58a3", "#123a6e"]
SEED, WEEKS, DISCOUNT, SWIPE = 2027, 8, 0.10, 0.9
SEG_ORDER = ["price_hunter", "border_shopper", "mainstream", "convenience", "quality"]


def setup():
    """Everything later Pillar 3 steps need: market, households, arrays, costs, prices."""
    mkt, theta = T20["load"]()
    a = dm.to_arrays(mkt)
    st = pd.read_parquet(PROCESSED / "twin/store.parquet").sort_values("store")
    xi = st.xi_perm.to_numpy()
    hh = pd.read_parquet(EV / "households.parquet")
    sup = json.load(open(ROOT / "assets/p1/m4/1_4_supply_params.json"))
    cost = jnp.array([sup["marginal_cost"][b] for b in dm.BANNERS])
    lm = jnp.array([sup["lambda_mu"][b] for b in dm.BANNERS])
    pr = pd.read_csv(ROOT / "assets/p1/m6/1_6_prices.csv", index_col=0)
    pre = jnp.array(pr.p_pre.reindex(dm.BANNERS).to_numpy())
    post = jnp.array(pr.p_post.reindex(dm.BANNERS).to_numpy())
    ahT = pe.panel_arrays(a, hh, xi, informed=True)
    ahB = pe.panel_arrays(a, hh, xi, informed=False)
    return dict(mkt=mkt, theta=theta, a=a, hh=hh, cost=cost, lm=lm, pre=pre, post=post, ahT=ahT, ahB=ahB, xi=xi)


def location_features(S):
    mkt, hh, a = S["mkt"], S["hh"], S["a"]
    sb = np.asarray(mkt.store_banner)
    ban = np.array(dm.BANNERS)[sb[mkt.pair_store]]
    t = pd.DataFrame({"loc": mkt.pair_loc, "time": mkt.pair_time,
                      "grp": np.select([np.isin(ban, pe.MIGROS_GROUP), np.isin(ban, pe.COOP_GROUP),
                                        np.isin(ban, ("Aldi", "Lidl"))], ["migros", "coop", "discount"], "other")})
    near = t.pivot_table(index="loc", columns="grp", values="time", aggfunc="min")
    n10 = t[t.time <= 10].groupby("loc").size()
    locs = pd.read_parquet(PROCESSED / "locations.parquet", columns=["loc_id", "mean_net_income"]).set_index("loc_id")
    L = hh["loc"].to_numpy()
    X = pd.DataFrame({"hh_id": hh.hh_id,
                      "min_to_migros": near["migros"].reindex(L).fillna(30).to_numpy(),
                      "min_to_coop": near["coop"].reindex(L).fillna(30).to_numpy(),
                      "min_to_discount": near["discount"].reindex(L).fillna(30).to_numpy(),
                      "stores_10min": n10.reindex(L).fillna(0).to_numpy(),
                      "border_min": np.asarray(a["border_min"])[L],
                      "muni_income": locs.mean_net_income.reindex(hh.loc_id).to_numpy()})
    return X.set_index("hh_id")


def card_features(card, prefix):
    c = pd.read_parquet(EV / f"{card}.parquet")
    sw = pd.read_parquet(EV / "store_week.parquet", columns=["store_id", "week", "price", "flyer"])
    sw["rel_price"] = np.log(sw.price / sw.groupby("store_id").price.transform("mean"))
    c = c.merge(sw[["store_id", "week", "rel_price", "flyer"]], on=["store_id", "week"], how="left")
    c["conv"] = c.banner.isin(["Migrolino", "Coop Pronto"])
    c["lspend"] = np.log(c.spend)
    g = c.groupby("card_id")
    f = pd.DataFrame({"trips_wk": g.size() / 156, "mean_lspend": g.lspend.mean(), "sd_lspend": g.lspend.std(),
                      "fresh": g.dept_fresh.mean(), "staples": g.dept_staples.mean(), "treats": g.dept_treats.mean(),
                      "conv_share": g.conv.mean(), "n_stores": g.store_id.nunique(), "flyer_share": g.flyer.mean(),
                      "rel_price_paid": g.rel_price.mean(), "weeks_active": g.week.nunique() / 156})
    return f.add_prefix(prefix)


def experiment(S, rng, side):
    hh, theta = S["hh"], S["theta"]
    H = len(hh)
    one, cut = jnp.ones(H), jnp.full(H, 1 - DISCOUNT)
    al = jnp.asarray(hh.alpha_i.to_numpy())
    mM1, mC1 = (cut, one) if side == "M" else (one, cut)
    s1 = np.asarray(pe.chain_share(theta, S["ahT"], S["pre"], al, mM1, mC1, side))
    s0 = np.asarray(pe.chain_share(theta, S["ahT"], S["pre"], al, one, one, side))
    # eligible: holds the card and has a store of the chain within 25 minutes
    holder = (hh.cumulus.to_numpy() if side == "M" else hh.supercard.to_numpy()) & (s0 > 1e-12)
    d = (rng.random(H) < 0.5) & holder
    s0, s1 = np.maximum(s0, 1e-300), np.maximum(s1, 1e-300)
    lam = WEEKS * hh.trip_rate.to_numpy() * SWIPE * np.where(d, s1, s0)
    y = rng.poisson(lam)
    return pd.DataFrame({"hh_id": hh.hh_id, "holder": holder, "treated": d, "y": y, "s0": s0, "s1": s1,
                         "tau_true": np.log(s1) - np.log(s0)})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P3.mkdir(parents=True, exist_ok=True)
    S = setup()
    hh, theta = S["hh"], S["theta"]
    rng = np.random.default_rng(SEED + 30)
    XL = location_features(S)
    XM, XC = card_features("cumulus", "M_"), card_features("supercard", "C_")
    X = XL.join(XM).join(XC)
    X["M_share_of_card_trips"] = X.M_trips_wk.fillna(0) / (X.M_trips_wk.fillna(0) + X.C_trips_wk.fillna(0))
    X.to_parquet(P3 / "features.parquet")
    exps = {s: experiment(S, rng, s) for s in ("M", "C")}

    cols = {"location only": list(XL.columns),
            "own card": None,
            "pooled": list(X.columns)}
    res = pd.DataFrame({"hh_id": hh.hh_id, "segment": hh.segment, "alpha_true": hh.alpha_i,
                        "cumulus": hh.cumulus, "supercard": hh.supercard}).set_index("hh_id")
    for side, card_cols in (("M", list(XL.columns) + list(XM.columns)), ("C", list(XL.columns) + list(XC.columns))):
        e = exps[side].set_index("hh_id")
        res[f"tau_true_{side}"] = e.tau_true
        res[f"treated_{side}"] = e.treated
        res[f"y_{side}"] = e.y
        h = e.holder.to_numpy()
        for reg, cl in (("location only", cols["location only"]), ("own card", card_cols), ("pooled", cols["pooled"])):
            Xr = X.loc[e.index[h], cl].to_numpy(float)
            tau, _ = pe.t_learner(Xr, e.y.to_numpy()[h], e.treated.to_numpy()[h].astype(int), seed=SEED)
            key = reg.replace(" ", "_")
            res.loc[e.index[h], f"tau_{key}_{side}"] = tau
            full = np.full(len(hh), np.nan)
            full[h] = tau
            fill = np.where(np.isnan(full), np.nanmedian(tau), full)
            alpha = pe.invert_alpha(theta, S["ahB"], S["pre"], jnp.asarray(fill), side, DISCOUNT)
            res[f"alpha_{key}_{side}"] = np.where(h, alpha, np.nan)
            print(side, reg, "corr tau", np.corrcoef(tau, e.tau_true.to_numpy()[h])[0, 1].round(3),
                  "corr alpha", np.corrcoef(alpha[h], hh.alpha_i.to_numpy()[h])[0, 1].round(3), flush=True)
    # what each firm believes about each household
    mean_a = float(np.nanmean(res.alpha_location_only_M))
    res["alpha_hat_M_own"] = res.alpha_own_card_M.fillna(mean_a)
    res["alpha_hat_C_own"] = res.alpha_own_card_C.fillna(mean_a)
    res["alpha_hat_pooled"] = res[["alpha_pooled_M", "alpha_pooled_C"]].mean(1).fillna(mean_a)
    res["alpha_hat_location"] = res[["alpha_location_only_M", "alpha_location_only_C"]].mean(1).fillna(mean_a)
    res.to_parquet(P3 / "alpha_hat.parquet")

    # ---- tables
    rows = []
    for side, name in (("M", "Migros (Cumulus)"), ("C", "Coop (Supercard)")):
        e = exps[side][exps[side].holder]
        for seg in ["all"] + SEG_ORDER:
            g = e if seg == "all" else e[hh.segment.to_numpy()[e.hh_id] == seg]
            y1, y0 = g.y[g.treated], g.y[~g.treated]
            ate = np.log(y1.mean() / y0.mean())
            se = np.sqrt(y1.var() / len(y1) / y1.mean() ** 2 + y0.var() / len(y0) / y0.mean() ** 2)
            rows.append({"chain": name, "segment": seg, "households": len(g), "treated": int(g.treated.sum()),
                         "ate_log_est": ate, "se": se, "ate_log_true": np.log((g.s1 * g.y.size).sum() / (g.s0 * g.y.size).sum()),
                         "mean_tau_true": g.tau_true.mean(), "trips_control_8wk": y0.mean()})
    ex = pd.DataFrame(rows)
    ex.round(4).to_csv(OUT / "3_1_experiment.csv", index=False)
    print(ex.round(3).to_string(index=False))
    sc = []
    for side in ("M", "C"):
        for reg in ("location_only", "own_card", "pooled"):
            m = res[f"tau_{reg}_{side}"].notna()
            t, tt = res.loc[m, f"tau_{reg}_{side}"], res.loc[m, f"tau_true_{side}"]
            sc.append({"chain": side, "regime": reg, "households": int(m.sum()), "corr": np.corrcoef(t, tt)[0, 1],
                       "rmse": np.sqrt(((t - tt) ** 2).mean()), "bias": (t - tt).mean(), "sd_true": tt.std()})
    sc = pd.DataFrame(sc)
    sc.round(4).to_csv(OUT / "3_1_uplift_scorecard.csv", index=False)
    print(sc.round(3).to_string(index=False))
    figures(res, ex)


def figures(res, ex):
    viz.setup()
    # 1. uplift by segment: truth vs regimes (Migros coupon)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    regs = [("tau_true", "truth", viz.INK_2), ("tau_location_only", "location only", viz.MUTED),
            ("tau_own_card", "own card data", viz.ORANGE), ("tau_pooled", "pooled (both cards)", viz.BLUE)]
    for ax, side, ttl in zip(axes, ("M", "C"), ("Migros coupon (Cumulus holders)", "Coop coupon (Supercard holders)")):
        x = np.arange(len(SEG_ORDER))
        for k, (col, lab, colr) in enumerate(regs):
            held = res[f"tau_pooled_{side}"].notna()          # households in that chain's experiment
            v = [100 * res.loc[(res.segment == s) & held, f"{col}_{side}"].mean() for s in SEG_ORDER]
            ax.bar(x + (k - 1.5) * 0.2, v, 0.19, color=colr, label=lab)
        ax.set_xticks(x, [s.replace("_", "\n") for s in SEG_ORDER], fontsize=8.5)
        ax.set_title(ttl, fontsize=10.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("uplift in chain trips from a 10% coupon (%)")
    axes[0].legend(fontsize=8.5, loc="upper right")
    a = ex[ex.segment == "all"].set_index("chain")
    fig.suptitle(f"A 10% coupon lifts trips by {100 * a.ate_log_est.iloc[0]:.0f}% (Migros) and "
                 f"{100 * a.ate_log_est.iloc[1]:.0f}% (Coop); the card data recover who reacts most [sim]",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    viz.source(axes[0], "Randomised 8-week coupon test on each chain's card holders; T-learner (LightGBM Poisson, "
                        "5-fold cross-fitting); bars = mean uplift by hidden segment (the firm never sees the segment).")
    viz.save(fig, 3, "3_1", "uplift_segment")

    # 2. map: pooled estimated vs true uplift of the Migros coupon by home canton
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cant = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)[["ref", "geometry"]]
    hh = pd.read_parquet(EV / "households.parquet").set_index("hh_id")
    r = res.join(hh[["canton"]])
    m = r.tau_pooled_M.notna()
    can = r[m].groupby("canton").agg(est=("tau_pooled_M", "mean"), true=("tau_true_M", "mean"), n=("tau_true_M", "size"))
    g = cant.merge(100 * can[["est", "true"]], left_on="ref", right_index=True)
    lo, hi = np.floor(g[["est", "true"]].min().min()), np.ceil(g[["est", "true"]].max().max())
    bins = np.linspace(lo, hi, 7)
    cmap, norm = ListedColormap(BLUES), BoundaryNorm(bins, len(BLUES))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, ttl in zip(axes, ["true", "est"], ["Truth (twin)", "Estimated from the experiment + pooled card data"]):
        g.plot(ax=ax, column=col, cmap=cmap, norm=norm, edgecolor=viz.SURFACE, linewidth=0.6)
        viz.map_axes(ax, cant)
        ax.set_title(ttl, fontsize=10.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.03, shrink=0.5)
    cb.set_label("mean uplift in Migros trips from a 10% coupon, % (by home canton)", fontsize=8.5)
    cc = np.corrcoef(can.est, can.true)[0, 1]
    fig.suptitle(f"Where coupons work: the experiment maps price sensitivity across Switzerland "
                 f"(canton correlation {cc:.2f}) [sim]", x=0.02, ha="left", fontsize=12, fontweight="bold")
    viz.source(axes[0], "Cumulus holders in the 20,000-household panel, by home canton. [sim]")
    viz.save(fig, 3, "3_1", "uplift_map")


if __name__ == "__main__":
    main()
