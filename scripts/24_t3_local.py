"""Pillar 2 step 5: test T3, local elasticities [sim]. Can the parties' sales data tell us how price-sensitive
each store's shoppers are, and does that get the local GUPPI screen right?

Five estimators of store-level elasticities (national, banner, store-by-store IV, linear IV with store features,
causal IV forest) on the cross-fitted DML residuals of 2.4, scored against the true store elasticity. Then local
GUPPIs for every Migros-group and Coop-group store, with margins from each method, against the true GUPPI.

Run:  conda activate keystone && python scripts/24_t3_local.py      (needs 20; recomputes the 2.4 residuals)
Outputs (assets/p2/m5/):
  2_5_store_estimates.csv    per store: truth, five estimates, forest 95% interval, features
  2_5_scorecard.csv          RMSE, correlation, calibration slope, coverage per method
  2_5_guppi.csv              per merging-party store: true GUPPI and GUPPI under each method
  2_5_guppi_scorecard.csv    GUPPI RMSE and flag errors at the 10% threshold
  2_5_feature_importance.csv forest split importance
  2_5_scatter.png, 2_5_scorecard.png, 2_5_elasticity_maps.png, 2_5_guppi_maps.png, 2_5_importance.png
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from mxc import viz  # noqa: E402
from mxc.causal import dml  # noqa: E402
from mxc.causal import forest as fo  # noqa: E402
from mxc.structural import demand as dm  # noqa: E402
from mxc.structural import substitution as sb  # noqa: E402

PROCESSED, INTERIM = ROOT / "data/processed", ROOT / "data/interim"
TWIN = PROCESSED / "twin"
OUT = ROOT / "assets/p2/m5"
P1 = ROOT / "assets/p1"
METHODS = ["national", "banner", "store-by-store IV", "linear IV", "causal forest"]
COL = dict(zip(METHODS, [viz.MUTED, viz.INK_2, viz.ORANGE, viz.AQUA, viz.BLUE]))
BLUES = ["#d6e6f8", "#a9cbf0", "#6fa6e6", "#2a78d6", "#1d58a3", "#123a6e"]
REDS = ["#fbe3e2", "#f6c7c6", "#ee9190", "#e34948", "#b52f2e", "#7a1f1e"]
THRESH = 0.10


def load_theta():
    par = json.load(open(P1 / "m2/1_2_params.json"))
    return dm.prior_params()._replace(
        delta=jnp.array([par["delta"][b] for b in dm.BANNERS]), alpha_ref=par["alpha_ref"], eta=par["eta"],
        gamma=par["gamma"], kappa=par["kappa"], sigma=par["sigma"],
        p=jnp.array([par["price_index"][b] for b in dm.BANNERS]), p_abroad=par["p_abroad"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(PROCESSED / "market.npz", allow_pickle=True)
    mkt = dm.Market(**{k: z[k] for k in z.files})
    J = len(mkt.store_id)
    truth = pd.read_parquet(TWIN / "truth_store.parquet")
    banner = truth.banner.to_numpy()

    # store features from public data
    stores = pd.read_parquet(PROCESSED / "stores.parquet", columns=["store_id", "lon", "lat"]).set_index("store_id").loc[mkt.store_id]
    pts = gpd.GeoSeries(gpd.points_from_xy(stores.lon, stores.lat), crs=4326).to_crs(2056)
    xy = np.c_[pts.x, pts.y] / 1000
    loc = pd.read_parquet(PROCESSED / "locations.parquet", columns=["loc_id", "x", "y", "median_net_income"]).set_index("loc_id").loc[mkt.loc_id]
    F = fo.store_features(xy, banner, np.asarray(mkt.store_nest), np.c_[loc.x, loc.y] / 1000, np.asarray(mkt.hh, float),
                          loc.median_net_income.fillna(loc.median_net_income.median()).to_numpy(),
                          np.asarray(mkt.border_min))
    feats = list(F.columns)

    # DML residuals (as in 2.4)
    sw = pd.read_parquet(TWIN / "store_week.parquet").merge(pd.read_parquet(TWIN / "store.parquet", columns=["store", "canton"]), on="store")
    df = dml.add_features(sw)
    pred, _ = dml.crossfit(df, ["y_dm", "d_dm", "z_dm"])
    yr, dr, zr = (df[c].to_numpy() - pred[c] for c in ("y_dm", "d_dm", "z_dm"))
    s = df.store.to_numpy()

    est = pd.DataFrame({"store_id": mkt.store_id, "banner": banner, "truth": truth.own_elasticity.to_numpy()})
    nat = dml.theta_by_cell(yr, dr, zr, np.array(["all"] * len(s)), s).theta.iloc[0]
    est["national"] = nat
    bt = dml.theta_by_cell(yr, dr, zr, df.banner.to_numpy(), s).set_index("cell").theta
    est["banner"] = bt.reindex(banner).to_numpy()
    th, se = fo.store_by_store(yr, dr, zr, s, J)
    est["store-by-store IV"], est["sbs_se"] = th, se
    Xf = F.to_numpy(float)
    mu, sd = np.nanmean(Xf, 0), np.nanstd(Xf, 0) + 1e-9
    Xs = np.nan_to_num((Xf - mu) / sd)
    b = fo.linear_iv(yr, dr, zr, Xs[s])
    est["linear IV"] = np.c_[np.ones(J), Xs] @ b

    from econml.grf import CausalIVForest
    cf = CausalIVForest(n_estimators=400, min_samples_leaf=400, max_samples=0.2, max_depth=None, honest=True,
                        inference=True, n_jobs=-1, random_state=0)
    cf.fit(np.nan_to_num(Xf)[s], dr, yr, Z=zr)
    lo, hi = cf.predict_interval(np.nan_to_num(Xf), alpha=0.05)
    est["causal forest"] = cf.predict(np.nan_to_num(Xf)).ravel()
    est["cf_lo95"], est["cf_hi95"] = lo.ravel(), hi.ravel()
    imp = pd.Series(cf.feature_importances_, index=feats).sort_values(ascending=False)
    imp.round(4).to_csv(OUT / "2_5_feature_importance.csv", header=["importance"])
    est = pd.concat([est, F], axis=1)
    est.round(4).to_csv(OUT / "2_5_store_estimates.csv", index=False)

    def score(m):
        e, t = est[m], est.truth
        ok = np.isfinite(e)
        row = {"method": m, "rmse": np.sqrt(((e - t)[ok] ** 2).mean()), "bias": (e - t)[ok].mean(),
               "corr": np.corrcoef(e[ok], t[ok])[0, 1] if e[ok].std() > 0 else np.nan,
               "calibration_slope": np.polyfit(e[ok], t[ok], 1)[0] if e[ok].std() > 0 else np.nan,
               "share_within_0.25": (np.abs(e - t)[ok] < 0.25).mean()}
        if m == "causal forest":
            row["coverage_95"] = ((est.cf_lo95 <= t) & (t <= est.cf_hi95)).mean()
        if m == "store-by-store IV":
            row["coverage_95"] = ((e - 1.96 * est.sbs_se <= t) & (t <= e + 1.96 * est.sbs_se))[ok].mean()
        return row
    sc = pd.DataFrame([score(m) for m in METHODS]).set_index("method")
    sc.round(4).to_csv(OUT / "2_5_scorecard.csv")
    print(sc.round(3).to_string())
    print(imp.head(8).round(3))

    # local GUPPI: true diversion; margins from each method's elasticities
    theta = load_theta()
    sbn = np.asarray(mkt.store_banner)
    a0 = {**dm.to_arrays(mkt), "xi_store": jnp.asarray(truth.xi_perm.to_numpy()),
          "p_store": jnp.asarray(np.asarray(theta.p)[sbn])}
    Jac, _, _ = sb.jacobian(theta, a0)
    price = np.asarray(theta.p)[sbn]
    party = np.array([fo.GROUPS.get(b, "") if fo.GROUPS.get(b, "") in ("migros", "coop") else "" for b in banner])
    gu = pd.DataFrame({"store_id": mkt.store_id, "banner": banner, "party": party})
    gu["truth"] = fo.local_guppi(Jac, price, est.truth.to_numpy(), party)
    ols = pd.read_csv(ROOT / "assets/p2/m4/2_4_estimates.csv")
    ols_nat = float(ols[(ols.level == "national") & (ols.method == "OLS-FE")].theta.iloc[0])
    gu["OLS national"] = fo.local_guppi(Jac, price, np.full(J, ols_nat), party)
    for m in METHODS:
        e = est[m].to_numpy()
        e = np.where(np.isfinite(e) & (e < -0.5), e, nat)                   # guard: implausible store-by-store values
        gu[m] = fo.local_guppi(Jac, price, e, party)
    gu = gu[gu.party != ""].reset_index(drop=True)
    gu.round(5).to_csv(OUT / "2_5_guppi.csv", index=False)
    gs = []
    for m in ["OLS national"] + METHODS:
        err = gu[m] - gu.truth
        tf, ef = gu.truth >= THRESH, gu[m] >= THRESH
        gs.append({"method": m, "rmse_pp": 100 * np.sqrt((err ** 2).mean()), "bias_pp": 100 * err.mean(),
                   "true_flags": int(tf.sum()), "flags": int(ef.sum()), "missed_flags": int((tf & ~ef).sum()),
                   "false_flags": int((~tf & ef).sum()), "misclassified_pct": 100 * (tf != ef).mean()})
    gs = pd.DataFrame(gs).set_index("method")
    gs.round(3).to_csv(OUT / "2_5_guppi_scorecard.csv")
    print(gs.round(2).to_string())

    figures(est, sc, gu, gs, imp, xy)


def figures(est, sc, gu, gs, imp, xy):
    viz.setup()
    # 1. scatter estimate vs truth
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharex=True, sharey=True)
    lim = (est.truth.min() - 0.3, est.truth.max() + 0.3)
    for ax, m in zip(axes, METHODS[1:]):
        e = est[m].clip(lim[0] - 2, lim[1] + 2)
        ax.scatter(est.truth, e, s=3, color=COL[m], alpha=0.5, linewidths=0)
        ax.plot(lim, lim, color=viz.INK_2, linewidth=0.8)
        ax.set_xlim(*lim)
        ax.set_ylim(lim[0] - 1.5, lim[1] + 1.5)
        ax.set_title(f"{m}\nRMSE {sc.loc[m, 'rmse']:.2f}, corr {sc.loc[m, 'corr']:.2f}", fontsize=9.5)
        ax.set_xlabel("true store elasticity", fontsize=8.5)
    axes[0].set_ylabel("estimated store elasticity")
    fig.suptitle("Store by store is noise; the causal forest pools similar stores and tracks the truth best [sim]",
                 x=0.01, ha="left", fontweight="bold")
    fig.subplots_adjust(top=0.78)
    viz.source(axes[0], "4,565 real stores; diagonal = perfect estimate. Store-by-store values clipped for display.")
    viz.save(fig, 2, "2_5", "scatter")

    # 2. scorecard
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    yy = np.arange(len(METHODS))
    for ax, (col, lab, fmt) in zip(axes, [("rmse", "RMSE vs true store elasticity", "{:.2f}"),
                                          ("corr", "correlation with truth", "{:.2f}"),
                                          ("share_within_0.25", "stores within ±0.25 of truth", "{:.0%}")]):
        v = sc[col].reindex(METHODS).fillna(0)
        ax.barh(yy, v.values, color=[COL[m] for m in METHODS], height=0.6)
        for i, val in enumerate(v.values):
            ax.text(val, i, " " + fmt.format(val), va="center", fontsize=8.5)
        ax.set_yticks(yy, METHODS if col == "rmse" else [""] * len(METHODS))
        ax.invert_yaxis()
        ax.set_xlim(0, v.max() * 1.3)
        ax.set_title(lab, fontsize=10.5)
        ax.grid(axis="y", visible=False)
    cov = sc.loc["causal forest", "coverage_95"]
    fig.suptitle(f"T3 scorecard: the causal forest is best on every store-level measure (its 95% intervals cover "
                 f"{cov:.0%}) [sim]", x=0.01, ha="left", fontweight="bold")
    fig.subplots_adjust(top=0.80)
    viz.save(fig, 2, "2_5", "scorecard")

    # 3. maps: truth vs store-by-store vs forest
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cantons = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)
    show = {"Truth": est.truth, "Store-by-store IV": est["store-by-store IV"], "Causal forest": est["causal forest"]}
    q = np.quantile(-est.truth, np.linspace(0, 1, 7))
    allv = -np.concatenate([v.to_numpy() for v in show.values()])
    allv = allv[np.isfinite(allv)]
    bins = [min(allv.min(), q[0]) - 0.01] + list(np.round(q[1:-1], 2)) + [max(allv.max(), q[-1]) + 0.01]
    cmap, norm = ListedColormap(BLUES), BoundaryNorm(bins, len(BLUES))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.2, wspace=0.01)
    for ax, (ttl, v) in zip(axes, show.items()):
        viz.map_axes(ax, cantons)
        o = np.argsort(-v.to_numpy())
        ax.scatter(xy[o, 0] * 1000, xy[o, 1] * 1000, c=-v.to_numpy()[o], cmap=cmap, norm=norm, s=6, linewidths=0,
                   zorder=2)
        ax.set_title(ttl, fontsize=10.5)
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, orientation="horizontal",
                       fraction=0.05, pad=0.03, aspect=60, ticks=bins[1:-1])
    cb_.ax.set_xticklabels([f"{b:.2f}" for b in bins[1:-1]], fontsize=8)
    cb_.set_label("own-price elasticity, absolute value (classes = sextiles of the truth)", fontsize=8.5)
    cb_.outline.set_visible(False)
    fig.suptitle(f"The forest gets the pattern (corr {sc.loc['causal forest', 'corr']:.2f}) but pulls extremes to the middle; "
                 f"store-by-store is too noisy (corr {sc.loc['store-by-store IV', 'corr']:.2f}) [sim]",
                 x=0.01, y=0.99, ha="left", fontweight="bold")
    viz.save(fig, 2, "2_5", "elasticity_maps")

    # 4. GUPPI maps: truth vs forest vs OLS-national margins
    sid = pd.Series(np.arange(len(est)), index=est.store_id)
    idx = sid.loc[gu.store_id].to_numpy()
    show = {"True GUPPI": gu.truth, "Causal-forest margins": gu["causal forest"], "OLS margins (national)": gu["OLS national"]}
    bins = [0, 0.05, 0.075, 0.10, 0.125, 0.15, 1.0]
    cmap, norm = ListedColormap(REDS), BoundaryNorm(bins, len(REDS))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.2, wspace=0.01)
    for ax, (ttl, v) in zip(axes, show.items()):
        viz.map_axes(ax, cantons)
        o = np.argsort(v.to_numpy())
        vv, big = v.to_numpy()[o], v.to_numpy()[o] >= THRESH
        ax.scatter(xy[idx[o], 0] * 1000, xy[idx[o], 1] * 1000, c=vv, cmap=cmap, norm=norm, s=np.where(big, 16, 4),
                   edgecolors=np.where(big, viz.INK, "none"), linewidths=np.where(big, 0.3, 0), zorder=2)
        name = {"True GUPPI": "truth", "Causal-forest margins": "causal forest", "OLS margins (national)": "OLS national"}[ttl]
        n = gs["true_flags"].iloc[0] if name == "truth" else gs.loc[name, "flags"]
        ax.set_title(f"{ttl}: {int(n):,} stores ≥ 10% (outlined)", fontsize=10.5)
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, orientation="horizontal",
                       fraction=0.05, pad=0.03, aspect=60, ticks=bins[1:-1])
    cb_.ax.set_xticklabels([f"{b:.1%}" for b in bins[1:-1]], fontsize=8)
    cb_.set_label("local GUPPI of each Migros-group / Coop-group store (true diversion × margin from each method)",
                  fontsize=8.5)
    cb_.outline.set_visible(False)
    fig.suptitle(f"OLS margins would flag {gs.loc['OLS national', 'false_flags']:,} stores wrongly; forest margins "
                 f"misclassify {gs.loc['causal forest', 'misclassified_pct']:.1f}% [sim]",
                 x=0.01, y=0.99, ha="left", fontweight="bold")
    viz.save(fig, 2, "2_5", "guppi_maps")

    # 5. importance
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    top = imp.head(12)[::-1]
    ax.barh(top.index, top.values, color=viz.BLUE, height=0.6)
    ax.set_xlabel("share of forest splits (weighted)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"What drives local price sensitivity in the forest: {imp.index[0]}, {imp.index[1]}, {imp.index[2]}",
                 fontsize=10.5)
    viz.source(ax, "Features built from public data only (store locations, households, income, border time). [sim]")
    viz.save(fig, 2, "2_5", "importance")


def figures_only():
    """Redraw from the saved csv files (skips the 4-minute forest fit)."""
    z = np.load(PROCESSED / "market.npz", allow_pickle=True)
    stores = pd.read_parquet(PROCESSED / "stores.parquet", columns=["store_id", "lon", "lat"]).set_index("store_id").loc[z["store_id"]]
    pts = gpd.GeoSeries(gpd.points_from_xy(stores.lon, stores.lat), crs=4326).to_crs(2056)
    xy = np.c_[pts.x, pts.y] / 1000
    est = pd.read_csv(OUT / "2_5_store_estimates.csv")
    sc = pd.read_csv(OUT / "2_5_scorecard.csv", index_col=0)
    gu = pd.read_csv(OUT / "2_5_guppi.csv")
    gs = pd.read_csv(OUT / "2_5_guppi_scorecard.csv", index_col=0)
    imp = pd.read_csv(OUT / "2_5_feature_importance.csv", index_col=0).importance
    figures(est, sc, gu, gs, imp, xy)


if __name__ == "__main__":
    figures_only() if "--figures" in sys.argv else main()
