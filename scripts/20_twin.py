"""Pillar 2 step 1: the digital twin. Simulated weekly store and store × category sales on real Swiss geography,
generated from the Pillar 1 demand model with known ("true") parameters. Everything here is [sim].

Run:  conda activate keystone && python scripts/20_twin.py      (needs 10-13)
Data (data/processed/twin/, not in git):
  store_week.parquet    what an analyst would get: price, baskets, revenue, promotions, cost index
  category.parquet      store × category × week sales (CHF), ~13M rows
  truth_week.parquet    hidden truth: true baskets, demand shocks, cost shocks
  weekly.parquet, store.parquet, truth_store.parquet
Outputs (assets/p2/m1/):
  2_1_twin_spec.csv         every twin parameter, its value and what it does
  2_1_panel_summary.csv     rows, stores, weeks, gaps, price-variance decomposition
  2_1_truth_store.csv       true own-price elasticity and diversion of every store (the answer key)
  2_1_validation.csv        twin vs Pillar 1 model; naive OLS vs cost-IV vs truth (preview of T2)
  2_1_weekly_sales.png, 2_1_price_components.png, 2_1_category_season.png, 2_1_elasticity_map.png,
  2_1_endogeneity.png
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
import scipy.sparse as sp  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm  # noqa: E402

from mxc import viz  # noqa: E402
from mxc.structural import calibrate as cb  # noqa: E402
from mxc.structural import demand as dm  # noqa: E402
from mxc.structural import substitution as sb  # noqa: E402
from mxc.twin import generate as tw  # noqa: E402

PROCESSED, INTERIM = ROOT / "data/processed", ROOT / "data/interim"
TWIN = PROCESSED / "twin"
OUT = ROOT / "assets/p2/m1"
P1 = ROOT / "assets/p1"
BLUES = ["#d6e6f8", "#a9cbf0", "#6fa6e6", "#2a78d6", "#1d58a3", "#123a6e"]
SPEC_TEXT = {
    "seed": "random seed (everything reproducible)", "weeks": "length of the panel", "start": "first Monday",
    "xi_perm_sd": "permanent store quality spread (utility)", "xi_rho": "demand-shock persistence (AR1)",
    "xi_sd": "demand-shock innovation (utility)", "cost_region_sd": "banner × canton cost shock (log)",
    "cost_store_sd": "store cost shock (log)", "cost_rho": "cost-shock persistence",
    "cost_obs_noise_sd": "noise in the analyst's cost index", "pass_through": "log price per log cost",
    "price_on_xi": "log price per unit demand shock (endogeneity)", "flyer_share": "banner-weeks with a national flyer",
    "flyer_depth": "flyer basket discount", "promo_share": "store-weeks with a local promotion",
    "promo_depth": "local promotion discount", "price_noise_sd": "other price noise (log)",
    "fx_rho": "EUR/CHF persistence", "fx_sd": "weekly EUR/CHF shock → abroad price",
    "sales_noise_sd": "scanner noise on baskets (log)", "cat_noise_sd": "noise on category sales (log)",
    "missing_rate": "random missing store-weeks", "outage_store_share": "stores with a 4-12 week data outage",
}


def load():
    z = np.load(PROCESSED / "market.npz", allow_pickle=True)
    mkt = dm.Market(**{k: z[k] for k in z.files})
    par = json.load(open(P1 / "m2/1_2_params.json"))
    theta = dm.prior_params()._replace(
        delta=jnp.array([par["delta"][b] for b in dm.BANNERS]), alpha_ref=par["alpha_ref"], eta=par["eta"],
        gamma=par["gamma"], kappa=par["kappa"], sigma=par["sigma"],
        p=jnp.array([par["price_index"][b] for b in dm.BANNERS]), p_abroad=par["p_abroad"])
    return mkt, theta


def two_way_demean(df, cols, i="store", t="week", iters=30):
    """Remove store and week fixed effects by alternating projections."""
    x = df[cols].to_numpy(float).copy()
    gi, gt = df[i].to_numpy(), df[t].to_numpy()
    ni, nt = np.bincount(gi), np.bincount(gt)
    for _ in range(iters):
        for g, n in ((gi, ni), (gt, nt)):
            means = np.stack([np.bincount(g, x[:, k], len(n)) for k in range(x.shape[1])], 1) / np.maximum(n, 1)[:, None]
            x -= means[g]
    return pd.DataFrame(x, columns=cols, index=df.index)


def truth_table(theta, a, mkt, xi_perm):
    """Answer key at normal times (base prices, permanent quality, no shocks)."""
    sbn = np.asarray(a["store_banner"])
    a0 = {**a, "xi_store": jnp.asarray(xi_perm), "p_store": jnp.asarray(np.asarray(theta.p)[sbn])}
    Jac, d0, q = sb.jacobian(theta, a0)
    p = np.asarray(theta.p)[sbn]
    own = Jac.diagonal()
    pidx = cb.player_index()[sbn]
    players = list(cb.PLAYERS)
    G = sp.csr_matrix((np.ones(len(sbn)), (np.arange(len(sbn)), pidx)), shape=(len(sbn), len(players)))
    to_player = ((Jac - sp.diags(own)) @ G).toarray() / -own[:, None]
    out = pd.DataFrame({"store_id": mkt.store_id, "banner": np.array(dm.BANNERS)[sbn], "price": p,
                        "baskets_week": q, "own_elasticity": own * p / q, "xi_perm": xi_perm})
    for k, name in enumerate(players):
        out[f"div_to_{name}"] = to_player[:, k]
    out["div_to_abroad"] = d0 / -own
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TWIN.mkdir(parents=True, exist_ok=True)
    mkt, theta = load()
    a = dm.to_arrays(mkt)
    stores = pd.read_parquet(PROCESSED / "stores.parquet", columns=["store_id", "canton", "lon", "lat", "geometry"])
    stores = stores.set_index("store_id").loc[mkt.store_id]
    cfg = tw.TwinConfig()
    tw_out = tw.generate(theta, a, stores.canton.to_numpy(), np.asarray(mkt.store_nest), cfg,
                         store_id=mkt.store_id)
    sw, tr, cat, wk, st = (tw_out[k] for k in ("store_week", "truth_week", "category", "weekly", "store"))
    for k in ("store_week", "truth_week", "category", "weekly", "store"):
        tw_out[k].to_parquet(TWIN / f"{k}.parquet", index=False)
    print(f"store-week rows {len(sw):,}; category rows {len(cat):,}")

    truth = truth_table(theta, a, mkt, st.xi_perm.to_numpy())
    truth.insert(2, "canton", stores.canton.to_numpy())
    truth.to_parquet(TWIN / "truth_store.parquet", index=False)
    truth.round(5).to_csv(OUT / "2_1_truth_store.csv", index=False)

    # spec
    pd.DataFrame([{"parameter": k, "value": v, "what_it_does": SPEC_TEXT[k], "label": "[sim]"}
                  for k, v in tw_out["config"].items()]).to_csv(OUT / "2_1_twin_spec.csv", index=False)

    # price-variance decomposition (log price around the banner base), from the generator's own pieces
    J, T = len(st), cfg.weeks
    full = sw.merge(tr, on=["store", "week"])
    lnp_dev = np.log(full.price) - np.log(np.asarray(theta.p)[[dm.BANNERS.index(b) for b in full.banner]])
    parts = {"cost shocks (instrument)": cfg.pass_through * full.cost_shock,
             "demand response (endogeneity)": cfg.price_on_xi * full.xi_t,
             "national flyers": -cfg.flyer_depth * full.flyer, "local promotions": -cfg.promo_depth * full.local_promo}
    parts["other noise"] = lnp_dev - sum(parts.values())
    var = {k: float(np.var(v)) for k, v in parts.items()}
    tot = float(np.var(lnp_dev))
    summary = [("store-week rows (observed)", len(sw)), ("category rows", len(cat)), ("stores", J), ("weeks", T),
               ("first week", str(wk.date.iloc[0].date())), ("last week", str(wk.date.iloc[-1].date())),
               ("missing store-weeks, %", 100 * (1 - len(sw) / (J * T))),
               ("price sd around banner base, %", 100 * np.sqrt(tot))]
    summary += [(f"price variance share: {k}, %", 100 * v / tot) for k, v in var.items()]
    pd.DataFrame(summary, columns=["item", "value"]).to_csv(OUT / "2_1_panel_summary.csv", index=False)

    # validation 1: at base prices, no shocks, M = 1, the twin's demand equals the Pillar 1 model's
    wd = tw.make_week_demand(a)
    sbn = np.asarray(a["store_banner"])
    q_twin, _ = wd(theta, jnp.asarray(np.asarray(theta.p)[sbn]), jnp.zeros(J), float(theta.p_abroad))
    q_p1, _ = dm.quantities(theta, a)
    gap = float(np.abs(np.asarray(q_twin) - np.asarray(q_p1)).max())

    # validation 2 (preview of T2): log baskets on log price with store and week fixed effects
    d = sw.assign(lq=np.log(sw.baskets), lp=np.log(sw.price), z=sw.cost_index,
                  fl=sw.flyer.astype(float), pr=sw.local_promo.astype(float))
    dd = two_way_demean(d, ["lq", "lp", "z", "fl", "pr"])
    X = dd[["lp", "fl", "pr"]].to_numpy()
    b_ols = np.linalg.lstsq(X, dd.lq.to_numpy(), rcond=None)[0][0]
    Z = dd[["z", "fl", "pr"]].to_numpy()
    Xhat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    b_iv = np.linalg.lstsq(Xhat, dd.lq.to_numpy(), rcond=None)[0][0]
    w = truth.baskets_week
    e_true = float(truth.own_elasticity.mean())
    val = pd.DataFrame([
        ("max |twin − Pillar 1| baskets per store, base week", gap, "should be ~0"),
        ("true own-price elasticity, mean over stores", e_true, "answer key"),
        ("true own-price elasticity, basket-weighted", float(np.average(truth.own_elasticity, weights=w)), "answer key"),
        ("naive OLS (store + week FE)", b_ols, "biased towards 0: prices rise when demand is strong"),
        ("IV with cost index (store + week FE)", b_iv, "cost shocks are partly regional, so sister stores move too"),
    ], columns=["check", "value", "comment"])
    val.to_csv(OUT / "2_1_validation.csv", index=False)
    print(val.round(3).to_string(index=False))

    figures(sw, wk, st, truth, stores, var, tot, b_ols, b_iv, e_true, cfg)


def figures(sw, wk, st, truth, stores, var, tot, b_ols, b_iv, e_true, cfg):
    viz.setup()
    # 1. weekly baskets by player, indexed to the first-year mean
    pl = {b: p for p, bs in cb.PLAYERS.items() for b in bs}
    g = sw.assign(player=sw.banner.map(pl)).groupby(["date", "player"]).baskets.sum().unstack()
    g = g / g.iloc[:52].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, colr in zip(["Migros", "Coop", "Denner", "Lidl"], [viz.BLUE, viz.ORANGE, viz.AQUA, viz.MUTED]):
        ax.plot(g.index, g[name] * 100, color=colr, linewidth=1.6, label=name)
    ax.plot(wk.date, wk.q_abroad_true / wk.q_abroad_true.iloc[:52].mean() * 100, color=viz.INK_2, linewidth=1.2,
            linestyle="--", label="abroad (true)")
    ax.set_ylabel("weekly baskets, index (year-1 mean = 100)")
    ax.legend(ncol=5, loc="upper left", frameon=False)
    ax.set_title("The twin has Christmas peaks, summer dips and week-to-week noise [sim]")
    viz.source(ax, "Observed baskets (with scanner noise and data gaps) summed by player. Simulated, not real sales.")
    viz.save(fig, 2, "2_1", "weekly_sales")

    # 2. where price variation comes from
    fig, ax = plt.subplots(figsize=(8, 3.4))
    names = list(var)
    shares = [100 * var[k] / tot for k in names]
    cols = [viz.AQUA, viz.ORANGE, viz.BLUE, viz.BLUE, viz.MUTED]
    ax.barh(names[::-1], shares[::-1], color=cols[::-1], height=0.6)
    for i, v in enumerate(shares[::-1]):
        ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=9, color=viz.INK)
    ax.set_xlabel("share of the variance of log price around its banner base, %")
    ax.set_xlim(0, max(shares) * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"About {100 * var['demand response (endogeneity)'] / tot:.0f}% of price movement is managers "
                 "reacting to demand: the trap for OLS")
    viz.source(ax, f"Store-week log price; price sd {100 * np.sqrt(tot):.1f}% around the banner price. [sim]")
    viz.save(fig, 2, "2_1", "price_components")

    # 3. category seasonality heatmap (one year)
    S = tw.category_season(tw.season(pd.DatetimeIndex(wk.date)))[:52]
    names = [c[0].replace("_", " ") for c in tw.CATEGORIES]
    o = np.argsort(-np.abs(S - 1).max(0))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow((S[:, o].T - 1) * 100, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -60, 60))
    ax.set_yticks(range(len(o)), [names[i] for i in o], fontsize=8.5)
    ticks = np.arange(0, 52, 4)
    ax.set_xticks(ticks, [wk.date.iloc[i].strftime("%d %b") for i in ticks], fontsize=8)
    ax.grid(False)
    cb_ = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, ticks=[-60, -30, 0, 30, 60])
    cb_.set_label("demand vs normal week, %", fontsize=8.5)
    cb_.outline.set_visible(False)
    ax.set_title("Chocolate and baking spike at Christmas and Easter; drinks and ice cream in summer [sim]")
    viz.source(ax, "Category multipliers before normalising within the basket; 2022 shown.")
    viz.save(fig, 2, "2_1", "category_season")

    # 4. map: true own-price elasticity of every store
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cantons = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)
    xy = gpd.GeoSeries(gpd.points_from_xy(stores.lon, stores.lat), crs=4326).to_crs(2056)
    v = -truth.own_elasticity.to_numpy()
    bins = np.round(np.quantile(v, np.linspace(0, 1, 7)), 1)
    bins[0], bins[-1] = v.min() - 0.01, v.max() + 0.01
    cmap, norm = ListedColormap(BLUES), BoundaryNorm(bins, len(BLUES))
    fig, ax = plt.subplots(figsize=(9, 6))
    viz.map_axes(ax, cantons)
    o = np.argsort(v)
    ax.scatter(xy.x.to_numpy()[o], xy.y.to_numpy()[o], c=v[o], cmap=cmap, norm=norm, s=4, linewidths=0, zorder=2)
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal",
                       fraction=0.04, pad=0.02, aspect=45, ticks=bins[1:-1])
    cb_.ax.set_xticklabels([f"{b:.1f}" for b in bins[1:-1]], fontsize=8)
    cb_.set_label("true own-price elasticity (absolute value): % baskets lost per 1% price rise", fontsize=8.5)
    cb_.outline.set_visible(False)
    ax.set_title("The answer key: how price-sensitive each store's shoppers truly are [sim]", loc="left")
    viz.source(ax, f"Stores = real OSM locations; elasticity from the Pillar 1 model + simulated store quality. "
                   f"Mean {-truth.own_elasticity.mean():.2f}.")
    viz.save(fig, 2, "2_1", "elasticity_map")

    # 5. endogeneity preview
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels = ["naive OLS", "IV: cost index", "truth (mean store)"]
    vals = [b_ols, b_iv, e_true]
    ax.barh(labels[::-1], vals[::-1], color=[viz.INK_2, viz.AQUA, viz.MUTED][::-1], height=0.55)
    for i, x in enumerate(vals[::-1]):
        ax.text(x - 0.08, i, f"{x:.2f}", va="center", ha="right", fontsize=9.5, color=viz.INK)
    ax.axvline(0, color=viz.INK_2, linewidth=0.8)
    ax.set_xlim(min(vals) * 1.25, 0.2)
    ax.set_xlabel("own-price elasticity of store baskets (store and week fixed effects)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Naive regression on the twin misses the true elasticity; the cost instrument gets closer")
    viz.source(ax, "Preview of test T2; the full comparison (DML, causal forest) follows in 2.3-2.4. [sim]")
    viz.save(fig, 2, "2_1", "endogeneity")


if __name__ == "__main__":
    main()
