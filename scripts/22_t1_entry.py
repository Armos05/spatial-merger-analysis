"""Pillar 2 step 3: test T1, entry analysis on the twin [sim]. How much do nearby stores lose when a discounter
opens, and where do the entrant's sales come from? Three estimators, scored against the answer key from 2.2:
traditional store-month TWFE, a stacked event study with clean controls, and a pooled exposure regression.
Each also runs on noise-free data (true baskets) to separate design bias from noise.

The analyst sees only: the observed store-week panel of the events world (noise, gaps), store locations, banners,
and the list of openings with dates. Nothing hidden is used for estimation.

Run:  conda activate keystone && python scripts/22_t1_entry.py      (needs 21)
Outputs (assets/p2/m3/):
  2_3_estimates.csv         loss % by distance band × group: truth and the three methods (95% intervals), both data
  2_3_scorecard.csv         bias, RMSE, 95% coverage, interval width per method and data
  2_3_diversion.csv         share of the entrant's sales taken from each group within 10 km: truth vs methods
  2_3_event_study.csv       weekly event-time effects, stores within 3 km
  2_3_per_entrant.csv       single-opening DiD within 3 km vs truth
  2_3_estimates_vs_truth.png, 2_3_event_study.png, 2_3_scorecard.png, 2_3_diversion.png, 2_3_entrant_map.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from mxc import viz  # noqa: E402
from mxc.causal import entry_pooled as ep  # noqa: E402
from mxc.structural import calibrate as cb  # noqa: E402
from mxc.structural import demand as dm  # noqa: E402

PROCESSED, INTERIM = ROOT / "data/processed", ROOT / "data/interim"
EV = PROCESSED / "twin/events"
OUT = ROOT / "assets/p2/m3"
BANDS = (0, 1, 2, 3, 5, 10)
GROUPS = ["Aldi / Lidl", "Migros group", "Coop group", "Others"]
PLAYER = {b: p for p, bs in cb.PLAYERS.items() for b in bs}
GMAP = {"Migros": "Migros group", "Denner": "Migros group", "Coop": "Coop group", "Aldi": "Aldi / Lidl",
        "Lidl": "Aldi / Lidl", "Others": "Others"}
BLUES = ["#d6e6f8", "#a9cbf0", "#6fa6e6", "#2a78d6", "#1d58a3", "#123a6e"]
COL = {"TWFE (store-month)": viz.INK_2, "stacked event study": viz.ORANGE, "pooled exposure": viz.BLUE}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(PROCESSED / "market.npz", allow_pickle=True)
    store_id, sbn = z["store_id"], z["store_banner"]
    group = np.array([GMAP[PLAYER[dm.BANNERS[i]]] for i in sbn])
    stores = pd.read_parquet(PROCESSED / "stores.parquet", columns=["store_id", "lon", "lat"]).set_index("store_id").loc[store_id]
    pts = gpd.GeoSeries(gpd.points_from_xy(stores.lon, stores.lat), crs=4326).to_crs(2056)
    xy = np.c_[pts.x, pts.y] / 1000
    panel = pd.read_parquet(EV / "store_week.parquet", columns=["store", "week", "date", "baskets", "revenue"])
    ev = pd.read_csv(ROOT / "assets/p2/m2/2_2_events.csv", parse_dates=["date"])
    ent = ev[ev.event == "entry"][["store", "week", "date"]].reset_index(drop=True)
    closed = ev[ev.event == "closure"].store.to_numpy()

    # answer key on the same cells: unweighted mean store loss (matches unweighted log regressions)
    et = pd.read_parquet(EV / "entry_truth.parquet")
    et = et[~et.store.isin(closed) & (et.dist_km <= BANDS[-1])]
    et["band"] = pd.cut(et.dist_km, BANDS, labels=[f"{a}-{b} km" for a, b in zip(BANDS[:-1], BANDS[1:])])
    truth = et.groupby(["band", "group"], observed=True).loss_pct.mean().rename("truth_loss_pct").reset_index()
    truth["band"] = truth.band.astype(str)

    # estimation: observed data (what an analyst gets) and noise-free data (true baskets, same gaps)
    tw_ = pd.read_parquet(EV / "truth_week.parquet", columns=["store", "week", "q_true"])
    clean = panel.merge(tw_, on=["store", "week"])
    clean = clean.assign(revenue=clean.q_true * clean.revenue / clean.baskets, baskets=clean.q_true)
    est, extra = [], {}
    for data, pnl in [("observed", panel), ("noise-free", clean)]:
        static, dynamic, n_rows, per = ep.stacked_event_study(pnl, ent, xy, group, BANDS, GROUPS, exclude=closed)
        twfe = ep.twfe_monthly(pnl, ent, xy, group, BANDS, GROUPS, exclude=closed)
        expo = ep.exposure_regression(pnl, ent, xy, group, groups=GROUPS, exclude=closed)
        est += [ep.to_loss(twfe).assign(method="TWFE (store-month)", data=data),
                ep.to_loss(static).assign(method="stacked event study", data=data),
                ep.to_loss(expo).assign(method="pooled exposure", data=data)]
        extra[data] = (dynamic, per, n_rows)
        print(f"{data}: stacked rows {n_rows:,}")
    est = pd.concat(est).merge(truth, on=["band", "group"], how="left")
    est.round(4).to_csv(OUT / "2_3_estimates.csv", index=False)
    dynamic, per, _ = extra["observed"]
    est_all = est
    est = est[est.truth_loss_pct.notna()]

    def score(g):
        err = g.loss_pct - g.truth_loss_pct
        return pd.Series({"cells": len(g), "bias_pp": err.mean(), "rmse_pp": np.sqrt((err ** 2).mean()),
                          "coverage_95": ((g.lo95 <= g.truth_loss_pct) & (g.truth_loss_pct <= g.hi95)).mean(),
                          "mean_ci_width_pp": (g.hi95 - g.lo95).mean(),
                          "rmse_within_3km_pp": np.sqrt((err[g.band.isin(["0-1 km", "1-2 km", "2-3 km"])] ** 2).mean()),
                          "bias_within_3km_pp": err[g.band.isin(["0-1 km", "1-2 km", "2-3 km"])].mean()})
    sc = est.groupby(["data", "method"]).apply(score, include_groups=False)
    sc.round(4).to_csv(OUT / "2_3_scorecard.csv")
    print(sc.round(3).to_string())
    est = est[est.data == "observed"]

    # diversion to the entrant within 10 km: Σ_k loss_k × pre-opening baskets_k / entrant's baskets after opening
    pre = {}
    for e, w0 in zip(ent.store, ent.week):
        d = np.hypot(*(xy - xy[e]).T)
        win = panel[(panel.week >= w0 - 26) & (panel.week < w0)]
        k = np.where((d <= BANDS[-1]) & ~np.isin(np.arange(len(d)), np.r_[ent.store, closed]))[0]
        q = win[win.store.isin(k)].groupby("store").baskets.mean()
        pre[e] = pd.DataFrame({"store": q.index, "q_pre": q.values,
                               "band": [f"{BANDS[i]}-{BANDS[i + 1]} km" for i in np.digitize(d[q.index], BANDS, right=True) - 1],
                               "group": group[q.index]})
    pre = pd.concat(pre, names=["entrant"]).reset_index(level=0)
    q_ent = sum(panel[(panel.store == e) & (panel.week >= w0) & (panel.week < w0 + 26)].baskets.mean()
                for e, w0 in zip(ent.store, ent.week))
    div = []
    for m, g in est.groupby("method"):
        lk = pre.merge(g[["band", "group", "loss_pct", "lo95", "hi95"]], on=["band", "group"])
        for grp, h in lk.groupby("group"):
            div.append({"method": m, "group": grp, "share": (h.loss_pct / 100 * h.q_pre).sum() / q_ent,
                        "lo95": (h.lo95 / 100 * h.q_pre).sum() / q_ent, "hi95": (h.hi95 / 100 * h.q_pre).sum() / q_ent})
    src = pd.read_csv(ROOT / "assets/p2/m2/2_2_entry_sources.csv")
    et10 = et.groupby("group").apply(lambda h: -h.dq.sum(), include_groups=False) / src.q_entrant.sum()
    div += [{"method": "truth", "group": g, "share": v, "lo95": np.nan, "hi95": np.nan} for g, v in et10.items()]
    div = pd.DataFrame(div)
    div.round(4).to_csv(OUT / "2_3_diversion.csv", index=False)
    print(div.pivot(index="group", columns="method", values="share").round(3))

    dynamic = ep.to_loss(dynamic)
    dynamic.round(4).to_csv(OUT / "2_3_event_study.csv", index=False)
    t3 = et[et.dist_km <= 3].groupby("entrant").loss_pct.mean().rename("truth_loss_pct")
    per = per.set_index("entrant").join(t3)
    per.insert(0, "store_id", store_id[per.index])
    per.round(4).to_csv(OUT / "2_3_per_entrant.csv")
    truth3 = float(et[et.dist_km <= 3].loss_pct.mean())
    print("per-entrant corr", per[["did_loss_pct", "truth_loss_pct"]].corr().iloc[0, 1].round(3))

    figures(est, sc, div, dynamic, per, truth3, xy)


def figures(est, sc, div, dynamic, per, truth3, xy):
    viz.setup()
    bands = [f"{a}-{b} km" for a, b in zip(BANDS[:-1], BANDS[1:])]
    x = np.arange(len(bands))
    # 1. estimates vs truth, one panel per group
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, g in zip(axes, GROUPS):
        e = est[est.group == g]
        tr = e.drop_duplicates("band").set_index("band").truth_loss_pct.reindex(bands)
        ax.plot(x, tr.values, color=viz.AQUA, linewidth=2.2, label="truth", zorder=3)
        for off, (m, colr) in zip([-0.22, 0.0, 0.22], COL.items()):
            h = e[e.method == m].set_index("band").reindex(bands)
            ax.errorbar(x + off, h.loss_pct, yerr=[h.loss_pct - h.lo95, h.hi95 - h.loss_pct], fmt="o", color=colr,
                        markersize=4.5, capsize=0, linewidth=1.2, label=m)
        ax.axhline(0, color=viz.INK_2, linewidth=0.7)
        ax.set_xticks(x, [b.replace(" km", "") for b in bands], fontsize=8.5)
        ax.set_title(g, fontsize=10.5)
        ax.set_xlabel("km from the new store", fontsize=8.5)
    axes[0].set_ylabel("loss of baskets, %")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    o = sc.loc["observed"]
    fig.suptitle(f"Within 3 km the pooled exposure regression is closest (RMSE {o.loc['pooled exposure', 'rmse_within_3km_pp']:.2f} pp; "
                 f"stacked {o.loc['stacked event study', 'rmse_within_3km_pp']:.2f}, TWFE {o.loc['TWFE (store-month)', 'rmse_within_3km_pp']:.2f}) [sim]",
                 x=0.01, ha="left", fontweight="bold")
    viz.source(axes[0], "Bars: 95% intervals, clustered (stacked: by opening; others: by store). Truth = 2.2 answer key.")
    viz.save(fig, 2, "2_3", "estimates_vs_truth")

    # 2. event study
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    d = pd.concat([dynamic, pd.DataFrame({"rel_week": [-1], "loss_pct": [0.0], "lo95": [0.0], "hi95": [0.0]})]).sort_values("rel_week")
    ax.fill_between(d.rel_week, d.lo95, d.hi95, color=BLUES[1], linewidth=0)
    ax.plot(d.rel_week, d.loss_pct, color=viz.BLUE, marker="o", markersize=3, linewidth=1.4, label="stacked estimate")
    ax.hlines(truth3, 0, d.rel_week.max(), color=viz.AQUA, linewidth=2, label=f"truth ({truth3:.2f}%)")
    ax.hlines(0, d.rel_week.min(), -1, color=viz.AQUA, linewidth=2)
    ax.axvline(-0.5, color=viz.ORANGE, linewidth=1)
    ax.set_xlabel("weeks since the discounter opened")
    ax.set_ylabel("loss of baskets, %  (stores within 3 km)")
    ax.legend(frameon=False, loc="upper left")
    post = d[d.rel_week >= 0].loss_pct.mean()
    prem = d[d.rel_week < -1].loss_pct.mean()
    ax.set_title(f"A clear step at opening, but pre-weeks drift ({prem:+.1f}%) and the level overshoots: "
                 f"{post:.2f}% vs truth {truth3:.2f}%", fontsize=10.5)
    viz.source(ax, "Event-time effects relative to week −1; 95% band clustered by opening. [sim]")
    viz.save(fig, 2, "2_3", "event_study")

    # 3. scorecard: observed (solid) vs noise-free (hatched)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    methods = list(COL)
    yy = np.arange(len(methods))
    for ax, (col, lab, fmt) in zip(axes, [("bias_within_3km_pp", "bias within 3 km, pp", "{:+.2f}"),
                                          ("rmse_pp", "RMSE, all 20 cells, pp", "{:.2f}"),
                                          ("coverage_95", "95% coverage", "{:.0%}")]):
        for j, (data, hatch) in enumerate([("observed", None), ("noise-free", "///")]):
            v = sc.loc[data].reindex(methods)[col]
            ax.barh(yy + (j - 0.5) * 0.36, v.values, height=0.36, color=[COL[m] for m in methods],
                    hatch=hatch, edgecolor=viz.SURFACE, linewidth=0, alpha=1 if j == 0 else 0.55,
                    label=data if col == "bias_within_3km_pp" else None)
            for i, val in enumerate(v.values):
                ax.text(val, yy[i] + (j - 0.5) * 0.36, " " + fmt.format(val) + " ", va="center", fontsize=8,
                        ha="left" if val >= 0 else "right")
        ax.axvline(0, color=viz.INK_2, linewidth=0.7)
        if col == "coverage_95":
            ax.axvline(0.95, color=viz.AQUA, linewidth=1, linestyle="--")
            ax.set_xlim(0, 1.2)
        else:
            lim = np.abs(sc[col]).max() * 1.45
            ax.set_xlim(-lim if (sc[col] < 0).any() else 0, lim)
        ax.set_yticks(yy, methods if col == "bias_within_3km_pp" else [""] * len(methods))
        ax.invert_yaxis()
        ax.set_title(lab, fontsize=10.5)
        ax.grid(axis="y", visible=False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("T1 scorecard vs the answer key: noise-free data barely help, so the biases come from design [sim]",
                 x=0.01, ha="left", fontweight="bold")
    viz.save(fig, 2, "2_3", "scorecard")

    # 4. diversion to the entrant
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ms = ["truth"] + list(COL)
    cols = [viz.AQUA] + list(COL.values())
    y0 = np.arange(len(GROUPS))
    for i, (m, colr) in enumerate(zip(ms, cols)):
        h = div[div.method == m].set_index("group").reindex(GROUPS)
        err = None if m == "truth" else [100 * (h.share - h.lo95), 100 * (h.hi95 - h.share)]
        ax.barh(y0 + (i - 1.5) * 0.2, 100 * h.share, height=0.2, color=colr, xerr=err,
                error_kw={"ecolor": viz.INK, "linewidth": 0.9}, label=m)
    ax.set_yticks(y0, GROUPS)
    ax.invert_yaxis()
    ax.set_xlabel("share of the new store's baskets taken from stores within 10 km, %")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    sh = div.pivot(index="group", columns="method", values="share").loc["Migros group"]
    ax.set_title(f"Share of an entrant's sales from Migros group within 10 km: truth {sh['truth']:.0%}, "
                 f"exposure {sh['pooled exposure']:.0%}, stacked {sh['stacked event study']:.0%}, TWFE {sh['TWFE (store-month)']:.0%}",
                 fontsize=10.5)
    viz.source(ax, "Estimated losses × each store's pre-opening baskets ÷ entrant's first-half-year baskets. [sim]")
    viz.save(fig, 2, "2_3", "diversion")

    # 5. map: truth vs single-opening estimate
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cantons = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)
    vals = np.r_[per.truth_loss_pct, per.did_loss_pct]
    bins = [min(-0.01, vals.min() - 0.01), 0, 0.5, 1, 2, 4, max(4.01, vals.max() + 0.01)]
    cmap = ListedColormap(["#f6c7c6"] + BLUES[1:])
    norm = BoundaryNorm(bins, cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, col, ttl in zip(axes, ["truth_loss_pct", "did_loss_pct"], ["Truth", "Single-opening DiD estimate"]):
        viz.map_axes(ax, cantons)
        ax.scatter(xy[per.index, 0] * 1000, xy[per.index, 1] * 1000, c=per[col], cmap=cmap, norm=norm, s=30,
                   edgecolors=viz.SURFACE, linewidths=0.6, zorder=3)
        ax.set_title(ttl, fontsize=10.5)
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, orientation="horizontal",
                       fraction=0.05, pad=0.03, aspect=50, ticks=[0, 0.5, 1, 2, 4])
    cb_.set_label("mean loss of stores within 3 km of each opening, %  (pink = estimated gain)", fontsize=8.5)
    cb_.outline.set_visible(False)
    r = per[["did_loss_pct", "truth_loss_pct"]].corr().iloc[0, 1]
    neg = (per.did_loss_pct < 0).mean()
    fig.suptitle(f"One opening at a time is noise: correlation with the truth {r:.2f}, {neg:.0%} show a gain [sim]",
                 x=0.01, y=0.98, ha="left", fontweight="bold")
    viz.save(fig, 2, "2_3", "entrant_map")


if __name__ == "__main__":
    main()
