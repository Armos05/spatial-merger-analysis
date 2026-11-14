"""Pillar 3 step 2: how much better does the merged firm know each shopper? Cumulus only vs Cumulus + Supercard. [sim]

Uses the experiments and α̂ of 3.1. Metrics for each data regime, against the true α_i:
    correlation, RMSE / mean α, precision at the top 20% (share of the truly most price-sensitive fifth that the
    firm puts in its own top fifth) and calibration by decile.
Learning curve (Migros experiment): features from only the first 13, 26, 52 or 156 weeks of card history,
own card vs pooled.

Run:  conda activate keystone && python scripts/31_prediction.py      (needs 30)
Outputs (assets/p3/m2/):
  3_2_quality.csv          accuracy by regime × household group (both cards / one card) and segment
  3_2_learning.csv         accuracy by weeks of history, own vs pooled
  3_2_quality.png          calibration by decile + precision, Cumulus only vs pooled
  3_2_learning.png         learning curve
  3_2_quality_map.png      by home canton: share of households holding both cards and gain in correlation
"""
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

T30 = runpy.run_path(str(ROOT / "scripts/30_experiments.py"), run_name="lib")
P3, INTERIM, EV = ROOT / "data/processed/p3", ROOT / "data/interim", ROOT / "data/processed/twin/events"
OUT = ROOT / "assets/p3/m2"
BLUES = T30["BLUES"]
SEG_ORDER = T30["SEG_ORDER"]


def metrics(a_hat, a_true):
    top_t = a_true >= np.quantile(a_true, 0.8)
    top_h = a_hat >= np.quantile(a_hat, 0.8)
    return {"n": len(a_true), "corr": np.corrcoef(a_hat, a_true)[0, 1],
            "rmse_rel": np.sqrt(((a_hat - a_true) ** 2).mean()) / a_true.mean(),
            "precision_top20": (top_t & top_h).sum() / top_h.sum()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = pd.read_parquet(P3 / "alpha_hat.parquet")
    rows = []
    for side, card in (("M", "Cumulus"), ("C", "Supercard")):
        m = r[f"alpha_own_card_{side}"].notna()
        both = r.cumulus & r.supercard
        for grp, gm in (("all holders", m), ("hold both cards", m & both), ("hold only this card", m & ~both)):
            for reg in ("location_only", "own_card", "pooled"):
                rows.append({"chain": card, "group": grp, "regime": reg,
                             **metrics(r.loc[gm, f"alpha_{reg}_{side}"].to_numpy(), r.loc[gm, "alpha_true"].to_numpy())})
        for seg in SEG_ORDER:
            gm = m & (r.segment == seg)
            for reg in ("own_card", "pooled"):
                x, t = r.loc[gm, f"alpha_{reg}_{side}"], r.loc[gm, "alpha_true"]
                rows.append({"chain": card, "group": f"segment {seg}", "regime": reg, "n": int(gm.sum()),
                             "corr": np.corrcoef(x, t)[0, 1], "rmse_rel": np.sqrt(((x - t) ** 2).mean()) / t.mean(),
                             "mean_hat_over_true": x.mean() / t.mean()})
    q = pd.DataFrame(rows)
    q.round(4).to_csv(OUT / "3_2_quality.csv", index=False)
    print(q[q.group.isin(["all holders", "hold both cards", "hold only this card"])].round(3).to_string(index=False))

    # learning curve (Migros experiment)
    S = T30["setup"]()
    XL = T30["location_features"](S)
    lc = []
    full_c = pd.read_parquet(EV / "cumulus.parquet")
    full_s = pd.read_parquet(EV / "supercard.parquet")
    e = r[r.alpha_own_card_M.notna()]
    for weeks in (4, 13, 26, 52, 156):
        XM = feats(full_c, weeks, "M_")
        XC = feats(full_s, weeks, "C_")
        X = XL.join(XM).join(XC)
        X["M_share_of_card_trips"] = X.M_trips_wk.fillna(0) / (X.M_trips_wk.fillna(0) + X.C_trips_wk.fillna(0))
        for reg, cl in (("own card", list(XL.columns) + list(XM.columns)), ("pooled", list(X.columns))):
            tau, _ = pe.t_learner(X.loc[e.index, cl].to_numpy(float), e.y_M.to_numpy(), e.treated_M.to_numpy().astype(int),
                                  seed=T30["SEED"])
            fill = pd.Series(np.nanmedian(tau), index=r.index)
            fill.loc[e.index] = tau
            al = pe.invert_alpha(S["theta"], S["ahB"], S["pre"], jnp.asarray(fill.to_numpy()), "M", T30["DISCOUNT"])
            al = pd.Series(al, index=r.index).loc[e.index]
            both = (e.cumulus & e.supercard).to_numpy()
            lc.append({"weeks": weeks, "regime": reg, **metrics(al.to_numpy(), e.alpha_true.to_numpy()),
                       "corr_both_cards": np.corrcoef(al.to_numpy()[both], e.alpha_true.to_numpy()[both])[0, 1]})
            print(lc[-1], flush=True)
    lc = pd.DataFrame(lc)
    lc.round(4).to_csv(OUT / "3_2_learning.csv", index=False)
    figures(r, q, lc)


def feats(c, weeks, prefix):
    sw = pd.read_parquet(EV / "store_week.parquet", columns=["store_id", "week", "price", "flyer"])
    sw["rel_price"] = np.log(sw.price / sw.groupby("store_id").price.transform("mean"))
    c = c[c.week < weeks].merge(sw[["store_id", "week", "rel_price", "flyer"]], on=["store_id", "week"], how="left")
    c["conv"] = c.banner.isin(["Migrolino", "Coop Pronto"])
    c["lspend"] = np.log(c.spend)
    g = c.groupby("card_id")
    f = pd.DataFrame({"trips_wk": g.size() / weeks, "mean_lspend": g.lspend.mean(), "sd_lspend": g.lspend.std(),
                      "fresh": g.dept_fresh.mean(), "staples": g.dept_staples.mean(), "treats": g.dept_treats.mean(),
                      "conv_share": g.conv.mean(), "n_stores": g.store_id.nunique(), "flyer_share": g.flyer.mean(),
                      "rel_price_paid": g.rel_price.mean(), "weeks_active": g.week.nunique() / weeks})
    return f.add_prefix(prefix)


def figures(r, q, lc):
    viz.setup()
    # 1. calibration by decile of α̂ (Migros) + precision
    m = r.alpha_own_card_M.notna() & r.supercard
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    for reg, lab, colr in (("location_only", "location only", viz.MUTED), ("own_card", "Cumulus only", viz.ORANGE),
                           ("pooled", "Cumulus + Supercard", viz.BLUE)):
        x = r.loc[m, f"alpha_{reg}_M"]
        dec = pd.qcut(x.rank(method="first"), 10, labels=False)
        v = r.loc[m, "alpha_true"].groupby(dec.to_numpy()).mean()
        ax.plot(np.arange(1, 11), 100 * v.to_numpy(), marker="o", color=colr, label=lab)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("decile of the firm's predicted price sensitivity α̂")
    ax.set_ylabel("true mean α (× 100, per CHF)")
    ax.legend(fontsize=8.5)
    ax.set_title("Households holding both cards: true sensitivity by predicted decile", fontsize=10.5)
    g = q[(q.group == "hold both cards")].set_index(["chain", "regime"])
    x = np.arange(2)
    for k, (reg, lab, colr) in enumerate((("location_only", "location only", viz.MUTED),
                                          ("own_card", "own card", viz.ORANGE), ("pooled", "pooled", viz.BLUE))):
        v = [100 * g.loc[(c, reg), "precision_top20"] for c in ("Cumulus", "Supercard")]
        ax2.bar(x + (k - 1) * 0.27, v, 0.26, color=colr, label=lab)
        for xi, vi in zip(x + (k - 1) * 0.27, v):
            ax2.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=8)
    ax2.axhline(20, color=viz.INK_2, lw=0.8, ls="--")
    ax2.text(1.45, 21, "random = 20", fontsize=8, color=viz.INK_2, ha="right")
    ax2.set_xticks(x, ["Migros side", "Coop side"])
    ax2.set_ylabel("% of the truly most sensitive fifth\nfound in the predicted top fifth")
    ax2.set_title("Precision at the top 20%", fontsize=10.5)
    ax2.set_axisbelow(True)
    gm, gc = g.loc[("Cumulus", "own_card"), "corr"], g.loc[("Cumulus", "pooled"), "corr"]
    fig.suptitle(f"Pooling the cards sharpens the picture of each shopper: correlation {gm:.2f} → {gc:.2f} "
                 f"for households with both cards [sim]", x=0.02, ha="left", fontsize=12, fontweight="bold")
    viz.source(ax, "α̂ from the 3.1 coupon tests: T-learner on each data regime + inversion of the firm's demand model. "
                   "Households holding both cards and a reachable Migros store. [sim]")
    viz.save(fig, 3, "3_2", "quality")

    # 2. learning curve
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for reg, colr in (("own card", viz.ORANGE), ("pooled", viz.BLUE)):
        d = lc[lc.regime == reg]
        ax.plot(d.weeks, d.corr_both_cards, marker="o", color=colr, label=f"{reg} (households with both cards)")
        ax.plot(d.weeks, d["corr"], marker="o", color=colr, ls="--", alpha=0.6, label=f"{reg} (all Cumulus holders)")
    ax.set_xscale("log")
    ax.set_xticks([4, 13, 26, 52, 156], ["4", "13", "26", "52", "156"])
    ax.set_xlabel("weeks of card history used")
    ax.set_ylabel("correlation of α̂ with true α")
    ax.legend(fontsize=8.5)
    d = lc.set_index(["weeks", "regime"]).corr_both_cards
    ax.set_title(f"Pooled data get there faster: {d.loc[(13, 'pooled')]:.2f} after one quarter vs "
                 f"{d.loc[(13, 'own card')]:.2f} with Cumulus alone [sim]", fontsize=11)
    viz.source(ax, "Migros coupon experiment (3.1); features from the first k weeks only. [sim]")
    viz.save(fig, 3, "3_2", "learning")

    # 3. map
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cant = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)[["ref", "geometry"]]
    hh = pd.read_parquet(EV / "households.parquet").set_index("hh_id")
    rr = r.join(hh[["canton"]])
    both = rr.groupby("canton").apply(lambda g: 100 * (g.cumulus & g.supercard).mean(), include_groups=False)
    mm = rr[rr.alpha_own_card_M.notna()]
    gain = mm.groupby("canton").apply(lambda g: np.corrcoef(g.alpha_pooled_M, g.alpha_true)[0, 1]
                                      - np.corrcoef(g.alpha_own_card_M, g.alpha_true)[0, 1], include_groups=False)
    gdf = cant.merge(pd.DataFrame({"both": both, "gain": gain}), left_on="ref", right_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, ttl, lab, bins in (
            (axes[0], "both", "Households holding both cards", "% of panel households", None),
            (axes[1], "gain", "Gain in prediction from pooling (Migros side)", "Δ correlation, pooled − Cumulus only", None)):
        v = gdf[col]
        b = np.linspace(np.floor(v.min() * 100) / 100, np.ceil(v.max() * 100) / 100, 7) if col == "gain" else \
            np.linspace(np.floor(v.min()), np.ceil(v.max()), 7)
        cmap, norm = ListedColormap(BLUES), BoundaryNorm(b, len(BLUES))
        gdf.plot(ax=ax, column=col, cmap=cmap, norm=norm, edgecolor=viz.SURFACE, linewidth=0.6)
        viz.map_axes(ax, cant)
        ax.set_title(ttl, fontsize=10.5)
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal",
                          fraction=0.05, pad=0.03, shrink=0.7)
        cb.set_label(lab, fontsize=8.5)
        cb.set_ticks(b)
        cb.set_ticklabels([f"{x:.2f}" if col == "gain" else f"{x:.0f}" for x in b])
    fig.suptitle(f"Pooling helps in almost every canton; {100 * (r.cumulus & r.supercard).mean():.0f}% of households hold "
                 f"both cards [sim]", x=0.02, ha="left", fontsize=12, fontweight="bold")
    viz.source(axes[0], "Card holding is random in the panel (80% Cumulus, 70% Supercard) [assumed]; canton gains are "
                        "noisy in small cantons. [sim]")
    viz.save(fig, 3, "3_2", "quality_map")


if __name__ == "__main__":
    if "--figures" in sys.argv:
        figures(pd.read_parquet(P3 / "alpha_hat.parquet"), pd.read_csv(OUT / "3_2_quality.csv"),
                pd.read_csv(OUT / "3_2_learning.csv"))
    else:
        main()
