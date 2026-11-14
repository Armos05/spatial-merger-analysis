"""Pillar 2 step 4: test T2, own-price elasticities on the twin [sim]. Four estimators on the observed 2.1 panel,
scored against the answer key (true store elasticities): OLS-FE, IV-FE (cost index), DML-PLR (ML controls, no
instrument), DML-PLIV (ML controls + cost instrument). Cells: national, 12 banners, 26 cantons.

The analyst sees: store-week price, baskets, flyer and promotion flags, cost index, banner, canton. Not the demand
shocks, not true baskets, not store quality.

Run:  conda activate keystone && python scripts/23_t2_elasticity.py      (needs 20)
Outputs (assets/p2/m4/):
  2_4_estimates.csv       θ by method × level (national / banner / canton) with 95% intervals and truth
  2_4_scorecard.csv       bias, RMSE, 95% coverage per method and level
  2_4_nuisance.csv        out-of-fold R² of the ML nuisance models; first-stage strength
  2_4_national.png, 2_4_banner.png, 2_4_scorecard.png, 2_4_canton_map.png
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
from mxc.causal import dml  # noqa: E402
from mxc.structural import demand as dm  # noqa: E402

PROCESSED, INTERIM = ROOT / "data/processed", ROOT / "data/interim"
TWIN = PROCESSED / "twin"
OUT = ROOT / "assets/p2/m4"
METHODS = ["OLS-FE", "IV-FE (cost)", "DML-PLR (ML controls)", "DML-PLIV (ML + cost)"]
COL = dict(zip(METHODS, [viz.INK_2, viz.AQUA, viz.ORANGE, viz.BLUE]))
BLUES = ["#d6e6f8", "#a9cbf0", "#6fa6e6", "#2a78d6", "#1d58a3", "#123a6e"]


def partial_linear(r, controls):
    """Residual of r on a few exogenous controls (linear), for the fixed-effects-only methods."""
    b = np.linalg.lstsq(controls, r, rcond=None)[0]
    return r - controls @ b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sw = pd.read_parquet(TWIN / "store_week.parquet")
    st = pd.read_parquet(TWIN / "store.parquet", columns=["store", "canton"])
    sw = sw.merge(st, on="store")
    truth = pd.read_parquet(TWIN / "truth_store.parquet")
    truth["store"] = np.arange(len(truth))
    df = dml.add_features(sw)
    print(f"rows {len(df):,}")

    C = df[["flyer_dm", "local_promo_dm"]].to_numpy()
    y0, d0, z0 = (partial_linear(df[c].to_numpy(), C) for c in ("y_dm", "d_dm", "z_dm"))
    pred, r2 = dml.crossfit(df, ["y_dm", "d_dm", "z_dm"])
    y1, d1, z1 = (df[c].to_numpy() - pred[c] for c in ("y_dm", "d_dm", "z_dm"))
    resid = {"OLS-FE": (y0, d0, d0), "IV-FE (cost)": (y0, d0, z0),
             "DML-PLR (ML controls)": (y1, d1, d1), "DML-PLIV (ML + cost)": (y1, d1, z1)}
    nz = pd.DataFrame({"target": list(r2), "out_of_fold_r2": list(r2.values())})
    nz.loc[len(nz)] = ["first-stage corr(z̃, d̃), FE only", np.corrcoef(z0, d0)[0, 1]]
    nz.loc[len(nz)] = ["first-stage corr(z̃, d̃), DML", np.corrcoef(z1, d1)[0, 1]]
    nz.to_csv(OUT / "2_4_nuisance.csv", index=False)
    print(nz.round(3).to_string(index=False))

    store = df.store.to_numpy()
    levels = {"national": np.array(["all"] * len(df)), "banner": df.banner.to_numpy(), "canton": df.canton.to_numpy()}
    tr = {"national": pd.Series({"all": truth.own_elasticity.mean()}),
          "banner": truth.groupby("banner").own_elasticity.mean(), "canton": truth.groupby("canton").own_elasticity.mean()}
    est = []
    for m, (yr, dr, zr) in resid.items():
        for lev, cell in levels.items():
            t = dml.theta_by_cell(yr, dr, zr, cell, store).assign(method=m, level=lev)
            t["truth"] = tr[lev].reindex(t.cell).to_numpy()
            est.append(t)
    est = pd.concat(est, ignore_index=True)
    est["lo95"], est["hi95"] = est.theta - 1.96 * est.se, est.theta + 1.96 * est.se
    est.round(4).to_csv(OUT / "2_4_estimates.csv", index=False)
    print(est[est.level == "national"][["method", "theta", "se", "truth"]].round(3).to_string(index=False))

    def score(g):
        err = g.theta - g.truth
        return pd.Series({"cells": len(g), "bias": err.mean(), "rmse": np.sqrt((err ** 2).mean()),
                          "rel_bias_pct": 100 * (err / g.truth.abs()).mean(),
                          "coverage_95": ((g.lo95 <= g.truth) & (g.truth <= g.hi95)).mean(), "mean_se": g.se.mean()})
    sc = est.groupby(["level", "method"]).apply(score, include_groups=False)
    sc.round(4).to_csv(OUT / "2_4_scorecard.csv")
    print(sc.round(3).to_string())
    figures(est, sc)


def figures(est, sc):
    viz.setup()
    # 1. national
    nat = est[est.level == "national"].set_index("method").loc[METHODS]
    truth = float(nat.truth.iloc[0])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    yy = np.arange(len(METHODS))[::-1]
    ax.errorbar(nat.theta, yy, xerr=1.96 * nat.se, fmt="o", color=viz.INK, ecolor=viz.INK_2, capsize=0, markersize=0)
    ax.scatter(nat.theta, yy, color=[COL[m] for m in METHODS], s=70, zorder=3)
    for v, y_ in zip(nat.theta, yy):
        ax.text(v, y_ - 0.32, f"{v:.2f}", ha="center", fontsize=9)
    ax.axvline(truth, color=viz.AQUA, linewidth=2, label=f"truth {truth:.2f}")
    ax.set_ylim(-0.6, len(METHODS) - 0.5)
    ax.set_yticks(yy, METHODS)
    ax.set_xlabel("own-price elasticity (national)")
    ax.legend(frameon=False, loc="center")
    ax.grid(axis="y", visible=False)
    best = nat.loc[METHODS[-1], "theta"]
    ax.set_title(f"ML controls alone cannot fix endogeneity; with the cost instrument DML lands at {best:.2f} "
                 f"(truth {truth:.2f})", fontsize=10.5)
    viz.source(ax, "95% intervals clustered by store. [sim]")
    viz.save(fig, 2, "2_4", "national")

    # 2. banners
    b = est[est.level == "banner"]
    order = b[b.method == METHODS[0]].sort_values("truth").cell.tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    yy = np.arange(len(order))
    tb = b.drop_duplicates("cell").set_index("cell").truth.reindex(order)
    ax.scatter(tb, yy, marker="|", s=400, color=viz.AQUA, linewidths=3, label="truth", zorder=4)
    for off, m in zip([-0.27, -0.09, 0.09, 0.27], METHODS):
        h = b[b.method == m].set_index("cell").reindex(order)
        ax.errorbar(h.theta, yy + off, xerr=1.96 * h.se, fmt="o", color=COL[m], markersize=4, capsize=0,
                    linewidth=1, label=m)
    ax.set_yticks(yy, order)
    ax.set_xlabel("own-price elasticity")
    ax.legend(frameon=False, fontsize=8, loc="center")
    ax.grid(axis="y", visible=False)
    s = sc.loc["banner"]
    ax.set_title(f"By banner: DML-PLIV RMSE {s.loc[METHODS[3], 'rmse']:.2f} vs OLS {s.loc[METHODS[0], 'rmse']:.2f} "
                 f"and IV {s.loc[METHODS[1], 'rmse']:.2f} [sim]", fontsize=10.5)
    viz.source(ax, "Each banner's true value = mean store elasticity (answer key, 2.1).")
    viz.save(fig, 2, "2_4", "banner")

    # 3. scorecard
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    yy = np.arange(len(METHODS))
    for ax, (col, lab, fmt) in zip(axes, [("rel_bias_pct", "relative bias, %", "{:+.0f}%"),
                                          ("rmse", "RMSE (elasticity points)", "{:.2f}"),
                                          ("coverage_95", "95% coverage", "{:.0%}")]):
        for j, (lev, alpha) in enumerate([("banner", 1.0), ("canton", 0.55)]):
            v = sc.loc[lev].reindex(METHODS)[col]
            ax.barh(yy + (j - 0.5) * 0.36, v.values, height=0.36, color=[COL[m] for m in METHODS], alpha=alpha,
                    hatch=None if j == 0 else "///", edgecolor=viz.SURFACE, linewidth=0,
                    label=f"{lev} cells" if col == "rel_bias_pct" else None)
            for i, val in enumerate(v.values):
                ax.text(val, yy[i] + (j - 0.5) * 0.36, " " + fmt.format(val) + " ", va="center", fontsize=8,
                        ha="left" if val >= 0 else "right")
        ax.axvline(0, color=viz.INK_2, linewidth=0.7)
        if col == "coverage_95":
            ax.axvline(0.95, color=viz.AQUA, linewidth=1, linestyle="--")
            ax.set_xlim(0, 1.2)
        else:
            lim = np.abs(sc[col]).max() * 1.5
            ax.set_xlim(-lim if (sc[col] < 0).any() else 0, lim)
        ax.set_yticks(yy, METHODS if col == "rel_bias_pct" else [""] * len(METHODS))
        ax.invert_yaxis()
        ax.set_title(lab, fontsize=10.5)
        ax.grid(axis="y", visible=False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("T2 scorecard: 12 banner and 26 canton elasticities vs the answer key [sim]", x=0.01, ha="left",
                 fontweight="bold")
    viz.save(fig, 2, "2_4", "scorecard")

    # 4. canton map: truth vs OLS vs DML-PLIV
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cc = adm[(adm.admin_level.astype(str) == "4") & adm.ref.notna()].to_crs(2056)
    c = est[est.level == "canton"]
    show = {"Truth": c[c.method == METHODS[0]].set_index("cell").truth,
            "OLS-FE": c[c.method == METHODS[0]].set_index("cell").theta,
            "DML-PLIV": c[c.method == METHODS[3]].set_index("cell").theta}
    v = -np.concatenate([s.to_numpy() for s in show.values()])
    bins = [v.min() - 0.01, 3.0, 4.0, 4.2, 4.4, 4.6, v.max() + 0.01]
    cmap, norm = ListedColormap(BLUES), BoundaryNorm(bins, len(BLUES))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for ax, (ttl, s_) in zip(axes, show.items()):
        g = cc.merge(s_.rename("val").reset_index().rename(columns={"cell": "ref"}), on="ref", how="left")
        g.plot(column=-g.val, ax=ax, cmap=cmap, norm=norm, edgecolor=viz.SURFACE, linewidth=0.6)
        ax.set_axis_off()
        ax.set_title(ttl, fontsize=10.5)
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, orientation="horizontal",
                       fraction=0.05, pad=0.03, aspect=60, ticks=bins[1:-1])
    cb_.set_label("own-price elasticity by canton (absolute value)", fontsize=8.5)
    cb_.outline.set_visible(False)
    s = sc.loc["canton"]
    fig.suptitle(f"OLS paints Switzerland as price-insensitive; DML-PLIV recovers the map (canton RMSE "
                 f"{s.loc[METHODS[3], 'rmse']:.2f} vs {s.loc[METHODS[0], 'rmse']:.2f}) [sim]",
                 x=0.01, y=0.99, ha="left", fontweight="bold")
    viz.save(fig, 2, "2_4", "canton_map")


if __name__ == "__main__":
    main()
