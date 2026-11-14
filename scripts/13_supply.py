"""Phase 2 step 4: cooperative conduct λμ and marginal costs from 2024 annual-report margins.

Run:  conda activate keystone && python scripts/13_supply.py      (needs 10-12)
Outputs (assets/p1/m4/):
  1_4_accounts.csv          income-statement lines used, with margin arithmetic
  1_4_conduct.csv           λμ identified from Migros, over-identification check with Coop, range
  1_4_markups.csv           per banner: price, profit-max markup, markup, margin, marginal cost
  1_4_local_margin.csv      per Migros-/Coop-group store: margin it would pick if it priced alone
  1_4_supply_params.json    λμ and costs (input to GUPPI and simulation)
  1_4_margins.png, 1_4_lambda_range.png, 1_4_local_margin_map.png
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyproj  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from mxc import viz  # noqa: E402
from mxc.structural import demand as dm  # noqa: E402
from mxc.structural import substitution as sb  # noqa: E402
from mxc.structural import supply as su  # noqa: E402

PROCESSED, INTERIM = ROOT / "data/processed", ROOT / "data/interim"
OUT = ROOT / "assets/p1/m4"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# 2024 income statements, CHF m [real]. Price-cost margin = (sales − goods − share_var × personnel) / sales.
# Gross margin (share_var = 0) is the upper bound; all staff variable (share_var = 1) is the lower bound.
ACCOUNTS = {
    "Migros (Genossenschaft Migros Ostschweiz)": {"sales": 2484.600, "goods": 1642.978, "personnel": 448.470,
                                                  "source": "GMOS Geschäftsbericht 2024, Erfolgsrechnung"},
    "Coop group (consolidated)": {"sales": 33719.0, "goods": 22765.0, "personnel": 5963.0,
                                  "source": "Coop-Gruppe Geschäftsbericht 2024, Konsolidierte Erfolgsrechnung"},
}
SHARE_VAR = 0.5   # half of store staff cost varies with volume at the margin [assumed]; range 0-1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(PROCESSED / "market.npz", allow_pickle=True)
    mkt = dm.Market(**{k: z[k] for k in z.files})
    Jac = sp.load_npz(PROCESSED / "jacobian.npz").tocsr()
    ja = np.load(PROCESSED / "jacobian_abroad.npz", allow_pickle=True)
    d0, q = ja["d_abroad"], ja["q_store"]
    par = json.load(open(ROOT / "assets/p1/m2/1_2_params.json"))
    p = np.array([par["price_index"][b] for b in dm.BANNERS])
    nb = len(dm.BANNERS)

    # --- 1. accounting margins
    acc = pd.DataFrame(ACCOUNTS).T
    for c in ["sales", "goods", "personnel"]:
        acc[c] = acc[c].astype(float)
    acc["gross_margin"] = (acc.sales - acc.goods) / acc.sales
    acc["margin_all_staff_variable"] = (acc.sales - acc.goods - acc.personnel) / acc.sales
    acc["margin_central"] = (acc.sales - acc.goods - SHARE_VAR * acc.personnel) / acc.sales
    acc.round(4).to_csv(OUT / "1_4_accounts.csv")
    print(acc[["gross_margin", "margin_all_staff_variable", "margin_central"]].round(4).to_string())
    mM, mC = acc.margin_central.iloc[0], acc.margin_central.iloc[1]

    # --- 2. banner Jacobian and profit-max markups (λμ = 0)
    _, A, _ = sb.group_diversion(Jac, d0, mkt.store_banner, nb)
    Q = np.bincount(mkt.store_banner, weights=q, minlength=nb)
    m0 = su.markups(A, Q, su.OWNER_PRE, np.zeros(nb))
    iM, iC = dm.BANNERS.index("Migros"), dm.BANNERS.index("Coop")
    lmM = su.lambda_from_margin(m0[iM], p[iM], mM)
    lmC = su.lambda_from_margin(m0[iC], p[iC], mC)
    lm = np.array([lmM if su.OWNER_PRE[b] in ("Migros group", "Coop group") else 0.0 for b in dm.BANNERS])
    m = su.markups(A, Q, su.OWNER_PRE, lm)
    c = p - m
    resid = np.abs(su.foc_residual(A, Q, su.OWNER_PRE, lm, m)).max()
    print(f"profit-max markups m0: Migros {m0[iM]:.2f}, Coop {m0[iC]:.2f}; λμ from Migros {lmM:.4f}, from Coop {lmC:.4f}; "
          f"FOC residual {resid:.1e}")

    # λμ over the plausible margin range (share_var 0..1)
    rng = []
    for sv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mm = (acc.sales - acc.goods - sv * acc.personnel) / acc.sales
        rng.append({"share_staff_variable": sv, "migros_margin": mm.iloc[0], "coop_margin": mm.iloc[1],
                    "lambda_mu_from_migros": su.lambda_from_margin(m0[iM], p[iM], mm.iloc[0]),
                    "lambda_mu_from_coop": su.lambda_from_margin(m0[iC], p[iC], mm.iloc[1])})
    conduct = pd.DataFrame(rng)
    conduct.round(4).to_csv(OUT / "1_4_conduct.csv", index=False)
    print(conduct.round(3).to_string(index=False))

    mk = pd.DataFrame({"owner": [su.OWNER_PRE[b] for b in dm.BANNERS], "price": p, "baskets_per_week": Q,
                       "profit_max_markup": m0, "profit_max_margin": m0 / p, "lambda_mu": lm,
                       "markup": m, "margin": m / p, "marginal_cost": c}, index=dm.BANNERS)
    mk.round(4).to_csv(OUT / "1_4_markups.csv")
    print(mk.round(3).to_string())

    # --- 3. local margins: what each merging-party store would charge if it could price on its own
    owner_store = np.array([su.OWNER_PRE[dm.BANNERS[b]] for b in mkt.store_banner])
    lm_store = lm[mkt.store_banner]
    loc_m = su.local_margin(Jac, q, mkt.store_banner, owner_store, m, lm_store)
    st = pd.read_parquet(PROCESSED / "stores.parquet", columns=["store_id", "lon", "lat", "canton", "municipality"])
    lmdf = pd.DataFrame({"store_id": mkt.store_id, "banner": [dm.BANNERS[b] for b in mkt.store_banner],
                         "owner": owner_store, "national_margin": (m / p)[mkt.store_banner],
                         "local_margin": loc_m / p[mkt.store_banner], "baskets_per_week": q}).merge(st, on="store_id")
    lmdf["gap_pp"] = (lmdf.local_margin - lmdf.national_margin) * 100
    sel = lmdf[lmdf.owner.isin(["Migros group", "Coop group"])]
    sel.round(4).to_csv(OUT / "1_4_local_margin.csv", index=False)
    print(sel.groupby("banner")[["national_margin", "local_margin", "gap_pp"]].median().round(3).to_string())

    json.dump({"lambda_mu": dict(zip(dm.BANNERS, map(float, lm))), "marginal_cost": dict(zip(dm.BANNERS, map(float, c))),
               "markup": dict(zip(dm.BANNERS, map(float, m))), "target_margin_migros": float(mM),
               "target_margin_coop": float(mC), "share_staff_variable": SHARE_VAR},
              open(OUT / "1_4_supply_params.json", "w"), indent=2)
    figures(mk, conduct, sel, lmM, lmC)


def figures(mk, conduct, sel, lmM, lmC):
    viz.setup()
    # 1. margins: profit-max vs actual by banner
    t = mk.sort_values("margin")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.hlines(y, t.margin, t.profit_max_margin, color=viz.GRID, linewidth=2, zorder=1)
    ax.scatter(t.profit_max_margin, y, s=34, facecolors=viz.SURFACE, edgecolors=viz.INK_2, zorder=3,
               label="if it maximised profit only (λμ = 0)")
    col = [viz.BLUE if o == "Migros group" else viz.ORANGE if o == "Coop group" else viz.MUTED for o in t.owner]
    ax.scatter(t.margin, y, s=38, c=col, zorder=4, edgecolors=viz.SURFACE)
    ax.scatter([], [], s=38, c=viz.BLUE, label="model margin, Migros group")
    ax.scatter([], [], s=38, c=viz.ORANGE, label="model margin, Coop group")
    ax.scatter([], [], s=38, c=viz.MUTED, label="model margin, rivals (λμ = 0)")
    ax.set_yticks(y, t.index)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("price-cost margin (p − c) / p")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Cooperative conduct: Migros and Coop price {lmM:.0%} below the profit-maximising markup")
    viz.source(ax, "Uniform national banner prices; FOCs at banner level. λμ fitted to Migros Ostschweiz 2024 accounts.")
    viz.save(fig, 1, "1_4", "margins")

    # 2. λμ over the margin range
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = conduct.share_staff_variable
    ax.plot(x, conduct.lambda_mu_from_migros, color=viz.BLUE, marker="o", label="from Migros Ostschweiz margin")
    ax.plot(x, conduct.lambda_mu_from_coop, color=viz.ORANGE, marker="o", label="from Coop group margin")
    ax.axvline(SHARE_VAR, color=viz.INK_2, linestyle=":", linewidth=0.8)
    ax.text(SHARE_VAR + 0.02, ax.get_ylim()[1] * 0.95, "central case", fontsize=8.5, color=viz.INK_2, va="top")
    ax.set_xlabel("share of staff cost that varies with volume")
    ax.set_ylabel("implied λμ (weight on shoppers)")
    ax.legend(loc="upper left")
    ax.set_title("Migros and Coop accounts imply similar conduct (within 0.07) at every margin")
    viz.source(ax, "λμ = 1 − margin × p / profit-max markup. Margin = (sales − goods − share × staff) / sales, 2024 accounts.")
    viz.save(fig, 1, "1_4", "lambda_range")

    # 3. map: local margin gap (store would charge more if it priced alone)
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cantons = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)
    x, y2 = pyproj.Transformer.from_crs(4326, 2056, always_xy=True).transform(sel.lon.to_numpy(), sel.lat.to_numpy())
    v = sel.gap_pp.to_numpy()
    bins = [-100, -3, -1.5, -0.5, 0.5, 1.5, 3, 8, 100]          # diverging: blue = would charge less, red = more
    div = ["#2a78d6", "#6da7ec", "#b7d3f6", "#d9d8d3", "#f6c7c6", "#ee9190", "#e34948", "#a32f2e"]
    cmap, norm = ListedColormap(div), BoundaryNorm(bins, len(div))
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    viz.map_axes(ax, cantons)
    o = np.argsort(np.abs(v))
    ax.scatter(x[o], y2[o], c=v[o], cmap=cmap, norm=norm, s=np.where(np.abs(v[o]) > 3, 14, 5), linewidths=0, zorder=2)
    top = sel.iloc[np.argmax(v)]
    ax.set_title(f"National prices bind in resorts: alone, {top.banner} {top.municipality} would add {v.max():+.0f} pp")
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal",
                       fraction=0.04, pad=0.02, aspect=40, ticks=bins[1:-1])
    cb_.ax.set_xticklabels([f"{b:+g}" for b in bins[1:-1]], fontsize=8)
    cb_.set_label("store's own optimal margin minus its national banner margin, percentage points "
                  f"(median {np.median(v):+.1f})", fontsize=8.5, color=viz.INK_2)
    cb_.outline.set_visible(False)
    viz.source(ax, "Each dot is a Migros-group or Coop-group store; it alone re-prices, all other prices fixed. Model [calibrated].")
    viz.save(fig, 1, "1_4", "local_margin_map")


if __name__ == "__main__":
    main()
