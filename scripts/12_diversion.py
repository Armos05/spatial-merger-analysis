"""Phase 2 step 3: store Jacobian, diversion ratios (store, player, canton) and elasticities.

Run:  conda activate keystone && python scripts/12_diversion.py      (needs 10_demand.py, 11_calibrate.py)
Outputs:
  data/processed/jacobian.npz          store × store dq_k/dp_j (sparse) + abroad column (input to GUPPI, simulation)
  assets/p1/m3/1_3_player_diversion.csv   D(A→B) when every store of player A raises price by CHF 1
  assets/p1/m3/1_3_elasticities.csv       own-price elasticity per banner (uniform banner price rise)
  assets/p1/m3/1_3_canton_diversion.csv   diversion from each merging group's stores in a canton
  assets/p1/m3/1_3_store_diversion.csv    per Migros-/Coop-group store: diversion to each owner class
  assets/p1/m3/1_3_player_diversion.png, 1_3_canton_diversion.png, 1_3_store_diversion_map.png
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyproj  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from mxc import viz  # noqa: E402
from mxc.structural import calibrate as cb  # noqa: E402
from mxc.structural import demand as dm  # noqa: E402
from mxc.structural import substitution as sb  # noqa: E402

PROCESSED, INTERIM = ROOT / "data/processed", ROOT / "data/interim"
OUT = ROOT / "assets/p1/m3"
OWNER = {"Migros": "Migros group", "Coop": "Coop group", "Aldi": "Aldi + Lidl", "Lidl": "Aldi + Lidl"}
CLASSES = ["Migros group", "Coop group", "Aldi + Lidl", "Others"]
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load():
    z = np.load(PROCESSED / "market.npz", allow_pickle=True)
    mkt = dm.Market(**{k: z[k] for k in z.files})
    p = json.load(open(ROOT / "assets/p1/m2/1_2_params.json"))
    theta = dm.prior_params()._replace(
        delta=jnp.array([p["delta"][b] for b in dm.BANNERS]), alpha_ref=p["alpha_ref"], eta=p["eta"],
        gamma=p["gamma"], kappa=p["kappa"], sigma=p["sigma"],
        p=jnp.array([p["price_index"][b] for b in dm.BANNERS]), p_abroad=p["p_abroad"])
    return mkt, theta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mkt, theta = load()
    a = dm.to_arrays(mkt)
    t0 = time.time()
    Jac, d0, q = sb.jacobian(theta, a)
    print(f"Jacobian {Jac.shape}, {Jac.nnz:,} non-zeros, built in {time.time() - t0:.0f}s")
    sp.save_npz(PROCESSED / "jacobian.npz", Jac)
    np.savez(PROCESSED / "jacobian_abroad.npz", d_abroad=d0, q_store=q, store_id=mkt.store_id)
    resid = np.abs(np.asarray(Jac.sum(1)).ravel() + d0).max()
    print(f"check: max |row sum + abroad| = {resid:.2e} (should be ~0)")

    price = np.asarray(theta.p)[mkt.store_banner]

    # --- 1. player-level diversion
    pidx = cb.player_index()[mkt.store_banner]
    players = list(cb.PLAYERS)
    D, agg, lost = sb.group_diversion(Jac, d0, pidx, len(players))
    pdiv = pd.DataFrame(D, index=players, columns=players + ["abroad"])
    # merging-party level (Migros group = Migros + Denner)
    oc = np.array([CLASSES.index(OWNER.get(g, "Others")) for g in mkt.store_group])
    D2, _, _ = sb.group_diversion(Jac, d0, oc, len(CLASSES))
    odiv = pd.DataFrame(D2, index=CLASSES, columns=CLASSES + ["abroad"])
    pd.concat({"player": pdiv, "owner": odiv}).round(4).to_csv(OUT / "1_3_player_diversion.csv")
    print(pdiv.round(3).to_string(), "\n", odiv.round(3).to_string())

    # --- 2. elasticities, banner-wide price rise: e_B = (Σ_{j,k∈B} dq_k/dp_j) × p_B / q_B
    bidx = mkt.store_banner
    _, aggb, lostb = sb.group_diversion(Jac, d0, bidx, len(dm.BANNERS))
    qb = np.bincount(bidx, weights=q, minlength=len(dm.BANNERS))
    store_e = Jac.diagonal() * price / np.where(q > 0, q, np.nan)
    el = pd.DataFrame({"banner_elasticity": -lostb * np.asarray(theta.p) / qb,
                       "median_store_elasticity": [np.nanmedian(store_e[bidx == b]) for b in range(len(dm.BANNERS))],
                       "baskets_per_week": qb}, index=dm.BANNERS)
    el.round(3).to_csv(OUT / "1_3_elasticities.csv")
    print(el.round(2).to_string())

    # --- 3. store-level diversion to each owner class
    M = sp.csr_matrix((np.ones(len(oc)), (np.arange(len(oc)), oc)), shape=(len(oc), len(CLASSES)))
    to_class = (Jac @ M).toarray()                                     # Σ_{k∈class} dq_k/dp_j, incl. own store
    own = -Jac.diagonal()
    to_class[np.arange(len(oc)), oc] -= Jac.diagonal()                 # exclude the store itself
    sd = pd.DataFrame(to_class / own[:, None], columns=[f"to_{c}" for c in CLASSES])
    sd["to_abroad"] = d0 / own
    sd.insert(0, "store_id", mkt.store_id)
    sd.insert(1, "banner", [dm.BANNERS[b] for b in bidx])
    sd.insert(2, "owner", [CLASSES[c] for c in oc])
    sd["own_elasticity"] = store_e
    sd["baskets_per_week"] = q
    st = pd.read_parquet(PROCESSED / "stores.parquet", columns=["store_id", "canton", "municipality", "lon", "lat"])
    sd = sd.merge(st, on="store_id", how="left")
    sd[sd.owner.isin(CLASSES[:2])].round(4).to_csv(OUT / "1_3_store_diversion.csv", index=False)

    # --- 4. canton: all stores of one merging group in a canton raise price by CHF 1
    rows = []
    for c in sorted(sd.canton.dropna().unique()):
        for g in CLASSES[:2]:
            sel = ((sd.canton == c) & (sd.owner == g)).to_numpy()
            if sel.sum() == 0:
                continue
            v = np.asarray(Jac[sel].sum(0)).ravel()                    # Σ_{j∈A} dq_k/dp_j for every k
            lost_c = -v[sel].sum()
            byclass = np.bincount(oc, weights=v, minlength=len(CLASSES))
            byclass[CLASSES.index(g)] -= v[sel].sum()                  # own group outside this canton stays in
            rows.append({"canton": c, "from": g, "stores": int(sel.sum()),
                         **{f"to_{k}": byclass[i] / lost_c for i, k in enumerate(CLASSES)},
                         "to_abroad": d0[sel].sum() / lost_c})
    cd = pd.DataFrame(rows)
    cd.round(4).to_csv(OUT / "1_3_canton_diversion.csv", index=False)
    print(cd[cd["from"] == "Migros group"].round(3).to_string(index=False))

    figures(pdiv, odiv, cd, sd)


def figures(pdiv, odiv, cd, sd):
    viz.setup()
    # 1. player diversion heatmap
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    cmap, norm = ListedColormap(RAMP), BoundaryNorm(bins, len(RAMP))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.imshow(pdiv.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto")
    for i in range(pdiv.shape[0]):
        for j in range(pdiv.shape[1]):
            v = pdiv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color=viz.SURFACE if v >= 0.3 else viz.INK)
    ax.set_xticks(range(pdiv.shape[1]), pdiv.columns)
    ax.set_yticks(range(pdiv.shape[0]), [f"{r} raises price" for r in pdiv.index])
    ax.xaxis.tick_top()
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    mc, cm = odiv.loc["Migros group", "Coop group"], odiv.loc["Coop group", "Migros group"]
    ax.set_title(f"Where lost sales go. Migros group → Coop {mc:.0%}; Coop → Migros group {cm:.0%}", pad=28)
    viz.source(ax, "Diversion D(A→B) = share of the sales A loses, after a CHF 1 rise at all its stores, that go to B. "
                   "Model [calibrated]; σ = 0.25 [assumed].")
    viz.save(fig, 1, "1_3", "player_diversion")

    # 2. canton dumbbell: Migros group's stores in canton → Coop group vs Aldi + Lidl vs abroad
    t = cd[cd["from"] == "Migros group"].sort_values("to_Coop group")
    fig, ax = plt.subplots(figsize=(7.5, 7.2))
    y = np.arange(len(t))
    ax.hlines(y, t["to_Aldi + Lidl"], t["to_Coop group"], color=viz.GRID, linewidth=2, zorder=1)
    ax.scatter(t["to_Coop group"], y, color=viz.ORANGE, s=34, zorder=3, label="to Coop group", edgecolors=viz.SURFACE)
    ax.scatter(t["to_Aldi + Lidl"], y, color=viz.AQUA, s=34, zorder=3, label="to Aldi + Lidl", edgecolors=viz.SURFACE)
    ax.scatter(t["to_abroad"], y, color=viz.INK_2, s=16, marker="|", zorder=3, label="abroad")
    ax.set_yticks(y, t.canton)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("share of the Migros group's lost sales")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("In every canton Coop catches far more of Migros's lost sales than Aldi + Lidl")
    viz.source(ax, "All Migros-group stores (Migros, Partner, Migrolino, Denner) in the canton raise prices by CHF 1. Model [calibrated].")
    viz.save(fig, 1, "1_3", "canton_diversion")

    # 3. map: Migros-group stores by diversion to Coop group
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cantons = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)
    m = sd[sd.owner == "Migros group"].copy()
    x, y2 = pyproj.Transformer.from_crs(4326, 2056, always_xy=True).transform(m.lon.to_numpy(), m.lat.to_numpy())
    v = m["to_Coop group"].to_numpy()
    bins = [0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 1.0]
    cmap, norm = ListedColormap(RAMP), BoundaryNorm(bins, len(RAMP))
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    viz.map_axes(ax, cantons)
    o = np.argsort(v)
    ax.scatter(x[o], y2[o], c=v[o], cmap=cmap, norm=norm, s=6, linewidths=0, zorder=2)
    top = m.iloc[np.argmax(v)]
    ax.set_title(f"The median Migros-group store sends {np.median(v):.0%} of lost sales to Coop; "
                 f"up to {v.max():.0%} ({top.municipality})")
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal",
                       fraction=0.04, pad=0.02, aspect=40, ticks=bins[1:-1])
    cb_.ax.set_xticklabels([f"{b:.0%}" for b in bins[1:-1]], fontsize=8)
    cb_.set_label("diversion from the store to Coop-group stores, D(j → Coop group)", fontsize=8.5, color=viz.INK_2)
    cb_.outline.set_visible(False)
    viz.source(ax, "Each dot is one Migros, Partner, Migrolino or Denner store; CHF 1 rise at that store only. Model [calibrated].")
    viz.save(fig, 1, "1_3", "store_diversion_map")


if __name__ == "__main__":
    main()
