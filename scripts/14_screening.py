"""Phase 2 step 5: GUPPI and CMCR screens, national and store by store, plus a Koh-style no-price check.

Run:  conda activate keystone && python scripts/14_screening.py      (needs 10-13)
Outputs (assets/p1/m5/):
  1_5_national.csv          per banner: national GUPPI and CMCR (joint re-optimisation)
  1_5_store_screen.csv      per Migros-/Coop-group store: diversion to the other party, GUPPI, CMCR, flag
  1_5_canton_screen.csv     flagged stores and their sales by canton
  1_5_koh_check.csv         GUPPIs when every domestic banner price is set to 100 and the model is re-fitted
  1_5_guppi_map.png, 1_5_guppi_distribution.png, 1_5_canton_flags.png
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
import pyproj  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from mxc import viz  # noqa: E402
from mxc.structural import calibrate as cb  # noqa: E402
from mxc.structural import demand as dm  # noqa: E402
from mxc.structural import screening as sc  # noqa: E402
from mxc.structural import substitution as sb  # noqa: E402
from mxc.structural import supply as su  # noqa: E402

PROCESSED, INTERIM = ROOT / "data/processed", ROOT / "data/interim"
OUT = ROOT / "assets/p1/m5"
P1 = ROOT / "assets/p1"


def load():
    z = np.load(PROCESSED / "market.npz", allow_pickle=True)
    mkt = dm.Market(**{k: z[k] for k in z.files})
    par = json.load(open(P1 / "m2/1_2_params.json"))
    sup = json.load(open(P1 / "m4/1_4_supply_params.json"))
    theta = dm.prior_params()._replace(
        delta=jnp.array([par["delta"][b] for b in dm.BANNERS]), alpha_ref=par["alpha_ref"], eta=par["eta"],
        gamma=par["gamma"], kappa=par["kappa"], sigma=par["sigma"],
        p=jnp.array([par["price_index"][b] for b in dm.BANNERS]), p_abroad=par["p_abroad"])
    return mkt, theta, par, sup


def screen(mkt, Jac, d0, q, price, lm, margin_targets=None):
    """Banner markups (refit λμ if margin_targets given), national GUPPI/CMCR, store GUPPIs."""
    nb = len(dm.BANNERS)
    _, A, _ = sb.group_diversion(Jac, d0, mkt.store_banner, nb)
    Q = np.bincount(mkt.store_banner, weights=q, minlength=nb)
    if margin_targets is not None:
        m0 = su.markups(A, Q, su.OWNER_PRE, np.zeros(nb))
        i = dm.BANNERS.index("Migros")
        lam = su.lambda_from_margin(m0[i], price[i], margin_targets)
        lm = np.array([lam if su.OWNER_PRE[b] in ("Migros group", "Coop group") else 0.0 for b in dm.BANNERS])
    m = su.markups(A, Q, su.OWNER_PRE, lm)
    cost = price - m
    g_nat = sc.banner_guppi(A, price, m)
    cmcr_nat, m_post = sc.national_cmcr(A, Q, lm, price, cost)
    party = np.array([su.OWNER_PRE[dm.BANNERS[b]] for b in mkt.store_banner])
    g, d = sc.store_guppi(Jac, party, m[mkt.store_banner], price[mkt.store_banner])
    return dict(A=A, Q=Q, lm=lm, m=m, cost=cost, g_nat=g_nat, cmcr_nat=cmcr_nat, m_post=m_post, g=g, d=d, party=party)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mkt, theta, par, sup = load()
    Jac = sp.load_npz(PROCESSED / "jacobian.npz").tocsr()
    ja = np.load(PROCESSED / "jacobian_abroad.npz", allow_pickle=True)
    price = np.asarray(theta.p)
    lm = np.array([sup["lambda_mu"][b] for b in dm.BANNERS])
    r = screen(mkt, Jac, ja["d_abroad"], ja["q_store"], price, lm)

    # --- national
    nat = pd.DataFrame({"owner": [su.OWNER_PRE[b] for b in dm.BANNERS], "price": price, "margin": r["m"] / price,
                        "national_guppi": r["g_nat"], "markup_today": r["m"], "markup_post_at_today_prices": r["m_post"],
                        "marginal_cost": r["cost"], "national_cmcr": r["cmcr_nat"]}, index=dm.BANNERS)
    nat = nat[nat.owner.isin(["Migros group", "Coop group"])]
    nat.round(4).to_csv(OUT / "1_5_national.csv")
    print(nat.round(3).to_string())

    # --- store level
    st = pd.read_parquet(PROCESSED / "stores.parquet", columns=["store_id", "lon", "lat", "canton", "municipality", "name"])
    sel = np.isfinite(r["g"])
    b = mkt.store_banner
    fmt = np.where(np.array([dm.BANNER_NEST[dm.BANNERS[x]] for x in b]) == "convenience", "convenience", "supermarket")
    thr = np.array([sc.THRESHOLD[f] for f in fmt])
    cost_store = r["cost"][b]
    sdf = pd.DataFrame({"store_id": mkt.store_id, "banner": [dm.BANNERS[x] for x in b], "party": r["party"],
                        "format": fmt, "diversion_to_other_party": r["d"], "guppi": r["g"],
                        "cmcr_one_sided": r["g"] * price[b] / cost_store, "threshold": thr,
                        "sales_chf_week": ja["q_store"] * price[b]})[sel].merge(st, on="store_id")
    sdf["flag"] = sdf.guppi > sdf.threshold
    sdf["tier"] = pd.cut(sdf.guppi, [0, 0.0275, 0.05, 0.075, 0.10, 1], labels=["< 2.75%", "2.75–5%", "5–7.5%", "7.5–10%", "≥ 10%"])
    sdf.round(5).to_csv(OUT / "1_5_store_screen.csv", index=False)
    print(f"stores screened {len(sdf)}, flagged {sdf.flag.sum()} ({sdf.flag.mean():.1%}); "
          f"sales flagged {sdf.sales_chf_week[sdf.flag].sum() / sdf.sales_chf_week.sum():.1%}")
    print(sdf.groupby("banner")[["diversion_to_other_party", "guppi", "cmcr_one_sided"]].median().round(4).to_string())
    print(sdf.tier.value_counts().sort_index().to_string())

    can = sdf.groupby("canton").apply(lambda g: pd.Series({
        "stores": len(g), "flagged": int(g.flag.sum()), "flagged_share": g.flag.mean(),
        "median_guppi": g.guppi.median(), "stores_guppi_ge_10pct": int((g.guppi >= 0.10).sum()),
        "sales_flagged_share": g.sales_chf_week[g.flag].sum() / g.sales_chf_week.sum()}), include_groups=False)
    can = can.sort_values("median_guppi", ascending=False)
    can.round(4).to_csv(OUT / "1_5_canton_screen.csv")

    # --- Koh-style check: no price levels. All domestic prices = 100; re-fit delta, Jacobian, λμ; same margins.
    theta_np = theta._replace(p=jnp.full(len(dm.BANNERS), 100.0), delta=jnp.zeros(len(dm.BANNERS)))
    a = dm.to_arrays(mkt)
    tgt = cb.targets(par["abroad_share_target"])
    theta_np, hist = cb.calibrate(theta_np, a, tgt)
    Jn, d0n, qn = sb.jacobian(theta_np, a)
    rn = screen(mkt, Jn, d0n, qn, np.full(len(dm.BANNERS), 100.0), None, margin_targets=sup["target_margin_migros"])
    koh = pd.DataFrame({"banner": sdf.banner, "guppi_price_index": sdf.guppi.to_numpy(),
                        "guppi_no_prices": rn["g"][sel]})
    kb = koh.groupby("banner").median()
    kb.loc["all stores"] = koh[["guppi_price_index", "guppi_no_prices"]].median()
    kb["lambda_mu_no_prices"] = rn["lm"][dm.BANNERS.index("Migros")]
    kb["corr_store_level"] = np.corrcoef(koh.guppi_price_index, koh.guppi_no_prices)[0, 1]
    kb.round(4).to_csv(OUT / "1_5_koh_check.csv")
    print("Koh check (medians):\n", kb.round(4).to_string())

    figures(sdf, can, nat)


def figures(sdf, can, nat):
    viz.setup()
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cantons = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)
    x, y = pyproj.Transformer.from_crs(4326, 2056, always_xy=True).transform(sdf.lon.to_numpy(), sdf.lat.to_numpy())
    v = sdf.guppi.to_numpy()

    # 1. map
    bins = [0, 0.0275, 0.05, 0.075, 0.10, 0.15, 1.0]
    cols = ["#d9d8d3", "#f6c7c6", "#ee9190", "#e34948", "#b52f2e", "#7a1f1e"]
    cmap, norm = ListedColormap(cols), BoundaryNorm(bins, len(cols))
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    viz.map_axes(ax, cantons)
    o = np.argsort(v)
    ax.scatter(x[o], y[o], c=v[o], cmap=cmap, norm=norm, s=np.where(v[o] >= 0.10, 12, 5), linewidths=0, zorder=2)
    ax.set_title(f"{sdf.flag.sum():,} of {len(sdf):,} Migros and Coop stores fail the CMA screen; median GUPPI {np.median(v):.1%}")
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal",
                       fraction=0.04, pad=0.02, aspect=40, ticks=bins[1:-1])
    cb_.ax.set_xticklabels(["2.75%", "5%", "7.5%", "10%", "15%"], fontsize=8)
    cb_.set_label("store GUPPI = Σ diversion to the other party × its margin / own price  (grey = passes 2.75%)",
                  fontsize=8.5, color=viz.INK_2)
    cb_.outline.set_visible(False)
    viz.source(ax, "Each dot is a Migros-group or Coop-group store. Threshold 2.75% (3.25% for Migrolino, Coop Pronto), "
                   "CMA Sainsbury's/Asda. Model [calibrated].")
    viz.save(fig, 1, "1_5", "guppi_map")

    # 2. distribution by party
    fig, ax = plt.subplots(figsize=(7.5, 4))
    edges = np.arange(0, 0.205, 0.005)
    for party, colr in [("Migros group", viz.BLUE), ("Coop group", viz.ORANGE)]:
        ax.hist(sdf.guppi[sdf.party == party].clip(upper=0.2), bins=edges, histtype="step", linewidth=2, color=colr,
                label=f"{party} stores (median {sdf.guppi[sdf.party == party].median():.1%})")
    ax.axvline(0.0275, color=viz.INK_2, linestyle=":", linewidth=1)
    ax.text(0.029, ax.get_ylim()[1] * 0.95, "CMA 2.75%", fontsize=8.5, color=viz.INK_2, va="top")
    ax.set_xticks(np.arange(0, 0.201, 0.05))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("store GUPPI (values above 20% shown at 20%)")
    ax.set_ylabel("stores")
    ax.legend(loc="upper right")
    gM, gC = nat.loc["Migros", "national_guppi"], nat.loc["Coop", "national_guppi"]
    ax.set_title(f"Almost no store is below the threshold; national GUPPI Migros {gM:.1%}, Coop {gC:.1%}")
    viz.source(ax, "GUPPI_j = Σ_k D(j→k) × (p_k − c_k) / p_j over the other party's stores. Model [calibrated].")
    viz.save(fig, 1, "1_5", "guppi_distribution")

    # 3. canton: stores with GUPPI ≥ 10%
    t = can.sort_values("median_guppi")
    fig, ax = plt.subplots(figsize=(7.5, 7))
    yy = np.arange(len(t))
    ax.barh(yy, t.median_guppi, color=viz.MUTED, height=0.6, label="median store GUPPI")
    ax.axvline(0.0275, color=viz.INK_2, linestyle=":", linewidth=1)
    for i, (mg, n10, n) in enumerate(zip(t.median_guppi, t.stores_guppi_ge_10pct, t.stores)):
        ax.text(mg + 0.001, i, f"{int(n10)}/{int(n)} stores ≥ 10%" if n10 else "", va="center", fontsize=7.5, color=viz.INK)
    ax.set_yticks(yy, t.index)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("median store GUPPI in the canton (dotted line: 2.75%)")
    ax.grid(axis="y", visible=False)
    top3 = ", ".join(t.index[::-1][:3])
    ax.set_title(f"Every canton's median store is above 2.75%; highest in {top3}")
    viz.source(ax, "Labels: stores with GUPPI ≥ 10% / all Migros- and Coop-group stores in the canton.")
    viz.save(fig, 1, "1_5", "canton_flags")


if __name__ == "__main__":
    main()
