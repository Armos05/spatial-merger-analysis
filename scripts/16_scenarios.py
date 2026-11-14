"""Phase 2 step 7: scenarios. Efficiencies 0-10% on the merged banners' marginal cost × cooperative weight
λμ after the merger: survives (0.357), dilutes (half) or disappears (0).

Run:  conda activate keystone && python scripts/16_scenarios.py      (needs 10-13)
Outputs (assets/p1/m7/):
  1_7_scenarios.csv          one row per scenario: price changes, ΔCS, profits, net welfare
  1_7_breakeven.csv          efficiency needed for zero consumer harm, per λμ case (extrapolated if > 10%)
  1_7_cs_canton.csv          ΔCS per household by canton, three λμ cases at 0% efficiency
  1_7_scenario_grid.png, 1_7_efficiency_lines.png, 1_7_cs_maps.png
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
from mxc.structural import demand as dm  # noqa: E402
from mxc.structural import simulate as sim  # noqa: E402
from mxc.structural import supply as su  # noqa: E402

PROCESSED, INTERIM = ROOT / "data/processed", ROOT / "data/interim"
OUT = ROOT / "assets/p1/m7"
P1 = ROOT / "assets/p1"
EFFICIENCIES = [0.0, 0.025, 0.05, 0.075, 0.10]
RED = ["#f6c7c6", "#ee9190", "#e34948", "#b52f2e", "#7a1f1e"]
SHOW = ["Migros", "Coop", "Denner", "Aldi", "Lidl"]


def load():
    z = np.load(PROCESSED / "market.npz", allow_pickle=True)
    mkt = dm.Market(**{k: z[k] for k in z.files})
    par = json.load(open(P1 / "m2/1_2_params.json"))
    sup = json.load(open(P1 / "m4/1_4_supply_params.json"))
    theta = dm.prior_params()._replace(
        delta=jnp.array([par["delta"][b] for b in dm.BANNERS]), alpha_ref=par["alpha_ref"], eta=par["eta"],
        gamma=par["gamma"], kappa=par["kappa"], sigma=par["sigma"],
        p=jnp.array([par["price_index"][b] for b in dm.BANNERS]), p_abroad=par["p_abroad"])
    lm = np.array([sup["lambda_mu"][b] for b in dm.BANNERS])
    cost = np.array([sup["marginal_cost"][b] for b in dm.BANNERS])
    return mkt, theta, lm, cost


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mkt, theta, lm_pre, cost = load()
    a = dm.to_arrays(mkt)
    bd = sim.make_banner_demand(a)
    p0 = np.asarray(theta.p)
    n_lr = np.asarray(a["hh"])[:, None] * np.asarray(a["type_w"])
    hh = n_lr.sum()
    merged = np.array([su.OWNER_POST[b] == "Migros-Coop" for b in dm.BANNERS])
    lam0 = float(lm_pre[merged][0])
    cases = {"survives": lam0, "dilutes": lam0 / 2, "disappears": 0.0}
    owner = np.array([su.OWNER_POST[b] for b in dm.BANNERS])
    Q0 = np.asarray(bd(theta)[0])

    rows, loc_cs, p_prev = [], {}, None
    for case, lam in cases.items():
        lm = np.where(merged, lam, 0.0)
        for e in EFFICIENCIES:
            c = np.where(merged, cost * (1 - e), cost)
            p1, hist = sim.solve_equilibrium(theta, bd, c, lm, su.OWNER_POST, p_start=p_prev)
            p_prev = p1
            th1 = theta._replace(p=jnp.asarray(p1))
            Q1 = np.asarray(bd(th1)[0])
            dcs = sim.delta_cs(theta, th1, a) * sim.WEEKS                           # (L,R) CHF / hh / yr
            tot = (n_lr * dcs).sum()
            d_prof_merged = (((p1 - c) * Q1)[merged].sum() - ((p0 - cost) * Q0)[merged].sum()) * sim.WEEKS
            d_prof_all = (((p1 - c) * Q1).sum() - ((p0 - cost) * Q0).sum()) * sim.WEEKS
            rows.append({"lambda_mu_case": case, "lambda_mu": lam, "efficiency": e, "iterations": len(hist) - 1,
                         **{f"dp_{b}_pct": (p1[dm.BANNERS.index(b)] / p0[dm.BANNERS.index(b)] - 1) * 100 for b in SHOW},
                         "dcs_chf_per_household_year": tot / hh, "dcs_chf_bn_year": tot / 1e9,
                         "dprofit_merged_chf_bn_year": d_prof_merged / 1e9,
                         "net_welfare_chf_bn_year": (tot + d_prof_all) / 1e9})
            print(f"{case:10s} λμ={lam:.3f} e={e:.3f}: Migros {rows[-1]['dp_Migros_pct']:+.1f}%, "
                  f"Coop {rows[-1]['dp_Coop_pct']:+.1f}%, ΔCS {tot / hh:,.0f} CHF/hh/yr ({len(hist) - 1} it)")
            if e == 0.0:
                loc_cs[case] = (n_lr * dcs).sum(1) / n_lr.sum(1)
        p_prev = None
    sc = pd.DataFrame(rows)
    sc.round(4).to_csv(OUT / "1_7_scenarios.csv", index=False)

    # break-even efficiency: linear fit of ΔCS on e within each case, solve for ΔCS = 0
    be = []
    for case, g in sc.groupby("lambda_mu_case", sort=False):
        b1, b0 = np.polyfit(g.efficiency, g.dcs_chf_per_household_year, 1)
        be.append({"lambda_mu_case": case, "chf_per_household_per_pp_efficiency": b1 / 100,
                   "breakeven_efficiency_linear": -b0 / b1})
    pd.DataFrame(be).round(4).to_csv(OUT / "1_7_breakeven.csv", index=False)
    print(pd.DataFrame(be).round(3).to_string(index=False))

    can = pd.DataFrame({"canton": mkt.canton, "hh": mkt.hh, **loc_cs}).groupby("canton").apply(
        lambda g: pd.Series({k: np.average(g[k], weights=g.hh) for k in cases}), include_groups=False)
    can.round(1).sort_values("disappears").to_csv(OUT / "1_7_cs_canton.csv")

    figures(sc, loc_cs, mkt, cases)


def figures(sc, loc_cs, mkt, cases):
    viz.setup()
    # 1. grid: ΔCS per household
    grid = sc.pivot(index="lambda_mu_case", columns="efficiency", values="dcs_chf_per_household_year").loc[list(cases)]
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    vals = -grid.to_numpy()
    bins = [0, 200, 400, 600, 800, 1000, 5000]
    cols = ["#fbe3e2"] + RED
    cmap, norm = ListedColormap(cols), BoundaryNorm(bins, len(cols))
    ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"−{vals[i, j]:,.0f}", ha="center", va="center", fontsize=10,
                    color=viz.SURFACE if vals[i, j] >= 600 else viz.INK)
    ax.set_xticks(range(len(grid.columns)), [f"{e:.1%}" for e in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"λμ {k} ({v:.2f})" for k, v in cases.items()])
    ax.set_xlabel("marginal-cost saving on the merged banners")
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Consumer loss, CHF per household per year: no plausible saving offsets it")
    viz.source(ax, "Full post-merger equilibrium for each cell; abroad price fixed. Model [calibrated].")
    viz.save(fig, 1, "1_7", "scenario_grid")

    # 2. lines: Migros price change vs efficiency
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (case, lam), colr in zip(cases.items(), [viz.BLUE, viz.ORANGE, viz.AQUA]):
        g = sc[sc.lambda_mu_case == case]
        ax.plot(g.efficiency * 100, g.dp_Migros_pct, color=colr, marker="o", label=f"λμ {case} ({lam:.2f})")
    ax.axhline(0, color=viz.INK_2, linewidth=0.8)
    ax.set_xlabel("marginal-cost saving on the merged banners, %")
    ax.set_ylabel("Migros basket price change, %")
    ax.legend(loc="lower left")
    ax.set_title("A 10% cost saving trims Migros's price rise by only a few points")
    viz.source(ax, "Coop, Denner and other merged banners move in parallel (1_7_scenarios.csv).")
    viz.save(fig, 1, "1_7", "efficiency_lines")

    # 3. small-multiple maps: ΔCS by location for the three λμ cases (0% efficiency), shared scale
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cantons = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)
    loc = pd.read_parquet(PROCESSED / "locations.parquet", columns=["loc_id", "x", "y"]).set_index("loc_id").loc[mkt.loc_id]
    allv = np.concatenate([-v for v in loc_cs.values()])
    bins = [0] + list(np.round(np.quantile(allv, [0.2, 0.4, 0.6, 0.8]), -1)) + [allv.max() + 1]
    cmap, norm = ListedColormap(RED), BoundaryNorm(bins, len(RED))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (case, lam) in zip(axes, cases.items()):
        viz.map_axes(ax, cantons)
        v = -loc_cs[case]
        o = np.argsort(v)
        ax.scatter(loc.x.to_numpy()[o], loc.y.to_numpy()[o], c=v[o], cmap=cmap, norm=norm, s=0.6, linewidths=0, zorder=2)
        avg = np.average(v, weights=mkt.hh)
        ax.set_title(f"λμ {case} ({lam:.2f}): −CHF {avg:,.0f}", fontsize=10.5)
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, orientation="horizontal",
                       fraction=0.05, pad=0.03, aspect=50, ticks=bins[1:-1])
    cb_.ax.set_xticklabels([f"{b:,.0f}" for b in bins[1:-1]], fontsize=8)
    cb_.set_label("consumer loss, CHF per household per year (0% cost saving)", fontsize=8.5, color=viz.INK_2)
    cb_.outline.set_visible(False)
    up = np.average(loc_cs["disappears"], weights=mkt.hh) / np.average(loc_cs["survives"], weights=mkt.hh) - 1
    fig.suptitle(f"If the merged cooperative stops weighing its members, the average loss rises {up:.0%}",
                 x=0.01, y=0.93, ha="left", fontweight="bold")
    viz.save(fig, 1, "1_7", "cs_maps")


if __name__ == "__main__":
    main()
