"""Phase 2 step 6: merger simulation (base case: Migros + Coop merge, cooperative weight λμ survives).

Run:  conda activate keystone && python scripts/15_simulation.py      (needs 10-13)
Outputs (assets/p1/m6/):
  1_6_validation.csv     pre-merger equilibrium from a +5% start returns today's prices
  1_6_convergence.csv    post-merger fixed-point history
  1_6_prices.csv         banner prices before / after, % change; markups and sales
  1_6_shares.csv         revenue shares before / after, incl. abroad
  1_6_welfare.csv        ΔCS (CHF/household/yr and CHF m/yr), Δprofit by firm, net
  1_6_cs_canton.csv, 1_6_cs_income.csv   distribution of the loss
  1_6_cs_map.png, 1_6_price_changes.png, 1_6_cs_income.png
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
OUT = ROOT / "assets/p1/m6"
P1 = ROOT / "assets/p1"
RED = ["#f6c7c6", "#ee9190", "#e34948", "#b52f2e", "#7a1f1e"]


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


def revenue_shares(Q, p, s0_total, p_abroad):
    rev = np.r_[Q * p, s0_total * p_abroad]
    return rev / rev.sum()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mkt, theta, lm, cost = load()
    a = dm.to_arrays(mkt)
    bd = sim.make_banner_demand(a)
    p0 = np.asarray(theta.p)
    n_lr = np.asarray(a["hh"])[:, None] * np.asarray(a["type_w"])

    # --- validation: pre-merger ownership from a +5% start must return today's prices
    p_chk, h_chk = sim.solve_equilibrium(theta, bd, cost, lm, su.OWNER_PRE, p_start=p0 * 1.05)
    val = pd.DataFrame({"today": p0, "solved_pre_merger": p_chk, "abs_error": np.abs(p_chk - p0)}, index=dm.BANNERS)
    val.to_csv(OUT / "1_6_validation.csv")
    print(f"validation: max |p_solved − p_today| = {val.abs_error.max():.1e} after {len(h_chk) - 1} iterations")

    # --- post-merger equilibrium (λμ survives at its pre-merger value)
    p1, hist = sim.solve_equilibrium(theta, bd, cost, lm, su.OWNER_POST)
    hist.to_csv(OUT / "1_6_convergence.csv", index=False)
    print(f"post-merger converged in {len(hist) - 1} iterations, last gap {hist.max_abs_price_gap.iloc[-1]:.1e}")
    th1 = theta._replace(p=jnp.asarray(p1))
    Q0, A0, ls0, S0, s00 = bd(theta)
    Q1, A1, ls1, S1, s01 = bd(th1)
    Q0, Q1 = np.asarray(Q0), np.asarray(Q1)
    ab0, ab1 = float((n_lr * np.asarray(s00)).sum()), float((n_lr * np.asarray(s01)).sum())

    prices = pd.DataFrame({"owner_pre": [su.OWNER_PRE[b] for b in dm.BANNERS], "p_pre": p0, "p_post": p1,
                           "pct_change": (p1 / p0 - 1) * 100, "markup_pre": p0 - cost, "markup_post": p1 - cost,
                           "baskets_pre": Q0, "baskets_post": Q1, "pct_change_baskets": (Q1 / Q0 - 1) * 100},
                          index=dm.BANNERS)
    prices.round(4).to_csv(OUT / "1_6_prices.csv")
    print(prices.round(2).to_string())

    sh = pd.DataFrame({"pre": revenue_shares(Q0, p0, ab0, theta.p_abroad),
                       "post": revenue_shares(Q1, p1, ab1, theta.p_abroad)}, index=dm.BANNERS + ["abroad"])
    sh["change_pp"] = (sh.post - sh.pre) * 100
    sh.round(5).to_csv(OUT / "1_6_shares.csv")

    # --- welfare
    dcs = sim.delta_cs(theta, th1, a)                                          # (L,R) CHF / household / week
    dcs_year = dcs * sim.WEEKS
    hh = n_lr.sum()
    total_cs = (n_lr * dcs_year).sum()
    firms = sorted(set(su.OWNER_POST.values()))
    owner = np.array([su.OWNER_POST[b] for b in dm.BANNERS])
    prof = lambda p, Q: {f: float(((p - cost) * Q)[owner == f].sum() * sim.WEEKS) for f in firms}  # noqa: E731
    pr0, pr1 = prof(p0, Q0), prof(p1, Q1)
    wel = pd.DataFrame([{"item": "consumer surplus, CHF per household per year", "value": total_cs / hh},
                        {"item": "consumer surplus, CHF m per year", "value": total_cs / 1e6}]
                       + [{"item": f"profit {f}, CHF m per year", "value": (pr1[f] - pr0[f]) / 1e6} for f in firms]
                       + [{"item": "net (CS + all profits), CHF m per year",
                           "value": (total_cs + sum(pr1.values()) - sum(pr0.values())) / 1e6}])
    wel.round(2).to_csv(OUT / "1_6_welfare.csv", index=False)
    print(wel.round(1).to_string(index=False))

    loc_cs = (n_lr * dcs_year).sum(1) / n_lr.sum(1)                            # CHF / household / yr by location
    cs_can = pd.DataFrame({"canton": mkt.canton, "hh": mkt.hh, "cs": loc_cs}).groupby("canton").apply(
        lambda g: pd.Series({"households": g.hh.sum(), "cs_chf_per_household_year": np.average(g.cs, weights=g.hh),
                             "cs_chf_m_year": (g.cs * g.hh).sum() / 1e6}), include_groups=False).sort_values(
        "cs_chf_per_household_year")
    cs_can.round(2).to_csv(OUT / "1_6_cs_canton.csv")
    ty = np.array(list(dm.INCOME_TYPES))
    y = np.asarray(mkt.type_y)
    cs_inc = pd.DataFrame({"income_type": ty, "representative_income": y, "households": n_lr.sum(0),
                           "cs_chf_per_household_year": (n_lr * dcs_year).sum(0) / n_lr.sum(0)})
    cs_inc["pct_of_income"] = -cs_inc.cs_chf_per_household_year / y * 100
    cs_inc.round(3).to_csv(OUT / "1_6_cs_income.csv", index=False)
    print(cs_can.round(1).head(5).to_string(), "\n", cs_inc.round(3).to_string(index=False))

    figures(mkt, loc_cs, prices, cs_inc, total_cs / hh)


def figures(mkt, loc_cs, prices, cs_inc, avg):
    viz.setup()
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cantons = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)
    loc = pd.read_parquet(PROCESSED / "locations.parquet", columns=["loc_id", "x", "y"]).set_index("loc_id").loc[mkt.loc_id]

    # 1. map of the consumer loss
    loss = -loc_cs
    q = np.quantile(loss, [0.1, 0.3, 0.5, 0.7, 0.9])
    bins = [0] + list(np.round(q[1:], -1)) + [loss.max() + 1]
    bins = sorted(set(bins))
    cols = RED[: len(bins) - 1]
    cmap, norm = ListedColormap(cols), BoundaryNorm(bins, len(cols))
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    viz.map_axes(ax, cantons)
    o = np.argsort(loss)
    ax.scatter(loc.x.to_numpy()[o], loc.y.to_numpy()[o], c=loss[o], cmap=cmap, norm=norm, s=1.2, linewidths=0, zorder=2)
    ax.set_title(f"Merger costs the average household CHF {-avg:,.0f} a year; most in Basel, Bern, Zurich")
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal",
                       fraction=0.04, pad=0.02, aspect=40, ticks=bins[1:-1])
    cb_.ax.set_xticklabels([f"{b:,.0f}" for b in bins[1:-1]], fontsize=8)
    cb_.set_label("consumer loss, CHF per household per year (compensating variation)", fontsize=8.5, color=viz.INK_2)
    cb_.outline.set_visible(False)
    viz.source(ax, "Base case: Migros + Coop merge, national prices re-set, cooperative weight λμ = 0.357 survives, "
                   "no efficiencies. Model [calibrated].")
    viz.save(fig, 1, "1_6", "cs_map")

    # 2. price changes by banner
    t = prices.sort_values("pct_change")
    col = [viz.BLUE if o == "Migros group" else viz.ORANGE if o == "Coop group" else viz.MUTED for o in t.owner_pre]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(t.index, t["pct_change"], color=col, height=0.6)
    for i, v in enumerate(t["pct_change"]):
        ax.text(v + 0.15, i, f"{v:+.1f}%", va="center", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("change in national basket price, %")
    mig = prices.loc["Migros", "pct_change"]
    coop = prices.loc["Coop", "pct_change"]
    ax.set_title(f"Migros +{mig:.0f}%, Coop +{coop:.0f}%; Aldi and Lidl follow by about 1%")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=viz.BLUE, label="Migros group"), Patch(color=viz.ORANGE, label="Coop group"),
                       Patch(color=viz.MUTED, label="rivals (best response)")], loc="lower right")
    viz.source(ax, "Full Bertrand equilibrium, all 12 banners re-optimise; abroad price fixed. Model [calibrated].")
    viz.save(fig, 1, "1_6", "price_changes")

    # 3. loss by income type: CHF and % of income
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].bar(cs_inc.income_type, -cs_inc.cs_chf_per_household_year, color=viz.BLUE, width=0.6)
    axes[0].set_title("CHF per household per year", fontsize=10.5)
    axes[1].bar(cs_inc.income_type, cs_inc.pct_of_income, color=viz.BLUE, width=0.6)
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.1f}%")
    axes[1].set_title("as % of net income", fontsize=10.5)
    for ax in axes:
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Richer households lose more francs; poorer households lose more of their income",
                 x=0.01, ha="left", fontweight="bold")
    fig.subplots_adjust(top=0.8, wspace=0.3)
    viz.source(axes[0], "Income types from ESTV 2023 taxpayer classes (net income). Model [calibrated].")
    viz.save(fig, 1, "1_6", "cs_income")


if __name__ == "__main__":
    main()
