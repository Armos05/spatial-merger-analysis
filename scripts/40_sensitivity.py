"""Phase 5 step 1: which assumptions drive the harm? Exact sensitivities by implicit differentiation (JAX).

For each assumed parameter ψ_k (σ, α_ref, η, γ, κ, λμ, p_abroad, efficiency) the derivative of the headline
numbers re-fits the model on the way: δ re-calibrated to the same market shares, costs re-recovered from today's
prices, new post-merger equilibrium (src/mxc/structural/sensitivity.py). Tornado: move each ψ_k from its 5th to
95th prior percentile, others fixed; linear prediction from the gradient, checked against exact re-solves.

Priors [assumed] (also used by 41_montecarlo.py):
    σ          normal(0.25, 0.10), truncated to [0.02, 0.60]   nest correlation, the crux (DESIGN §3.2)
    α_ref      lognormal(ln 0.03, 0.20)                         price sensitivity at the reference income
    η          uniform(0.25, 0.75)                              income gradient of price sensitivity
    γ          lognormal(ln 0.06, 0.25)                         disutility per drive minute
    κ          lognormal(ln 0.04, 0.30)                         disutility per minute to the border
    λμ         uniform(0.25, 0.45)                              cooperative weight (Coop alone 0.31, Migros 0.357)
    p_abroad   uniform(60, 80)                                  basket price abroad (Swiss = 100)
    efficiency uniform(0, 0.10) on merging banners' costs       row C only (B: 0)

Run:  conda activate keystone && python scripts/40_sensitivity.py      (needs 10-17)
Outputs (assets/p1/m9/):
  1_9_gradients.csv     d(output)/dψ, elasticities, for rows B and C
  1_9_tornado.csv       harm at the 5th / 95th percentile of each ψ: linear and exact
  1_9_tornado.png       tornado chart (row B harm, CHF per household per year)
  1_9_sigma_map.png     harm by canton at low vs high σ (the crux parameter)
"""
import json
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from scipy import stats  # noqa: E402

from mxc import viz  # noqa: E402
from mxc.structural import calibrate as cb  # noqa: E402
from mxc.structural import demand as dm  # noqa: E402
from mxc.structural import sensitivity as se  # noqa: E402

T20 = runpy.run_path(str(ROOT / "scripts/20_twin.py"), run_name="lib")
P1, INTERIM = ROOT / "assets/p1", ROOT / "data/interim"
OUT = P1 / "m9"
REDS = ["#fbe3e2", "#f6c7c6", "#ee9190", "#e34948", "#b52f2e", "#7a1f1e"]
LABEL = {"sigma": "σ nest correlation", "alpha_ref": "α price sensitivity", "eta": "η income gradient",
         "gamma": "γ travel cost", "kappa": "κ border cost", "lambda_mu": "λμ cooperative weight",
         "p_abroad": "abroad basket price", "efficiency": "cost efficiency"}


def priors():
    """scipy frozen distributions in PSI order (efficiency only for row C)."""
    s0 = 0.25
    a, b = (0.02 - s0) / 0.10, (0.60 - s0) / 0.10
    return {"sigma": stats.truncnorm(a, b, loc=s0, scale=0.10),
            "alpha_ref": stats.lognorm(0.20, scale=0.03), "eta": stats.uniform(0.25, 0.50),
            "gamma": stats.lognorm(0.25, scale=0.06), "kappa": stats.lognorm(0.30, scale=0.04),
            "lambda_mu": stats.uniform(0.25, 0.20), "p_abroad": stats.uniform(60, 20),
            "efficiency": stats.uniform(0.0, 0.10)}


def build():
    mkt, theta = T20["load"]()
    a = dm.to_arrays(mkt)
    par = json.load(open(P1 / "m2/1_2_params.json"))
    sup = json.load(open(P1 / "m4/1_4_supply_params.json"))
    target = cb.targets(par["abroad_share_target"])
    M = se.Model(theta, a, target, np.asarray(theta.p), sup["lambda_mu"]["Migros"])
    pr = pd.read_csv(P1 / "m6/1_6_prices.csv", index_col=0)
    p1 = jnp.asarray(pr.p_post.reindex(dm.BANNERS).to_numpy())
    return mkt, M, p1


def solve_at(M, psi, u0, p_start):
    u, r1 = M.solve_u(psi, u0)
    p, r3 = M.solve_p(psi, u, p_start)
    return u, p, r1, r3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mkt, M, p1 = build()
    psi0 = M.psi0
    t = time.time()
    u0, pB, r1, r3 = solve_at(M, psi0, jnp.zeros(6), p1)
    oB = np.asarray(M.outputs(pB, u0, psi0))
    print("base B", oB, "residuals", r1, r3, f"{time.time() - t:.0f}s", flush=True)
    psiC = psi0.at[7].set(0.05)
    _, pC, _, r3c = solve_at(M, psiC, u0, pB)
    oC = np.asarray(M.outputs(pC, u0, psiC))
    print("base C", oC, r3c, flush=True)
    gB, duB, dpB = M.gradient(psi0, u0, pB)
    gC, _, _ = M.gradient(psiC, u0, pC)
    print("grad B", gB[0].round(1), f"{time.time() - t:.0f}s", flush=True)
    rows = []
    for row, g, o, ps in (("B", gB, oB, psi0), ("C", gC, oC, psiC)):
        for k, name in enumerate(se.PSI):
            for j, out in enumerate(["harm_chf_hh_year", "dp_migros_pct", "dp_coop_pct"]):
                rows.append({"row": row, "parameter": name, "output": out, "base_value": float(ps[k]),
                             "base_output": o[j], "derivative": g[j, k],
                             "elasticity": g[j, k] * float(ps[k]) / o[j] if float(ps[k]) != 0 else np.nan})
    pd.DataFrame(rows).round(6).to_csv(OUT / "1_9_gradients.csv", index=False)

    pr = priors()
    tor = []
    for k, name in enumerate(se.PSI):
        lo, hi = pr[name].ppf(0.05), pr[name].ppf(0.95)
        row, g, o, ps, pstart = ("C", gC, oC, psiC, pC) if name == "efficiency" else ("B", gB, oB, psi0, pB)
        rec = {"parameter": name, "row": row, "base": float(ps[k]), "p05": lo, "p95": hi, "base_harm": o[0]}
        for tag, v in (("p05", lo), ("p95", hi)):
            rec[f"harm_linear_{tag}"] = o[0] + g[0, k] * (v - float(ps[k]))
            ps_k = ps.at[k].set(v)
            u, p, e1, e3 = solve_at(M, ps_k, u0, pstart)
            ex = np.asarray(M.outputs(p, u, ps_k))
            rec[f"harm_exact_{tag}"] = ex[0]
            rec[f"dp_migros_exact_{tag}"] = ex[1]
            rec[f"dp_coop_exact_{tag}"] = ex[2]
            rec[f"resid_{tag}"] = max(e1, e3)
        rec["swing_exact"] = abs(rec["harm_exact_p95"] - rec["harm_exact_p05"])
        tor.append(rec)
        print(name, {k_: round(float(v_), 2) for k_, v_ in rec.items() if k_.startswith("harm")},
              f"{time.time() - t:.0f}s", flush=True)
    tor = pd.DataFrame(tor).sort_values("swing_exact", ascending=False)
    tor.round(4).to_csv(OUT / "1_9_tornado.csv", index=False)

    # harm by canton at σ 5th vs 95th percentile (exact)
    cmap_rows = {}
    for tag in ("p05", "p95"):
        v = tor.set_index("parameter").loc["sigma", tag]
        ps_k = psi0.at[0].set(v)
        u, p, *_ = solve_at(M, ps_k, u0, pB)
        _, _, ls1 = dm.choice_probs(M.theta(ps_k, u, p), M.a)
        _, _, ls0 = dm.choice_probs(M.theta(ps_k, u), M.a)
        al = dm.alpha_types(M.theta(ps_k, u), M.a["type_y"])
        cs = 52 * np.asarray(((ls1 - ls0) / al[None, :] * M.a["type_w"]).sum(1))
        d = pd.DataFrame({"c": mkt.canton, "cs": cs, "w": mkt.hh})
        cmap_rows[tag] = d.groupby("c").apply(lambda g: np.average(g.cs, weights=g.w), include_groups=False)
    can = pd.DataFrame(cmap_rows)
    can.round(2).to_csv(OUT / "1_9_sigma_canton.csv")
    figures(tor, can)


def figures(tor, can):
    viz.setup()
    fig, ax = plt.subplots(figsize=(10, 5))
    tor = tor.sort_values("swing_exact")
    y = np.arange(len(tor))
    for yi, (_, r) in zip(y, tor.iterrows()):
        base = r.base_harm
        lo_v, hi_v = r.harm_exact_p05, r.harm_exact_p95
        ax.barh(yi, lo_v - base, left=base, color=viz.BLUE, height=0.55)
        ax.barh(yi, hi_v - base, left=base, color=viz.ORANGE, height=0.55)
        ax.plot([r.harm_linear_p05, r.harm_linear_p95], [yi, yi], "|", color=viz.INK, ms=12, mew=1.5)
        ends = [lo_v, hi_v, r.harm_linear_p05, r.harm_linear_p95]
        ax.text(min(ends) - 8, yi, f"{min(lo_v, hi_v):.0f}", va="center", ha="right", fontsize=8)
        ax.text(max(ends) + 8, yi, f"{max(lo_v, hi_v):.0f}", va="center", ha="left", fontsize=8)
    fmt = {"sigma": "{:.2f}", "alpha_ref": "{:.3f}", "eta": "{:.2f}", "gamma": "{:.3f}", "kappa": "{:.3f}",
           "lambda_mu": "{:.2f}", "p_abroad": "{:.0f}", "efficiency": "{:.0%}"}
    ax.set_yticks(y, [f"{LABEL[p]}\n{fmt[p].format(lo)} – {fmt[p].format(hi)}" + (" (row C)" if p == "efficiency" else "")
                      for p, lo, hi in zip(tor.parameter, tor.p05, tor.p95)], fontsize=8.5)
    ax.axvline(tor.base_harm[tor.row == "B"].iloc[0], color=viz.INK_2, lw=1)
    ax.set_xlabel("Δ consumer surplus, CHF per household per year (merger, no remedy)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=viz.BLUE, label="parameter at 5th percentile (exact)"),
                       Patch(color=viz.ORANGE, label="parameter at 95th percentile (exact)"),
                       Line2D([], [], marker="|", ls="none", color=viz.INK, ms=10,
                              label="linear prediction (implicit gradient)")], fontsize=8.5, loc="lower left")
    allv = tor[["harm_exact_p05", "harm_exact_p95", "harm_linear_p05", "harm_linear_p95"]].to_numpy()
    ax.set_xlim(allv.min() - 60, allv.max() + 60)
    top = tor.iloc[-1]
    ax.set_title(f"The harm hinges on {LABEL[top.parameter].split(' ', 1)[1]}: CHF {top.harm_exact_p05:.0f} to "
                 f"{top.harm_exact_p95:.0f} across its plausible range; nothing flips the sign [calibrated]",
                 fontsize=11)
    viz.source(ax, "Each bar: one parameter at its 5th / 95th prior percentile, the rest at base; model re-calibrated "
                   "to the same shares and costs each time (exact). Ticks: first-order prediction from JAX "
                   "implicit differentiation.")
    viz.save(fig, 1, "1_9", "tornado")

    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cant = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)[["ref", "geometry"]]
    g = cant.merge(-can, left_on="ref", right_index=True)
    hi = np.ceil(g[["p05", "p95"]].max().max() / 100) * 100
    lo = np.floor(g[["p05", "p95"]].min().min() / 100) * 100
    bins = np.arange(lo, hi + 1, 100)
    cols = REDS[-(len(bins) - 1):]
    cmap, norm = ListedColormap(cols), BoundaryNorm(bins, len(cols))
    s = tor.set_index("parameter").loc["sigma"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, ttl in ((axes[0], "p05", f"σ = {s.p05:.2f} (weak nests: diffuse substitution)"),
                         (axes[1], "p95", f"σ = {s.p95:.2f} (strong nests: Migros–Coop close substitutes)")):
        g.plot(ax=ax, column=col, cmap=cmap, norm=norm, edgecolor=viz.SURFACE, linewidth=0.6)
        viz.map_axes(ax, cant)
        ax.set_title(ttl, fontsize=10.5)
    cb_ = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, orientation="horizontal", fraction=0.05,
                       pad=0.03, shrink=0.5)
    cb_.set_ticks(bins)
    cb_.set_ticklabels([f"{b:.0f}" for b in bins])
    cb_.set_label("loss of consumer surplus, CHF per household per year (merger, no remedy)", fontsize=8.5)
    rho = g[["p05", "p95"]].rank().corr().iloc[0, 1]
    d_ = (g.p95 - g.p05)
    fig.suptitle(f"A higher σ raises the loss in every canton (by CHF {d_.min():.0f}-{d_.max():.0f}); the ranking of "
                 f"cantons barely moves (rank correlation {rho:.2f}) [calibrated]",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    viz.source(axes[0], "Exact re-solve at the 5th and 95th percentile of the σ prior, re-calibrated each time.")
    viz.save(fig, 1, "1_9", "sigma_map")


def redraw():
    """Figures only, from the saved CSVs."""
    tor = pd.read_csv(OUT / "1_9_tornado.csv")
    can = pd.read_csv(OUT / "1_9_sigma_canton.csv", index_col=0)
    figures(tor, can)


if __name__ == "__main__":
    redraw() if "--figures" in sys.argv else main()
