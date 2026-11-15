"""Phase 5 step 2: Monte Carlo — 90% intervals on the headline numbers (10,000 draws).

Draw all assumed parameters jointly and independently from their priors (40_sensitivity.py) and compute the
headline numbers for each draw. A full re-calibration + equilibrium per draw takes ~30 s, so 10,000 draws use
a second-order surrogate built from exact derivatives:
    y(ψ) ≈ y0 + g'(ψ − ψ0) + ½ Σ_k h_k (ψ_k − ψ0_k)²
g = implicit-function gradient (1_9_gradients.csv); h_k = curvature fitted to the exact re-solves at the 5th and
95th percentile of ψ_k (1_9_tornado.csv). The surrogate is validated against N_EXACT fully re-solved random draws.

Rows D (divestitures) are not re-solved per draw: their harm is scaled by the base ratio row/B [approx].
Rows E/F (data ring-fence) add the Pillar 3 increments (assets/p3/m6), whose own simulation spread is not included.

Run:  conda activate keystone && python scripts/41_montecarlo.py      (needs 40, 35)
Outputs (assets/p1/m10/):
  1_10_draws.csv            the 10,000 parameter draws and outputs (rows B, C)
  1_10_validation.csv       surrogate vs exact re-solve on random draws
  1_10_intervals.csv        median and 90% interval of every headline number
  1_10_outcome_table.csv    outcome table A-G with 90% intervals
  1_10_intervals.png        interval chart of harm by scenario
  1_10_validation.png       surrogate vs exact   (the map for this phase is 1_9_sigma_map.png)
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

from mxc import viz  # noqa: E402
from mxc.structural import sensitivity as se  # noqa: E402

T40 = runpy.run_path(str(ROOT / "scripts/40_sensitivity.py"), run_name="lib")
P1, P3, INTERIM = ROOT / "assets/p1", ROOT / "assets/p3", ROOT / "data/interim"
OUT = P1 / "m10"
N_DRAWS, N_EXACT, SEED = 10_000, 30, 2027
OUTS = ["harm_chf_hh_year", "dp_migros_pct", "dp_coop_pct"]


def draw(n, rng, eff):
    pr = T40["priors"]()
    D = np.column_stack([pr[k].rvs(n, random_state=rng) if k != "efficiency" else np.full(n, eff) for k in se.PSI])
    return D


def surrogate(row, grad, tor, psi0, y0):
    """y0 (3,), grad (3, 8): linear + per-parameter curvature (harm only; prices linear)."""
    h = np.zeros(len(se.PSI))
    t = tor.set_index("parameter")
    for k, name in enumerate(se.PSI):
        if name not in t.index or name == "efficiency":
            continue
        r = t.loc[name]
        d = np.array([r.p05 - psi0[k], r.p95 - psi0[k]])
        res = np.array([r.harm_exact_p05, r.harm_exact_p95]) - (r.base_harm + grad[0, k] * d)
        h[k] = 2 * (res * d ** 2).sum() / (d ** 4).sum()

    def f(P):
        dP = P - psi0[None, :]
        Y = y0[None, :] + dP @ grad.T
        Y[:, 0] += 0.5 * (dP ** 2) @ h
        return Y
    return f, h


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    G = pd.read_csv(P1 / "m9/1_9_gradients.csv")
    tor = pd.read_csv(P1 / "m9/1_9_tornado.csv")
    mkt, M, p1 = T40["build"]()
    psi0 = np.asarray(M.psi0)
    fs = {}
    for row, eff in (("B", 0.0), ("C", 0.05)):
        g = G[G.row == row].pivot(index="output", columns="parameter", values="derivative").loc[OUTS, se.PSI].to_numpy()
        y0 = G[(G.row == row) & (G.parameter == "sigma")].set_index("output").base_output.loc[OUTS].to_numpy()
        ps = psi0.copy()
        ps[7] = eff
        fs[row] = surrogate(row, g, tor[tor.row == "B"], ps, y0)[0], ps, y0    # curvature from row B re-solves
    # draws
    PB, PC = draw(N_DRAWS, rng, 0.0), None
    PC = PB.copy()
    PC[:, 7] = 0.05
    YB, YC = fs["B"][0](PB), fs["C"][0](PC)
    dr = pd.DataFrame(PB[:, :7], columns=se.PSI[:7])
    for j, o in enumerate(OUTS):
        dr[f"B_{o}"], dr[f"C_{o}"] = YB[:, j], YC[:, j]
    dr.round(5).to_csv(OUT / "1_10_draws.csv", index=False)

    # validation: exact re-solve on random draws (row B)
    val = []
    t = time.time()
    u0 = jnp.zeros(6)
    pB, *_ = M.solve_p(M.psi0, u0, p1)
    for i in range(N_EXACT):
        ps = jnp.asarray(PB[i])
        u, p, e1, e3 = T40["solve_at"](M, ps, u0, pB)
        ex = np.asarray(M.outputs(p, u, ps))
        val.append({"draw": i, **{f"exact_{o}": ex[j] for j, o in enumerate(OUTS)},
                    **{f"surrogate_{o}": YB[i, j] for j, o in enumerate(OUTS)}, "resid": max(e1, e3)})
        print(i, ex.round(1), YB[i].round(1), f"{time.time() - t:.0f}s", flush=True)
    val = pd.DataFrame(val)
    val.round(4).to_csv(OUT / "1_10_validation.csv", index=False)

    # intervals
    q = lambda x: np.percentile(x, [5, 50, 95])  # noqa: E731
    rows = []
    for row in ("B", "C"):
        for o in OUTS:
            lo, med, hi = q(dr[f"{row}_{o}"])
            rows.append({"row": row, "output": o, "base": fs[row][2][OUTS.index(o)], "p05": lo, "p50": med, "p95": hi})
    iv = pd.DataFrame(rows)
    iv.round(3).to_csv(OUT / "1_10_intervals.csv", index=False)
    print(iv.round(1).to_string(index=False))

    # outcome table with intervals
    ot = pd.read_csv(P1 / "m8/1_8_outcome_table.csv")
    p3 = P3 / "m6/3_6_outcome_rows.csv"
    extra = pd.read_csv(p3) if p3.exists() else None
    hB = dr.B_harm_chf_hh_year.to_numpy()
    hC = dr.C_harm_chf_hh_year.to_numpy()
    baseB = ot.set_index("row").loc["B", "dcs_chf_hh_year"]
    baseC = ot.set_index("row").loc["C", "dcs_chf_hh_year"]
    bn_per = ot.set_index("row").loc["B", "dcs_chf_bn_year"] / baseB
    out = []
    for _, r in ot.iterrows():
        if r.row in ("A", "G"):
            sim = np.zeros(N_DRAWS)
        elif r.row == "B":
            sim = hB
        elif r.row == "C":
            sim = hC
        elif str(r.row).startswith("D"):
            sim = hC * r.dcs_chf_hh_year / baseC
        else:
            sim = None
        if sim is None and extra is not None and r.row in set(extra.row):
            e = extra.set_index("row").loc[r.row]
            anchor = hC if e.anchor == "C" else hC * ot.set_index("row").loc[e.anchor, "dcs_chf_hh_year"] / baseC
            sim = anchor + e.data_increment_chf_hh_year
            sim_alt = anchor + e.alt_increment_if_personal_pricing_is_new
            r = r.copy()
            r["alt"] = e.alt_dcs_if_personal_pricing_is_new
            r["scenario"] = e.scenario
            r["dcs_chf_hh_year"] = e.dcs_chf_hh_year
        if sim is None:
            out.append({"row": r.row, "scenario": r.scenario})
            continue
        lo, med, hi = q(sim)
        rec = {"row": r.row, "scenario": r.scenario, "dcs_chf_hh_year": r.dcs_chf_hh_year,
               "dcs_p05": lo, "dcs_p50": med, "dcs_p95": hi,
               "dcs_bn_p05": lo * bn_per, "dcs_bn_p95": hi * bn_per,
               "dp_migros_pct": r.get("dp_Migros_pct", np.nan), "dp_coop_pct": r.get("dp_Coop_pct", np.nan)}
        if "alt" in r.index:
            a05, _, a95 = q(sim_alt)
            rec.update({"alt_dcs_if_personal_pricing_is_new": r.alt, "alt_p05": a05, "alt_p95": a95})
        out.append(rec)
    if extra is not None:                                  # data rows that Pillar 1 does not have (B+, B++, E')
        for _, e in extra[~extra.row.isin(ot.row)].iterrows():
            anchor = {"B": hB, "C": hC}.get(e.anchor, hC * ot.set_index("row").loc[e.anchor, "dcs_chf_hh_year"] / baseC)
            lo, med, hi = q(anchor + e.data_increment_chf_hh_year)
            a05, _, a95 = q(anchor + e.alt_increment_if_personal_pricing_is_new)
            out.append({"row": e.row, "scenario": e.scenario, "dcs_chf_hh_year": e.dcs_chf_hh_year,
                        "dcs_p05": lo, "dcs_p50": med, "dcs_p95": hi, "dcs_bn_p05": lo * bn_per,
                        "dcs_bn_p95": hi * bn_per, "alt_dcs_if_personal_pricing_is_new":
                        e.alt_dcs_if_personal_pricing_is_new, "alt_p05": a05, "alt_p95": a95})
    otab = pd.DataFrame(out)
    order = ["A", "B", "B+", "B++", "C", "E", "E'"] + [r for r in otab.row if str(r).startswith("D")] + ["F", "G"]
    otab = otab.set_index("row").loc[[r for r in order if r in set(otab.row)]].reset_index()
    otab.round(2).to_csv(OUT / "1_10_outcome_table.csv", index=False)
    print(otab.round(0).to_string(index=False))
    figures(otab, val, dr, iv)


def figures(otab, val, dr, iv):
    viz.setup()
    d = otab.dropna(subset=["dcs_p05"])
    d = d[~d.row.isin(["A", "G"])].iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(d) + 1.2))
    y = np.arange(len(d))
    ax.hlines(y, d.dcs_p05, d.dcs_p95, color=viz.BLUE, lw=4, alpha=0.6)
    ax.scatter(d.dcs_chf_hh_year, y, color=viz.INK, zorder=3, s=30, label="base; bar = 90% interval")
    if "alt_p05" in d:
        m = d.alt_p05.notna().to_numpy()
        ax.hlines(y[m], d.alt_p05[m], d.alt_p95[m], color=viz.ORANGE, lw=2, alpha=0.9)
        ax.scatter(d.alt_dcs_if_personal_pricing_is_new[m], y[m], color=viz.ORANGE, zorder=3, s=26,
                   label="data rows if personal pricing arrives with the merger")
        ax.legend(fontsize=8, loc="lower left")
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r.dcs_p95 + 10, yi, f"{r.dcs_chf_hh_year:.0f}  [{r.dcs_p05:.0f}, {r.dcs_p95:.0f}]", va="center",
                fontsize=8)
    import textwrap
    lab = [f"{r.row}  " + "\n     ".join(textwrap.wrap(r.scenario, 50)) for _, r in d.iterrows()]
    ax.set_yticks(y, lab, fontsize=8)
    ax.axvline(0, color=viz.INK_2, lw=0.8)
    ax.set_xlim(min(d.dcs_p05.min(), d.get("alt_p05", d.dcs_p05).min()) * 1.1, 260)
    ax.set_xlabel("Δ consumer surplus, CHF per household per year (dot = base; bar = 90% interval)")
    b = otab.set_index("row").loc["B"]
    ax.set_title(f"Even at the favourable end of the assumptions, the unremedied merger costs households "
                 f"CHF {-b.dcs_p95:.0f}+ a year [calibrated]", fontsize=11)
    viz.source(ax, f"{N_DRAWS:,} joint draws of σ, α, η, γ, κ, λμ, abroad price (priors in 40_sensitivity.py); "
                   "second-order surrogate from JAX implicit derivatives + exact re-solves. D rows scaled from C "
                   "[approx]; E/F add Pillar 3 data increments [sim].")
    viz.save(fig, 1, "1_10", "intervals")

    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.scatter(val.exact_harm_chf_hh_year, val.surrogate_harm_chf_hh_year, color=viz.BLUE, s=25)
    lo = min(val.exact_harm_chf_hh_year.min(), val.surrogate_harm_chf_hh_year.min()) - 20
    hi = max(val.exact_harm_chf_hh_year.max(), val.surrogate_harm_chf_hh_year.max()) + 20
    ax.plot([lo, hi], [lo, hi], color=viz.MUTED, ls="--")
    err = (val.surrogate_harm_chf_hh_year - val.exact_harm_chf_hh_year).abs()
    ax.set_xlabel("exact re-solve (CHF / household / year)")
    ax.set_ylabel("surrogate")
    ax.set_title(f"Surrogate vs exact on {len(val)} random draws:\nmedian absolute error CHF {err.median():.0f}, "
                 f"max {err.max():.0f}", fontsize=10.5)
    viz.source(ax, "Row B harm; each point = one joint parameter draw re-calibrated and re-solved.")
    viz.save(fig, 1, "1_10", "validation")


def redraw():
    """Figures only, from the saved CSVs."""
    figures(pd.read_csv(OUT / "1_10_outcome_table.csv"), pd.read_csv(OUT / "1_10_validation.csv"),
            pd.read_csv(OUT / "1_10_draws.csv"), pd.read_csv(OUT / "1_10_intervals.csv"))


if __name__ == "__main__":
    redraw() if "--figures" in sys.argv else main()
