"""Pillar 2 step 6: the validation scorecard — one table and one exhibit for tests T1-T3 on the twin [sim].

For each test: the quantity, its true scale, and for three methods (traditional, a plausible modern alternative
that fails, the best data-driven method) the bias, RMSE, error relative to the truth and 95% coverage.

Run:  conda activate keystone && python scripts/25_scorecard.py      (needs 22-24)
Outputs (assets/p2/m6/):
  2_6_scorecard.csv     one row per test × method
  2_6_scorecard.png     relative error of traditional vs modern alternative vs best method, per test
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mxc import viz  # noqa: E402

P2 = ROOT / "assets/p2"
OUT = P2 / "m6"
ROLE_COL = {"traditional": viz.INK_2, "modern alternative": viz.ORANGE, "best data-driven": viz.BLUE}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t1 = pd.read_csv(P2 / "m3/2_3_scorecard.csv").query("data == 'observed'").set_index("method")
    e1 = pd.read_csv(P2 / "m3/2_3_estimates.csv").query("data == 'observed'")
    scale1 = e1[e1.band.isin(["0-1 km", "1-2 km", "2-3 km"])].drop_duplicates(["band", "group"]).truth_loss_pct.mean()
    t2 = pd.read_csv(P2 / "m4/2_4_scorecard.csv").query("level == 'banner'").set_index("method")
    t3 = pd.read_csv(P2 / "m5/2_5_scorecard.csv").set_index("method")
    g3 = pd.read_csv(P2 / "m5/2_5_guppi_scorecard.csv").set_index("method")
    truth_e = 4.373

    rows = []
    def add(test, quantity, scale, unit, role, method, bias, rmse, cov, note):
        rows.append({"test": test, "quantity": quantity, "true_scale": scale, "unit": unit, "role": role,
                     "method": method, "bias": bias, "rmse": rmse, "relative_error_pct": 100 * rmse / scale,
                     "coverage_95": cov, "note": note})
    q1 = "loss of stores within 3 km of a new discounter (12 cells)"
    add("T1 entry", q1, scale1, "pp", "traditional", "TWFE store-month", t1.loc["TWFE (store-month)", "bias_within_3km_pp"],
        t1.loc["TWFE (store-month)", "rmse_within_3km_pp"], t1.loc["TWFE (store-month)", "coverage_95"], "monthly, nearest entrant only")
    add("T1 entry", q1, scale1, "pp", "modern alternative", "stacked event study", t1.loc["stacked event study", "bias_within_3km_pp"],
        t1.loc["stacked event study", "rmse_within_3km_pp"], t1.loc["stacked event study", "coverage_95"], "clean controls fail: wide spillovers")
    add("T1 entry", q1, scale1, "pp", "best data-driven", "pooled exposure regression", t1.loc["pooled exposure", "bias_within_3km_pp"],
        t1.loc["pooled exposure", "rmse_within_3km_pp"], t1.loc["pooled exposure", "coverage_95"], "far bands understated")
    q2 = "own-price elasticity by banner (12 cells)"
    for role, m, note in [("traditional", "OLS-FE", "prices react to demand"),
                          ("modern alternative", "DML-PLR (ML controls)", "ML removes the good price variation"),
                          ("best data-driven", "DML-PLIV (ML + cost)", "cost instrument + sister-store controls")]:
        add("T2 demand", q2, truth_e, "elasticity", role, m, t2.loc[m, "bias"], t2.loc[m, "rmse"], t2.loc[m, "coverage_95"], note)
    q3 = "own-price elasticity of each store (4,565)"
    for role, m, note in [("traditional", "national", "one number for all stores"),
                          ("modern alternative", "store-by-store IV", "unbiased but noisy"),
                          ("best data-driven", "causal forest", "pools similar stores; shrinks extremes")]:
        add("T3 local", q3, truth_e, "elasticity", role, m, t3.loc[m, "bias"], t3.loc[m, "rmse"],
            t3.loc[m, "coverage_95"] if "coverage_95" in t3 else np.nan, note)
    q4 = "local GUPPI ≥ 10%: share of 2,777 stores misclassified"
    for role, m, note in [("traditional", "OLS national", "OLS margins flag every store"),
                          ("modern alternative", "store-by-store IV", "lowest GUPPI RMSE (noise averages out)"),
                          ("best data-driven", "causal forest", "fewest wrong flags")]:
        rows.append({"test": "T3 screen", "quantity": q4, "true_scale": 100.0, "unit": "% of stores", "role": role,
                     "method": m, "bias": g3.loc[m, "bias_pp"], "rmse": g3.loc[m, "rmse_pp"],
                     "relative_error_pct": g3.loc[m, "misclassified_pct"], "coverage_95": np.nan, "note": note})
    sc = pd.DataFrame(rows)
    sc.round(4).to_csv(OUT / "2_6_scorecard.csv", index=False)
    print(sc[["test", "role", "method", "bias", "rmse", "relative_error_pct", "coverage_95"]].round(3).to_string(index=False))
    figure(sc)


def figure(sc):
    viz.setup()
    tests = ["T1 entry", "T2 demand", "T3 local", "T3 screen"]
    head = {"T1 entry": "T1 entry: loss within 3 km (error % of true loss)",
            "T2 demand": "T2 demand: banner elasticities (error % of truth)",
            "T3 local": "T3 local: store elasticities (error % of truth)",
            "T3 screen": "T3 screen: GUPPI ≥ 10% (% of stores misclassified)"}
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y, ticks, labels, heads = 0, [], [], []
    for t in tests:
        g = sc[sc.test == t].set_index("role")
        heads.append((y + 0.2, head[t]))
        y -= 1
        for role, colr in ROLE_COL.items():
            x = g.loc[role, "relative_error_pct"]
            ax.plot([0.8, x], [y, y], color=viz.GRID, linewidth=2, zorder=1)
            ax.scatter(x, y, s=80, color=colr, zorder=3, edgecolors=viz.SURFACE, linewidths=1,
                       label=role if t == tests[0] else None)
            ax.text(x * 1.12, y, f"{x:.1f}%" if x < 10 else f"{x:.0f}%", va="center", fontsize=8.5)
            ticks.append(y)
            labels.append(g.loc[role, "method"])
            y -= 1
        y -= 0.6
    for yh, txt in heads:
        ax.text(0.81, yh - 0.35, txt, fontsize=9.5, fontweight="bold", color=viz.INK, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 500)
    ax.set_xticks([1, 3, 10, 30, 100, 300], ["1%", "3%", "10%", "30%", "100%", "300%"])
    ax.set_yticks(ticks, labels, fontsize=8.5)
    ax.set_ylim(y + 0.8, 0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("error vs the known truth (log scale; lower is better)")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.set_title("The right data-driven design cuts errors 2 to 50 fold; naive ML or 'modern' designs\n"
                 "can be as wrong as the traditional ones [sim]", loc="left", fontsize=11)
    viz.source(ax, "RMSE / true scale (T1, T2, T3 local); misclassification rate (T3 screen). Steps 2.3-2.5. [sim]")
    viz.save(fig, 2, "2_6", "scorecard")


if __name__ == "__main__":
    main()
