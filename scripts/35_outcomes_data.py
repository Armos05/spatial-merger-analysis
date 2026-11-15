"""Pillar 3 step 6: outcome-table rows with the data channel (E, F and the 'with personal pricing' variants).

Pillar 1 rows (assets/p1/m8) assume uniform national prices. Pillar 3 (3.3) shows personal pricing on card
data already costs shoppers ~CHF 211/yr under competition. The merger-specific data effect is therefore the change
in that personalisation harm caused by the merger (main convention):
    I(regime) = [ΔCS(personal merger, regime) − ΔCS(uniform merger)] − [ΔCS(personal competition) − ΔCS(uniform today)]
Alternative convention (personal pricing arrives only with the merger): inc(regime) = P-M(regime) − U-M.
Rows (CHF per household per year; increments measured at row-B prices, applied to other rows [approx]):
    B+   B with pooled-data personal pricing, today's data quality      = B  + I(pooled)
    B++  B with pooled-data personal pricing, perfect data              = B  + I(perfect data)
    E    C + data ring-fence (privacy filter, ε = 1)                    = C  + I(DP ε = 1)
    E'   C + Cumulus and Supercard kept apart                           = C  + I(cards apart)
    F    D5 + data ring-fence (ε = 1)                                   = D5 + I(DP ε = 1)

Run:  conda activate keystone && python scripts/35_outcomes_data.py      (needs 17, 32, 33)
Outputs (assets/p3/m6/):
  3_6_outcome_rows.csv     the new rows (read by 41_montecarlo.py for intervals)
  3_6_outcome_table.csv    full table A-G with the data rows
  3_6_waterfall.png        harm by package: price harm + data harm
  3_6_data_share_map.png   data harm as % of the uniform-price harm, by home canton
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from mxc import viz  # noqa: E402

P1, P3A = ROOT / "assets/p1", ROOT / "assets/p3"
P3, INTERIM, EV = ROOT / "data/processed/p3", ROOT / "data/interim", ROOT / "data/processed/twin/events"
OUT = P3A / "m6"
BLUES = ["#d6e6f8", "#a9cbf0", "#6fa6e6", "#2a78d6", "#1d58a3", "#123a6e"]
EPS = 1.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ot = pd.read_csv(P1 / "m8/1_8_outcome_table.csv")
    sc = pd.read_csv(P3A / "m3/3_3_scenarios.csv").set_index("scenario").dcs_chf_hh_year
    dp = pd.read_csv(P3A / "m4/3_4_dp_curve.csv")
    tb = pd.read_csv(P3A / "m3/3_3_two_by_two.csv").set_index("item").value
    pc = sc["P-C"] - sc["U-C"]                                         # personalisation harm today
    alt = {"pooled": sc["P-M"] - sc["U-M"], "perfect": sc["P-M oracle"] - sc["U-M"],
           "apart": sc["P-M ring-fenced"] - sc["U-M"],
           "dp": float(dp.loc[dp.epsilon == EPS, "extra_harm_vs_uniform_merger"].iloc[0])}
    inc = {k: v - pc for k, v in alt.items()}
    inc["perfect"] = float(tb["oracle: interaction"])                  # vs perfect-data competition
    base = ot.set_index("row").dcs_chf_hh_year
    rows = [("B+", "B with pooled-data personal pricing (today's data quality)", "B", "pooled"),
            ("B++", "B with pooled-data personal pricing (perfect data)", "B", "perfect"),
            ("E", f"C + data ring-fence (privacy filter ε = {EPS:g})", "C", "dp"),
            ("E'", "C + Cumulus and Supercard kept apart (no pooling)", "C", "apart"),
            ("F", f"D5 + data ring-fence (privacy filter ε = {EPS:g})", "D5", "dp")]
    new = pd.DataFrame([{"row": r, "scenario": s_, "anchor": a_, "anchor_dcs": base[a_],
                         "data_increment_chf_hh_year": inc[k], "dcs_chf_hh_year": base[a_] + inc[k],
                         "alt_increment_if_personal_pricing_is_new": alt[k],
                         "alt_dcs_if_personal_pricing_is_new": base[a_] + alt[k]} for r, s_, a_, k in rows])
    new.round(2).to_csv(OUT / "3_6_outcome_rows.csv", index=False)
    print(new.round(1).to_string(index=False))
    full = ot[~ot.row.isin(["E", "F"])][["row", "scenario", "dcs_chf_hh_year"]].copy()
    full["data_increment_chf_hh_year"] = 0.0
    full["alt_dcs_if_personal_pricing_is_new"] = full.dcs_chf_hh_year
    full = pd.concat([full, new[["row", "scenario", "dcs_chf_hh_year", "data_increment_chf_hh_year",
                                 "alt_dcs_if_personal_pricing_is_new"]]])
    order = ["A", "B", "B+", "B++", "C", "E", "E'"] + [r for r in full.row if str(r).startswith("D")] + ["F", "G"]
    full = full.set_index("row").loc[order].reset_index()
    full.round(2).to_csv(OUT / "3_6_outcome_table.csv", index=False)
    figures(full, inc, alt)


def figures(full, inc, alt):
    import textwrap
    viz.setup()
    d = full[full.row.isin(["B", "B+", "B++", "C", "E", "E'", "D5", "F", "D6"])].iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    y = np.arange(len(d))
    a1, a2 = d.dcs_chf_hh_year, d.alt_dcs_if_personal_pricing_is_new
    for yi, u, v in zip(y, a1, a2):
        ax.plot([u, v], [yi, yi], color=viz.MUTED, lw=2, zorder=1)
    ax.scatter(a2, y, s=70, color=viz.ORANGE, zorder=3, edgecolor=viz.SURFACE, linewidth=1.5,
               label="personal pricing arrives with the merger")
    ax.scatter(a1, y, s=70, color=viz.BLUE, zorder=3, edgecolor=viz.SURFACE, linewidth=1.5,
               label="personal pricing already exists today (merger-specific change only)")
    for yi, u, v in zip(y, a1, a2):
        if abs(u - v) < 1:
            ax.text(u + 12, yi, f"{u:.0f}", va="center", fontsize=8.5, color=viz.INK_2)
        else:
            ax.text(max(u, v) + 12, yi, f"{max(u, v):.0f}", va="center", fontsize=8.5, color=viz.INK_2)
            ax.text(min(u, v) - 12, yi, f"{min(u, v):.0f}", va="center", ha="right", fontsize=8.5, color=viz.INK_2)
    ax.set_yticks(y, [f"{r}  " + "\n     ".join(textwrap.wrap(s_, 46)) for r, s_ in zip(d.row, d.scenario)],
                  fontsize=8.3)
    ax.axvline(0, color=viz.INK_2, lw=0.8)
    lo = np.floor(min(a1.min(), a2.min()) / 100) * 100 - 50
    ax.set_xlim(lo, 30)
    ax.set_xlabel("Δ consumer surplus vs today, CHF per household per year")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title(f"Personal pricing on pooled data adds CHF {-alt['pooled']:.0f} a year if it comes with the merger, "
                 f"{-inc['pooled']:.0f}-{-inc['perfect']:.0f} if it already exists;\na privacy filter (ε = 1) removes it "
                 f"in both cases [calibrated + sim]", fontsize=11, loc="left")
    viz.source(ax, "Uniform-price rows from Pillar 1 (1_8). Data increments from the 20,000-household panel (3.3, 3.4) "
                   "at row-B prices, added to the other rows [approx]. Blue E/F also count the filter removing "
                   "today's personalisation (+192), which is not merger harm.")
    viz.save(fig, 3, "3_6", "waterfall")

    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cant = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)[["ref", "geometry"]]
    L = pd.read_parquet(P3 / "pricing.parquet")
    hh = pd.read_parquet(EV / "households.parquet").set_index("hh_id")
    uM = L[L.scenario == "U-M"].set_index("hh_id").dcs_year
    pM = L[L.scenario == "P-M"].set_index("hh_id").dcs_year
    can = pd.DataFrame({"u": uM, "p": pM - uM, "canton": hh.canton}).groupby("canton").mean()
    can["share"] = 100 * can.p / can.u
    g = cant.merge(can, left_on="ref", right_index=True)
    bins = np.linspace(np.floor(g.share.min() / 5) * 5, np.ceil(g.share.max() / 5) * 5, 7)
    cmap, norm = ListedColormap(BLUES), BoundaryNorm(bins, len(BLUES))
    fig, ax = plt.subplots(figsize=(8, 5))
    g.plot(ax=ax, column="share", cmap=cmap, norm=norm, edgecolor=viz.SURFACE, linewidth=0.6)
    viz.map_axes(ax, cant)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal", fraction=0.05,
                      pad=0.03, shrink=0.7)
    cb.set_label("data harm as % of the uniform-price merger harm (panel, by home canton)", fontsize=8.5)
    cb.set_ticks(bins)
    cb.set_ticklabels([f"{b:.0f}%" for b in bins])
    nat = 100 * float((pM - uM).mean() / uM.mean())
    print("national data share %.1f%%, canton range %.0f-%.0f%%" % (nat, g.share.min(), g.share.max()))
    ax.set_title(f"If personal pricing arrives with the merger, it adds {nat:.0f}% to the harm nationally "
                 f"({g.share.min():.0f}-{g.share.max():.0f}% by canton) [sim]", fontsize=11)
    viz.source(ax, "Panel households; small cantons noisy. [sim]")
    viz.save(fig, 3, "3_6", "data_share_map")


if __name__ == "__main__":
    main()
