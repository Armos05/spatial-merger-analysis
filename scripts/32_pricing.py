"""Pillar 3 step 3: personalised prices × merger — the data theory of harm, quantified. [sim]

20,000 panel households (3.1). Uniform prices = Pillar 1 equilibrium banner prices (pre-merger; post-merger row B).
Personalised = on top, each card programme sends household i a coupon/surcharge m_fi on its card banners
(engine: src/mxc/datamerger/personal.py). Firms price with what their data tell them (α̂ from 3.1, winsorised at
the 5th/95th percentile: no firm prices on raw outliers); households respond with the truth.
Coupons/surcharges are bounded to m in [0.6, 2.0]; convergence = largest KKT violation (projected gradient).

Scenarios (CS in CHF per household per year = 52 × per-basket change, as in Pillar 1):
    U-C  uniform, competition                (status quo, reference)
    U-M  uniform, merger                     (Pillar 1 row B on this panel)
    P-C  personalised, competition           Nash; Migros uses Cumulus α̂, Coop uses Supercard α̂
    P-M  personalised, merger, pooled data   one firm, both cards
  and, to split P-M:
    P-M ring-fenced   merged firm, but each card team prices with its own card's α̂ only
    P-M location      merged firm knows only where shoppers live (α̂ = location-only)
    P-C / P-M oracle  firms know the true α_i (upper bound of what data can reveal)

Interaction (DESIGN §5): I = [P-M − U-M] − [P-C − U-C]: how much worse personalisation is once the rival is gone.

Run:  conda activate keystone && python scripts/32_pricing.py      (needs 30)
Data (data/processed/p3/): pricing.parquet (per household × scenario)
Outputs (assets/p3/m3/):
  3_3_scenarios.csv        ΔCS, Δprofits, prices, multipliers per scenario
  3_3_two_by_two.csv       the 2 × 2 and the interaction, with decomposition of P-M
  3_3_segments.csv         ΔCS by hidden segment and income type
  3_3_two_by_two.png       the 2 × 2 exhibit
  3_3_who_pays.png         personalised merger: ΔCS by segment and by income
  3_3_harm_map.png         extra harm from personalisation after the merger, by home canton
"""
import runpy
import sys
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
from mxc.datamerger import personal as pe  # noqa: E402

T30 = runpy.run_path(str(ROOT / "scripts/30_experiments.py"), run_name="lib")
P3, INTERIM, EV = ROOT / "data/processed/p3", ROOT / "data/interim", ROOT / "data/processed/twin/events"
OUT = ROOT / "assets/p3/m3"
REDS = ["#fbe3e2", "#f6c7c6", "#ee9190", "#e34948", "#b52f2e", "#7a1f1e"]
SEG_ORDER = T30["SEG_ORDER"]
WEEKS_YEAR = 52


def winsor(x, lo=5, hi=95):
    """A firm does not price on raw noisy estimates: α̂ is winsorised at its 5th / 95th percentile."""
    x = np.asarray(x, float)
    return np.clip(x, *np.percentile(x, [lo, hi]))


def run(S, alpha_M, alpha_C, merged, base, split=False, oracle=False):
    """Personalised multipliers and the true outcome. split: merged firm, each card team uses its own α̂."""
    th, ahB, cost, lm = S["theta"], S["ahB"], S["cost"], S["lm"]
    if not oracle:
        alpha_M, alpha_C = winsor(alpha_M), winsor(alpha_C)
    aM, aC = jnp.asarray(alpha_M), jnp.asarray(alpha_C)
    if split:
        mM, _, g1, _ = pe.solve(th, ahB, base, cost, lm, aM, aM, merged=True)
        _, mC, g2, _ = pe.solve(th, ahB, base, cost, lm, aC, aC, merged=True)
        g = max(g1, g2)
    else:
        mM, mC, g, _ = pe.solve(th, ahB, base, cost, lm, aM, aC, merged=merged)
    ev = pe.evaluate(th, S["ahT"], base, cost, lm, jnp.asarray(S["hh"].alpha_i.to_numpy()), mM, mC)
    ev["mM"], ev["mC"] = mM, mC
    return ev, g


def uniform(S, base):
    H = len(S["hh"])
    ev = pe.evaluate(S["theta"], S["ahT"], base, S["cost"], S["lm"], jnp.asarray(S["hh"].alpha_i.to_numpy()),
                     np.ones(H), np.ones(H))
    ev["mM"], ev["mC"] = 1.0, 1.0
    return ev


def scenarios(S, r):
    al = S["hh"].alpha_i.to_numpy()
    out = {"U-C": uniform(S, S["pre"]), "U-M": uniform(S, S["post"])}
    conv = {}
    specs = {
        "P-C": (r.alpha_hat_M_own, r.alpha_hat_C_own, False, S["pre"], False),
        "P-M": (r.alpha_hat_pooled, r.alpha_hat_pooled, True, S["post"], False),
        "P-M ring-fenced": (r.alpha_hat_M_own, r.alpha_hat_C_own, True, S["post"], True),
        "P-M location": (r.alpha_hat_location, r.alpha_hat_location, True, S["post"], False),
        "P-C oracle": (al, al, False, S["pre"], False),
        "P-M oracle": (al, al, True, S["post"], False),
    }
    for k, (aM, aC, merged, base, split) in specs.items():
        out[k], conv[k] = run(S, np.asarray(aM), np.asarray(aC), merged, base, split, oracle="oracle" in k)
        print(k, "max |grad|", conv[k], flush=True)
    return out, conv


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = T30["setup"]()
    hh = S["hh"]
    r = pd.read_parquet(P3 / "alpha_hat.parquet").reindex(hh.hh_id)
    out, conv = scenarios(S, r)
    ref = out["U-C"]
    holder = (hh.cumulus | hh.supercard).to_numpy()
    rows, long = [], []
    for k, ev in out.items():
        d = ev.sub(ref[["cs", "profit_M", "profit_C", "profit_other", "avg_price_paid"]], fill_value=0)
        rows.append({"scenario": k, "dcs_chf_hh_year": WEEKS_YEAR * d.cs.mean(),
                     "dcs_card_holders": WEEKS_YEAR * d.cs[holder].mean(),
                     "dprofit_MC_chf_hh_year": WEEKS_YEAR * (d.profit_M + d.profit_C).mean(),
                     "dprofit_other_chf_hh_year": WEEKS_YEAR * d.profit_other.mean(),
                     "dtotal_chf_hh_year": WEEKS_YEAR * (d.cs + d.profit_M + d.profit_C + d.profit_other).mean(),
                     "avg_price_paid_pct": 100 * (ev.avg_price_paid.mean() / ref.avg_price_paid.mean() - 1),
                     "mM_p10": np.percentile(ev.mM[hh.cumulus], 10) if k.startswith("P") else 1.0,
                     "mM_p50": np.percentile(ev.mM[hh.cumulus], 50) if k.startswith("P") else 1.0,
                     "mM_p90": np.percentile(ev.mM[hh.cumulus], 90) if k.startswith("P") else 1.0,
                     "corr_mM_alpha": np.corrcoef(ev.mM[hh.cumulus], hh.alpha_i[hh.cumulus])[0, 1]
                     if k.startswith("P") else np.nan,
                     "max_grad": conv.get(k, 0.0)})
        long.append(pd.DataFrame({"hh_id": hh.hh_id, "scenario": k, "dcs_year": WEEKS_YEAR * d.cs,
                                  "mM": ev.mM, "mC": ev.mC}))
    sc = pd.DataFrame(rows)
    sc.round(4).to_csv(OUT / "3_3_scenarios.csv", index=False)
    print(sc.round(2).to_string(index=False))
    L = pd.concat(long)
    L.to_parquet(P3 / "pricing.parquet", index=False)

    v = sc.set_index("scenario").dcs_chf_hh_year
    tb = pd.DataFrame([
        {"item": "U-C (status quo)", "value": v["U-C"]},
        {"item": "U-M (merger, uniform prices)", "value": v["U-M"]},
        {"item": "P-C (personalised, competition, own data)", "value": v["P-C"]},
        {"item": "P-M (personalised, merger, pooled data)", "value": v["P-M"]},
        {"item": "personalisation under competition: P-C − U-C", "value": v["P-C"] - v["U-C"]},
        {"item": "personalisation after the merger: P-M − U-M", "value": v["P-M"] - v["U-M"]},
        {"item": "interaction I = (P-M − U-M) − (P-C − U-C)", "value": (v["P-M"] - v["U-M"]) - (v["P-C"] - v["U-C"])},
        {"item": "  of which: rival gone, same data (ring-fenced − U-M) − (P-C − U-C)",
         "value": (v["P-M ring-fenced"] - v["U-M"]) - (v["P-C"] - v["U-C"])},
        {"item": "  of which: pooling the cards (P-M − ring-fenced)", "value": v["P-M"] - v["P-M ring-fenced"]},
        {"item": "location-only targeting after merger (P-M location − U-M)", "value": v["P-M location"] - v["U-M"]},
        {"item": "oracle: personalisation under competition", "value": v["P-C oracle"] - v["U-C"]},
        {"item": "oracle: personalisation after merger", "value": v["P-M oracle"] - v["U-M"]},
        {"item": "oracle: interaction", "value": (v["P-M oracle"] - v["U-M"]) - (v["P-C oracle"] - v["U-C"])},
    ])
    tb.round(2).to_csv(OUT / "3_3_two_by_two.csv", index=False)
    print(tb.round(1).to_string(index=False))

    seg = []
    for k in ("U-M", "P-C", "P-M", "P-M ring-fenced"):
        d = L[L.scenario == k].set_index("hh_id").dcs_year
        dU = L[L.scenario == "U-M"].set_index("hh_id").dcs_year
        for col, grp in (("segment", hh.set_index("hh_id").segment), ("income_type", hh.set_index("hh_id").income)):
            for g, idx in grp.groupby(grp).groups.items():
                seg.append({"scenario": k, "by": col, "group": g, "households": len(idx), "dcs_year": d.loc[idx].mean(),
                            "extra_vs_uniform_merger": (d.loc[idx] - dU.loc[idx]).mean()})
    seg = pd.DataFrame(seg)
    seg.round(3).to_csv(OUT / "3_3_segments.csv", index=False)
    figures(sc, tb, seg, L, hh)


def figures(sc, tb, seg, L, hh):
    viz.setup()
    v = sc.set_index("scenario").dcs_chf_hh_year
    # 1. 2 × 2
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.25]})
    cells = {(0, 0): ("U-C", "uniform\ncompetition"), (1, 0): ("P-C", "personalised\ncompetition"),
             (0, 1): ("U-M", "uniform\nmerger"), (1, 1): ("P-M", "personalised\nmerger (pooled)")}
    vmin = min(v[["U-M", "P-M"]].min(), -1)
    for (i, j), (k, lab) in cells.items():
        val = v[k]
        shade = 0 if val >= 0 else min(5, int(5 * val / vmin))
        ax.add_patch(plt.Rectangle((j, 1 - i), 0.96, 0.96, color=REDS[shade] if val < 0 else viz.GRID))
        ax.text(j + 0.48, 1 - i + 0.62, f"{val:+.0f}", ha="center", va="center", fontsize=20, fontweight="bold",
                color="white" if shade >= 3 else viz.INK)
        ax.text(j + 0.48, 1 - i + 0.28, lab, ha="center", va="center", fontsize=9,
                color="white" if shade >= 3 else viz.INK_2)
    ax.set_xlim(-0.05, 2.0)
    ax.set_ylim(-0.05, 2.0)
    ax.axis("off")
    ax.set_title("Δ consumer surplus, CHF per household per year", fontsize=10.5)
    t = tb.set_index("item").value
    I = t["interaction I = (P-M − U-M) − (P-C − U-C)"]
    items = [("personalisation,\ncompetition", t["personalisation under competition: P-C − U-C"], viz.ORANGE),
             ("personalisation,\nafter merger", t["personalisation after the merger: P-M − U-M"], viz.ORANGE),
             ("interaction", I, viz.BLUE),
             ("  rival gone", t["  of which: rival gone, same data (ring-fenced − U-M) − (P-C − U-C)"], viz.MUTED),
             ("  cards pooled", t["  of which: pooling the cards (P-M − ring-fenced)"], viz.MUTED),
             ("perfect data,\ncompetition", t["oracle: personalisation under competition"], viz.INK_2),
             ("perfect data,\nafter merger", t["oracle: personalisation after merger"], viz.INK_2),
             ("perfect data,\ninteraction", t["oracle: interaction"], viz.BLUE)]
    y = np.arange(len(items))[::-1]
    for yi, (lab, val, colr) in zip(y, items):
        ax2.barh(yi, val, color=colr, height=0.6)
        ax2.text(val + (2 if val >= 0 else -2), yi, f"{val:+.0f}", va="center", ha="left" if val >= 0 else "right",
                 fontsize=8.5)
    ax2.set_yticks(y, [x[0] for x in items], fontsize=8.5)
    ax2.axvline(0, color=viz.INK_2, lw=0.8)
    ax2.set_xlabel("Δ consumer surplus, CHF per household per year")
    ax2.set_title("What personalisation adds, and why", fontsize=10.5)
    lo = min(x[1] for x in items)
    ax2.set_xlim(lo * 1.35, max(10, max(x[1] for x in items) * 1.4))
    ax2.set_axisbelow(True)
    Io = t["oracle: interaction"]
    fig.suptitle(f"Personal pricing costs shoppers ~CHF {-(v['P-C'] - v['U-C']):.0f} a year with or without the merger; "
                 f"the merger adds {-I:.0f} with today's data, {-Io:.0f} with perfect data [sim]",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    viz.source(ax, "20,000 panel households; uniform prices = Pillar 1 equilibria; personal coupons on card banners "
                   "only, priced with α̂ from 3.1; households respond with the true model. Per basket × 52. [sim]")
    viz.save(fig, 3, "3_3", "two_by_two")

    # 2. who pays
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, by, order in ((axes[0], "segment", SEG_ORDER), (axes[1], "income_type", None)):
        d = seg[(seg.by == by)]
        groups = order or sorted(d.group.unique(), key=float)
        x = np.arange(len(groups))
        for k, (sc_, lab, colr) in enumerate((("P-C", "personalised, competition (vs uniform)", viz.ORANGE),
                                              ("P-M", "personalised, merger (vs uniform merger)", viz.BLUE))):
            dd = d[d.scenario == sc_].set_index("group")
            if sc_ == "P-C":
                vals = [dd.loc[g, "dcs_year"] for g in groups]
            else:
                vals = [dd.loc[g, "extra_vs_uniform_merger"] for g in groups]
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=colr, label=lab)
        ax.axhline(0, color=viz.INK_2, lw=0.8)
        labs = [g.replace("_", "\n") for g in groups] if order else [f"{float(g) / 1000:.0f}k" for g in groups]
        ax.set_xticks(x, labs, fontsize=8.5)
        ax.set_axisbelow(True)
    axes[0].set_title("By hidden shopper segment", fontsize=10.5)
    axes[1].set_title("By household income (CHF / year)", fontsize=10.5)
    axes[0].set_ylabel("Δ CS from personalisation,\nCHF per household per year")
    axes[0].legend(fontsize=8.5, loc="lower left")
    p = seg[(seg.scenario == "P-M") & (seg.by == "segment")].set_index("group").extra_vs_uniform_merger
    fig.suptitle(f"After the merger, loyal 'quality' shoppers pay most for personal pricing "
                 f"(CHF {-p['quality']:.0f}/yr); price hunters lose least [sim]", x=0.02, ha="left", fontsize=12,
                 fontweight="bold")
    viz.source(axes[0], "Personalisation effect = personalised minus uniform under the same market structure. [sim]")
    viz.save(fig, 3, "3_3", "who_pays")

    # 3. map of extra harm after merger
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cant = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)[["ref", "geometry"]]
    h = hh.set_index("hh_id")
    dPM = L[L.scenario == "P-M"].set_index("hh_id").dcs_year - L[L.scenario == "U-M"].set_index("hh_id").dcs_year
    dPC = L[L.scenario == "P-C"].set_index("hh_id").dcs_year
    can = pd.DataFrame({"PM": dPM, "PC": dPC, "canton": h.canton}).groupby("canton").mean()
    g = cant.merge(-can, left_on="ref", right_index=True)
    hi = np.ceil(max(g.PM.max(), g.PC.max()) / 10) * 10
    bins = np.linspace(0, hi, 7)
    cmap, norm = ListedColormap(REDS), BoundaryNorm(bins, len(REDS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, ttl in ((axes[0], "PC", "Today: personal prices, Migros and Coop compete"),
                         (axes[1], "PM", "After the merger: on top of the uniform price rise")):
        g.plot(ax=ax, column=col, cmap=cmap, norm=norm, edgecolor=viz.SURFACE, linewidth=0.6)
        viz.map_axes(ax, cant)
        ax.set_title(ttl, fontsize=10.5)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, orientation="horizontal", fraction=0.05,
                      pad=0.03, shrink=0.5)
    cb.set_label("loss from personalised prices, CHF per household per year (by home canton)", fontsize=8.5)
    fig.suptitle("Personal pricing costs households in every canton, today and after the merger [sim]", x=0.02, ha="left", fontsize=12, fontweight="bold")
    viz.source(axes[0], "Panel households by home canton; small cantons have few panel households. [sim]")
    viz.save(fig, 3, "3_3", "harm_map")


if __name__ == "__main__":
    if "--figures" in sys.argv:
        hh = pd.read_parquet(EV / "households.parquet")
        figures(pd.read_csv(OUT / "3_3_scenarios.csv"), pd.read_csv(OUT / "3_3_two_by_two.csv"),
                pd.read_csv(OUT / "3_3_segments.csv").astype({"group": str}), pd.read_parquet(P3 / "pricing.parquet"), hh)
    else:
        main()
