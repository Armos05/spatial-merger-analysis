"""Pillar 3 step 4: a differential-privacy ring-fence — how much targeting survives, how much harm remains. [sim]

Remedy: the merged firm may combine Cumulus and Supercard histories only through a privacy filter. Each
household's card features are released with local differential privacy (Laplace mechanism):
    x̃_k = clip(x_k, lo_k, hi_k) + Laplace(0, b_k),     b_k = d × (hi_k − lo_k) / ε
d = number of card features (budget ε split evenly: sequential composition), [lo, hi] = 5th-95th percentile.
The home address stays usable (firms know it anyway); the experiment outcomes are the firm's own.
Smaller ε = more privacy = noisier features. ε → 0: location-only targeting; ε → ∞: full pooling.

For each ε: T-learner on noisy pooled features → α̂_ε → merged firm's personal prices → true ΔCS.
Curve: extra harm from personalisation after the merger (vs uniform merger) and the firm's gain from targeting,
against ε. References: pooled without filter, ring-fenced (each card separately), location only (3.3).

Run:  conda activate keystone && python scripts/33_ring_fence.py      (needs 30, 32)
Outputs (assets/p3/m4/):
  3_4_dp_curve.csv         per ε: α̂ correlation, ΔCS, Δprofit (merged firm), share of targeting value kept
  3_4_dp_curve.png         privacy-vs-competition curve
  3_4_dp_map.png           by home canton: harm with full pooling vs with the ring-fence at ε = 1
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
T32 = runpy.run_path(str(ROOT / "scripts/32_pricing.py"), run_name="lib")
P3, INTERIM, EV = ROOT / "data/processed/p3", ROOT / "data/interim", ROOT / "data/processed/twin/events"
OUT = ROOT / "assets/p3/m4"
REDS = T32["REDS"]
EPS = [0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
EPS_MAP = 1.0


def privatise(X, cols, eps, rng):
    Z = X.copy()
    d = len(cols)
    for c in cols:
        x = Z[c].to_numpy(float).copy()
        ok = ~np.isnan(x)
        lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
        b = d * (hi - lo) / eps
        x[ok] = np.clip(x[ok], lo, hi) + rng.laplace(0, b, ok.sum())
        Z[c] = x
    return Z


def alpha_pooled(S, X, r, cols):
    out = {}
    for side in ("M", "C"):
        e = r[r[f"alpha_pooled_{side}"].notna()]
        tau, _ = pe.t_learner(X.loc[e.index, cols].to_numpy(float), e[f"y_{side}"].to_numpy(),
                              e[f"treated_{side}"].to_numpy().astype(int), seed=T30["SEED"])
        fill = pd.Series(np.nanmedian(tau), index=r.index)
        fill.loc[e.index] = tau
        a = pe.invert_alpha(S["theta"], S["ahB"], S["pre"], jnp.asarray(fill.to_numpy()), side, T30["DISCOUNT"])
        out[side] = pd.Series(np.where(r[f"alpha_pooled_{side}"].notna(), a, np.nan), index=r.index)
    mean_a = float(np.nanmean(r.alpha_location_only_M))
    return pd.concat(out, axis=1).mean(1).fillna(mean_a)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = T30["setup"]()
    hh = S["hh"]
    r = pd.read_parquet(P3 / "alpha_hat.parquet").reindex(hh.hh_id)
    X = pd.read_parquet(P3 / "features.parquet").reindex(hh.hh_id)
    loc_cols = [c for c in X.columns if not c.startswith(("M_", "C_"))]
    card_cols = [c for c in X.columns if c.startswith(("M_", "C_"))]
    sc = pd.read_csv(ROOT / "assets/p3/m3/3_3_scenarios.csv").set_index("scenario")
    base_cs, base_pi = sc.loc["U-M", "dcs_chf_hh_year"], sc.loc["U-M", "dprofit_MC_chf_hh_year"]
    holders = (hh.cumulus | hh.supercard).to_numpy()
    rng = np.random.default_rng(T30["SEED"] + 33)
    rows, keep = [], {}
    for eps in EPS:
        Xe = privatise(X, card_cols, eps, rng)
        a = alpha_pooled(S, Xe, r, loc_cols + card_cols)
        ev, g = T32["run"](S, a.to_numpy(), a.to_numpy(), True, S["post"])
        ref = T32["uniform"](S, S["pre"])
        d = ev[["cs", "profit_M", "profit_C"]] - ref[["cs", "profit_M", "profit_C"]]
        row = {"epsilon": eps, "corr_alpha": np.corrcoef(a[holders], hh.alpha_i[holders])[0, 1],
               "dcs_chf_hh_year": 52 * d.cs.mean(), "dprofit_MC_chf_hh_year": 52 * (d.profit_M + d.profit_C).mean(),
               "max_grad": g}
        row["extra_harm_vs_uniform_merger"] = row["dcs_chf_hh_year"] - base_cs
        row["targeting_gain_firm"] = row["dprofit_MC_chf_hh_year"] - base_pi
        rows.append(row)
        if eps == EPS_MAP:
            keep = 52 * d.cs
        print(row, flush=True)
    dp = pd.DataFrame(rows)
    refs = {k: (sc.loc[k, "dcs_chf_hh_year"] - base_cs, sc.loc[k, "dprofit_MC_chf_hh_year"] - base_pi)
            for k in ("P-M", "P-M ring-fenced", "P-M location")}
    full_gain, loc_gain = refs["P-M"][1], refs["P-M location"][1]
    dp["share_targeting_value_kept"] = (dp.targeting_gain_firm - loc_gain) / (full_gain - loc_gain)
    dp["share_extra_harm_kept"] = (dp.extra_harm_vs_uniform_merger - refs["P-M location"][0]) / \
        (refs["P-M"][0] - refs["P-M location"][0])
    ref_rows = pd.DataFrame([{"epsilon": np.nan, "reference": k, "extra_harm_vs_uniform_merger": v[0],
                              "targeting_gain_firm": v[1]} for k, v in refs.items()])
    pd.concat([dp.assign(reference="DP ring-fence"), ref_rows]).round(4).to_csv(OUT / "3_4_dp_curve.csv", index=False)
    print(dp.round(3).to_string(index=False))
    L = pd.read_parquet(P3 / "pricing.parquet")
    full = L[L.scenario == "P-M"].set_index("hh_id").dcs_year
    figures(dp, refs, full, pd.Series(keep.to_numpy(), index=hh.hh_id), hh)


def figures(dp, refs, full, dpe, hh):
    viz.setup()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax.plot(dp.epsilon, -dp.extra_harm_vs_uniform_merger, marker="o", color=viz.ORANGE, label="DP ring-fence")
    for k, (lab, ls) in {"P-M": ("pooled, no filter", "--"), "P-M ring-fenced": ("cards kept apart", ":"),
                         "P-M location": ("address only", "-.")}.items():
        ax.axhline(-refs[k][0], color=viz.INK_2, ls=ls, lw=1)
        ax.text(dp.epsilon.min(), -refs[k][0], lab, fontsize=8, color=viz.INK_2, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("privacy budget ε (smaller = more private)")
    ax.set_ylabel("extra loss from personal prices after\nthe merger, CHF per household per year")
    ax.set_title("Shoppers' extra loss", fontsize=10.5)
    ax2.plot(dp.epsilon, 100 * dp.share_targeting_value_kept, marker="o", color=viz.BLUE)
    ax2.set_xscale("log")
    ax2.set_ylim(-5, 110)
    ax2.set_xlabel("privacy budget ε")
    ax2.set_ylabel("targeting value kept by the firm (%)\n(0 = address only, 100 = full pooling)")
    ax2.set_title("Firm's value from pooled data", fontsize=10.5)
    for a_ in (ax, ax2):
        a_.axvline(EPS_MAP, color=viz.MUTED, lw=1)
    k1 = dp.set_index("epsilon").loc[EPS_MAP]
    fig.suptitle(f"At ε = {EPS_MAP:g} the filter cuts the extra loss from CHF {-refs['P-M'][0]:.0f} to "
                 f"{-k1.extra_harm_vs_uniform_merger:.0f} (address-only level); targeting returns between ε 30 and 100 [sim]",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    viz.source(ax, "Local DP: card features clipped to 5th-95th percentile, Laplace noise b = d(hi − lo)/ε; T-learner "
                   "and pricing as in 3.1/3.3; extra loss and gain measured against the uniform-price merger. [sim]")
    viz.save(fig, 3, "3_4", "dp_curve")

    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cant = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)[["ref", "geometry"]]
    L = pd.read_parquet(P3 / "pricing.parquet")
    uM = L[L.scenario == "U-M"].set_index("hh_id").dcs_year
    h = hh.set_index("hh_id")
    can = pd.DataFrame({"full": -(full - uM), "dp": -(dpe - uM), "canton": h.canton}).groupby("canton").mean()
    g = cant.merge(can, left_on="ref", right_index=True)
    hi = np.ceil(max(g.full.max(), g.dp.max()) / 100) * 100
    lo = min(0, np.floor(min(g.full.min(), g.dp.min()) / 100) * 100)
    bins = np.linspace(lo, hi, 7)
    print("cantons where the ring-fence lowers the extra loss:", int((g.dp < g.full).sum()), "of", len(g))
    cmap, norm = ListedColormap(REDS), BoundaryNorm(bins, len(REDS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, ttl in ((axes[0], "full", "Pooled data, no filter"), (axes[1], "dp", f"DP ring-fence, ε = {EPS_MAP:g}")):
        g.plot(ax=ax, column=col, cmap=cmap, norm=norm, edgecolor=viz.SURFACE, linewidth=0.6)
        viz.map_axes(ax, cant)
        ax.set_title(ttl, fontsize=10.5)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, orientation="horizontal", fraction=0.05,
                      pad=0.03, shrink=0.5)
    cb.set_ticks(bins)
    cb.set_ticklabels([f"{b:.0f}" for b in bins])
    cb.set_label("extra loss from personal prices after the merger, CHF per household per year "
                 "(negative = households gain)", fontsize=8.5)
    n_lower = int((g.dp < g.full).sum())
    where = "every canton" if n_lower == len(g) else f"{n_lower} of {len(g)} cantons"
    fig.suptitle(f"A privacy filter at ε = {EPS_MAP:g} lowers the data harm in {where} [sim]", x=0.02, ha="left",
                 fontsize=12, fontweight="bold")
    viz.source(axes[0], "Panel households by home canton. [sim]")
    viz.save(fig, 3, "3_4", "dp_map")


def redraw():
    """Figures only: replay the noise draws up to EPS_MAP (same seed) and re-solve that ε alone."""
    S = T30["setup"]()
    hh = S["hh"]
    r = pd.read_parquet(P3 / "alpha_hat.parquet").reindex(hh.hh_id)
    X = pd.read_parquet(P3 / "features.parquet").reindex(hh.hh_id)
    loc_cols = [c for c in X.columns if not c.startswith(("M_", "C_"))]
    card_cols = [c for c in X.columns if c.startswith(("M_", "C_"))]
    rng = np.random.default_rng(T30["SEED"] + 33)
    for eps in EPS:
        Xe = privatise(X, card_cols, eps, rng)
        if eps == EPS_MAP:
            break
    a = alpha_pooled(S, Xe, r, loc_cols + card_cols)
    ev, _ = T32["run"](S, a.to_numpy(), a.to_numpy(), True, S["post"])
    ref = T32["uniform"](S, S["pre"])
    keep = 52 * (ev.cs - ref.cs)
    print("check: mean extra loss at EPS_MAP", float(keep.mean()))
    t = pd.read_csv(OUT / "3_4_dp_curve.csv")
    dp = t[t.reference == "DP ring-fence"].reset_index(drop=True)
    refs = {k: (float(v.extra_harm_vs_uniform_merger), float(v.targeting_gain_firm))
            for k, v in t[t.reference != "DP ring-fence"].set_index("reference").iterrows()}
    L = pd.read_parquet(P3 / "pricing.parquet")
    full = L[L.scenario == "P-M"].set_index("hh_id").dcs_year
    figures(dp, refs, full, pd.Series(keep.to_numpy(), index=hh.hh_id), hh)


if __name__ == "__main__":
    redraw() if "--figures" in sys.argv else main()
