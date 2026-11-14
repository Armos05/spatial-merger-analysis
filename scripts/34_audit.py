"""Pillar 3 step 5: an audit WEKO could run — paired synthetic shopper profiles and a power calculation. [sim]

Design (correspondence-test logic, as in labour-market audit studies):
  - At each audited address, open two loyalty accounts that are identical except for their purchase history:
      profile L "looks loyal": the card history of a typical quality shopper (median features of that group)
      profile S "looks price-sensitive": the card history of a typical price hunter
  - Record the personal offers each account receives on the same basket. Offer gap g = m_L − m_S (in % of price).
  - Run one wave before and one after the merger (or compare with a control retailer). The harm signal is the
    rise in the gap: Δ = mean g_post − mean g_pre.

The twin plays the firm. Its pricing algorithm = the α̂ it learned in 3.1 (a LightGBM surrogate of α̂ on the
features) + the optimal coupons of 3.3: Nash with own-card data before, merged with pooled data after.

Power (two-sample, two-sided 5%, 80% power), n audit pairs per wave:
    n = 2 × (z_0.975 + z_0.80)² × σ² / Δ²,      σ² = var(g across addresses) + 2 × σ_offer²
σ_offer = 2% offer noise per account [assumed] (firms' own randomisation, rounding of coupons).

Run:  conda activate keystone && python scripts/34_audit.py      (needs 30, 32)
Outputs (assets/p3/m5/):
  3_5_audit_offers.csv     per audited address: offers to L and S, before and after
  3_5_power.csv            pairs needed per wave by effect size; power by n
  3_5_power.png            power curves
  3_5_audit_map.png        where the gap rises most (audit targeting), by canton
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
from scipy.stats import norm as N  # noqa: E402

from mxc import viz  # noqa: E402
from mxc.datamerger import personal as pe  # noqa: E402

T30 = runpy.run_path(str(ROOT / "scripts/30_experiments.py"), run_name="lib")
P3, INTERIM, EV, PROCESSED = (ROOT / "data/processed/p3", ROOT / "data/interim", ROOT / "data/processed/twin/events",
                              ROOT / "data/processed")
OUT = ROOT / "assets/p3/m5"
BLUES = T30["BLUES"]
N_ADDR, SIGMA_OFFER = 2000, 0.02


def surrogate(X, y):
    import lightgbm as lgb
    m = ~np.isnan(y)
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                             verbose=-1, random_state=0).fit(X[m], y[m])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = T30["setup"]()
    hh = S["hh"]
    r = pd.read_parquet(P3 / "alpha_hat.parquet").reindex(hh.hh_id)
    X = pd.read_parquet(P3 / "features.parquet").reindex(hh.hh_id)
    loc_cols = [c for c in X.columns if not c.startswith(("M_", "C_"))]
    m_cols = [c for c in X.columns if c.startswith("M_") and c != "M_share_of_card_trips"]
    c_cols = [c for c in X.columns if c.startswith("C_")]
    both = (hh.cumulus & hh.supercard).to_numpy()
    f_pool = surrogate(X.to_numpy(float), r.alpha_hat_pooled.where(both).to_numpy())
    f_M = surrogate(X[loc_cols + m_cols].to_numpy(float), r.alpha_own_card_M.to_numpy())
    f_C = surrogate(X[loc_cols + c_cols].to_numpy(float), r.alpha_own_card_C.to_numpy())

    # two profiles: median card history of quality shoppers vs price hunters (both cards)
    prof = {}
    for name, seg in (("L", "quality"), ("S", "price_hunter")):
        m = both & (hh.segment == seg).to_numpy()
        prof[name] = X.loc[m, m_cols + c_cols + ["M_share_of_card_trips"]].median()
    rng = np.random.default_rng(T30["SEED"] + 34)
    a = S["a"]
    w = np.asarray(a["hh"])
    locs = rng.choice(len(w), N_ADDR, p=w / w.sum(), replace=False)
    lmeta = pd.read_parquet(PROCESSED / "locations.parquet", columns=["loc_id", "canton"])
    mkt = S["mkt"]
    syn = pd.DataFrame({"hh_id": np.arange(2 * N_ADDR), "loc": np.r_[locs, locs],
                        "loc_id": np.r_[mkt.loc_id[locs], mkt.loc_id[locs]], "profile": ["L"] * N_ADDR + ["S"] * N_ADDR,
                        "segment": "mainstream", "gamma_mult": 1.0, "kappa_mult": 1.0, "cumulus": True,
                        "supercard": True, "border_min": np.asarray(a["border_min"])[np.r_[locs, locs]],
                        "alpha_i": 0.03})
    XL = T30["location_features"]({**S, "hh": syn}).reset_index(drop=True)
    XP = pd.DataFrame([prof[p] for p in syn.profile]).reset_index(drop=True)
    Xs = pd.concat([XL, XP], axis=1)[X.columns]
    # same rule as 3.3: the firm winsorises α̂ at the 5th / 95th percentile of its customers' estimates
    def clip(x, ref):
        return np.clip(x, *np.nanpercentile(ref, [5, 95]))
    a_pool = clip(f_pool.predict(Xs.to_numpy(float)), r.alpha_hat_pooled.where(both))
    a_M = clip(f_M.predict(Xs[loc_cols + m_cols].to_numpy(float)), r.alpha_own_card_M)
    a_C = clip(f_C.predict(Xs[loc_cols + c_cols].to_numpy(float)), r.alpha_own_card_C)
    ah = pe.panel_arrays(a, syn, S["xi"], informed=False)
    th, cost, lm = S["theta"], S["cost"], S["lm"]
    mM0, mC0, g0, _ = pe.solve(th, ah, S["pre"], cost, lm, jnp.asarray(a_M), jnp.asarray(a_C), merged=False)
    mM1, mC1, g1, _ = pe.solve(th, ah, S["post"], cost, lm, jnp.asarray(a_pool), jnp.asarray(a_pool), merged=True)
    print("convergence", g0, g1)
    # offers in CHF relative to the uniform banner price; average of the Migros and Coop coupons
    pre_b, post_b = np.asarray(S["pre"]), np.asarray(S["post"])
    iM, iC = T30["dm"].BANNERS.index("Migros"), T30["dm"].BANNERS.index("Coop")
    off = pd.DataFrame({"addr": np.r_[np.arange(N_ADDR), np.arange(N_ADDR)], "profile": syn.profile,
                        "loc_id": syn.loc_id, "alpha_hat_pre_M": a_M, "alpha_hat_post": a_pool,
                        "price_pre": 0.5 * (mM0 * pre_b[iM] + mC0 * pre_b[iC]),
                        "price_post": 0.5 * (mM1 * post_b[iM] + mC1 * post_b[iC]),
                        "uniform_pre": 0.5 * (pre_b[iM] + pre_b[iC]), "uniform_post": 0.5 * (post_b[iM] + post_b[iC])})
    W = off.pivot(index="addr", columns="profile")
    res = pd.DataFrame({"loc_id": W[("loc_id", "L")],
                        "gap_pre_pct": 100 * (W[("price_pre", "L")] - W[("price_pre", "S")]) / W[("uniform_pre", "L")],
                        "gap_post_pct": 100 * (W[("price_post", "L")] - W[("price_post", "S")]) / W[("uniform_pre", "L")],
                        "L_pre_vs_uniform_pct": 100 * (W[("price_pre", "L")] / W[("uniform_pre", "L")] - 1),
                        "S_pre_vs_uniform_pct": 100 * (W[("price_pre", "S")] / W[("uniform_pre", "L")] - 1),
                        "L_post_vs_uniform_pre_pct": 100 * (W[("price_post", "L")] / W[("uniform_pre", "L")] - 1),
                        "S_post_vs_uniform_pre_pct": 100 * (W[("price_post", "S")] / W[("uniform_pre", "L")] - 1)})
    res = res.merge(lmeta, on="loc_id", how="left")
    res["change_pp"] = res.gap_post_pct - res.gap_pre_pct
    res.round(4).to_csv(OUT / "3_5_audit_offers.csv", index=False)
    print(res.describe().round(2).to_string())

    # power
    z = N.ppf(0.975) + N.ppf(0.8)
    s_addr = res.gap_post_pct.std()
    s_addr0 = res.gap_pre_pct.std()
    sig2 = 0.5 * (s_addr ** 2 + s_addr0 ** 2) + 2 * (100 * SIGMA_OFFER) ** 2
    delta_twin = res.change_pp.mean()
    rows = []
    for d in (0.5, 1.0, 2.0, 3.0, round(delta_twin, 2)):
        rows.append({"effect_pp": d, "sigma_pp": np.sqrt(sig2), "pairs_per_wave": int(np.ceil(2 * z ** 2 * sig2 / d ** 2))})
    pw = pd.DataFrame(rows)
    # paired design (same addresses both waves): variance of the change only
    sig2_p = res.change_pp.var() + 4 * (100 * SIGMA_OFFER) ** 2
    pw["pairs_if_same_addresses"] = [int(np.ceil(z ** 2 * sig2_p / d ** 2)) for d in pw.effect_pp]
    pw.round(3).to_csv(OUT / "3_5_power.csv", index=False)
    print(pw.to_string(index=False))
    figures(res, pw, sig2, sig2_p, delta_twin)


def figures(res, pw, sig2, sig2_p, delta_twin):
    viz.setup()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1.2]})
    for col, lab, colr in (("gap_pre_pct", "before: Migros and Coop compete, own data", viz.ORANGE),
                           ("gap_post_pct", "after: merged firm, pooled data", viz.BLUE)):
        ax.hist(res[col], bins=40, color=colr, alpha=0.75, label=lab)
    ax.set_xlabel("offer gap: 'loyal' minus 'price-sensitive' profile, % of uniform price")
    ax.set_ylabel("audited addresses")
    ax.legend(fontsize=8.5)
    ax.set_title(f"Offer gap per address: {res.gap_pre_pct.mean():.1f}% → {res.gap_post_pct.mean():.1f}%", fontsize=10.5)
    ax.set_axisbelow(True)
    n = np.arange(5, 2001)
    z975 = N.ppf(0.975)
    for d, colr in ((0.5, viz.MUTED), (1.0, viz.INK_2), (2.0, viz.ORANGE), (delta_twin, viz.BLUE)):
        pw_ = N.cdf(d / np.sqrt(2 * sig2 / n) - z975)
        ax2.plot(n, 100 * pw_, color=colr, label=f"rise in gap {d:.1f} pp" + (" (twin)" if d == delta_twin else ""))
    ax2.axhline(80, color=viz.INK_2, lw=0.8, ls="--")
    ax2.set_xscale("log")
    ax2.set_xlabel("audit pairs per wave (log scale)")
    ax2.set_ylabel("power (%)")
    ax2.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.45))
    ax2.set_title("Power of a two-wave audit (5% test)", fontsize=10.5)
    nt = pw.loc[pw.effect_pp == round(delta_twin, 2), "pairs_per_wave"].iloc[0]
    n1 = pw.loc[pw.effect_pp == 1.0, "pairs_per_wave"].iloc[0]
    fig.suptitle(f"WEKO could detect the twin's data harm with {nt} pairs of test accounts per wave; "
                 f"a 1 pp rise needs {n1} [sim]", x=0.02, ha="left", fontsize=12, fontweight="bold")
    viz.source(ax, f"{len(res)} random addresses (∝ households); profiles = median card history of quality shoppers "
                   f"vs price hunters; offer noise {100 * SIGMA_OFFER:.0f}% per account [assumed]. [sim]")
    viz.save(fig, 3, "3_5", "power")

    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cant = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)[["ref", "geometry"]]
    can = res.groupby("canton").agg(change=("change_pp", "mean"), n=("change_pp", "size"))
    g = cant.merge(can, left_on="ref", right_index=True)
    lo, hi = np.floor(g.change.min() * 2) / 2, np.ceil(g.change.max() * 2) / 2
    bins = np.linspace(lo, hi, 7)
    cmap, norm = ListedColormap(BLUES), BoundaryNorm(bins, len(BLUES))
    fig, ax = plt.subplots(figsize=(8, 5))
    g.plot(ax=ax, column="change", cmap=cmap, norm=norm, edgecolor=viz.SURFACE, linewidth=0.6)
    viz.map_axes(ax, cant)
    for _, rr in g.nlargest(5, "change").iterrows():
        c = rr.geometry.representative_point()
        ax.text(c.x, c.y, rr.ref, ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal", fraction=0.05,
                      pad=0.03, shrink=0.7)
    cb.set_label("rise in offer gap after the merger, pp (mean over audited addresses)", fontsize=8.5)
    cb.set_ticks(bins)
    cb.set_ticklabels([f"{b:.1f}" for b in bins])
    ax.set_title("Where to audit: the gap rises most where Migros and Coop were close rivals [sim]", fontsize=11)
    viz.source(ax, "Labelled: five cantons with the largest rise. [sim]")
    viz.save(fig, 3, "3_5", "audit_map")


if __name__ == "__main__":
    main()
