"""Pillar 2 step 7: border evidence — does the Pillar 1 model get cross-border shopping right? Real card data.

The twin (2.1-2.6) shows the methods work when the model is right. Here we test the model itself out of sample,
with three natural experiments the model was not fitted to:

  A. COVID border closure, 16 Mar - 15 Jun 2020 [data]. Swiss shoppers could not cross. The model says which
     regions should gain most domestic grocery sales: close the abroad option (p_abroad -> infinity) and recompute
     store revenue. Prediction by merchant Grossregion r:  pred_r = ln(R_r closed / R_r open).
     Data: Swiss debit-card POS spending at food retailers by Grossregion (MCS, SIX), weekly, 2020 vs same ISO
     week of 2019:  g_rw = ln V_r,2020,w - ln V_r,2019,w.  Week effects remove the national lockdown shock:
        g_rw - mean_r g_rw = beta_w × (pred_r - mean pred) + e_rw.
     If the model's regional pattern and size are right, beta_w ≈ k during the closure (k = share of abroad
     spending that actually stopped, measured in the same data) and ≈ 0 before.
     7 regions only, so inference is by randomisation: all 7! = 5,040 permutations of pred across regions.
  B. Duty-free allowance cut CHF 300 -> 150 on 1 Jan 2025 [data]. Worldline-acquired POS by merchant canton
     (all categories, domestic + foreign cards). Monthly slope of year-on-year canton growth on the model's
     canton exposure; a lasting cut in food shopping abroad by a fraction f would raise the 2025 slope by
     f × φ, φ = food share of card POS spending.
  C. Franc shock, Jan 2015 (EUR/CHF 1.20 -> ~1.05) [calibrated]: no card data before 2019, so we report the
     model's prediction (abroad basket price × 0.875) by border distance and compare with the published
     cost-of-living gradient (Burstein, Lein & Vogel, JIE 2024).

Run:  conda activate keystone && python scripts/26_border.py
Outputs (assets/p2/m7/):
  2_7_region_covid.csv      model prediction vs observed closure effect, 7 Grossregionen
  2_7_covid_weekly.csv      beta_w per week 2020-2022, abroad share per week
  2_7_canton_exposure.csv   model abroad share of residents and predicted closure gain by canton
  2_7_dutyfree_monthly.csv  monthly slope 2021-2026
  2_7_franc_bands.csv       franc shock prediction by drive time to the border
  2_7_summary.csv           headline numbers
  2_7_exposure_map.png      model prediction by canton vs observed effect by Grossregion
  2_7_covid_event.png       weekly slope, closure shaded
  2_7_covid_scatter.png     7 regions: predicted vs observed
  2_7_dutyfree.png          monthly slope around 1 Jan 2025
  2_7_franc.png             franc shock: abroad share and gain by border distance
"""
import itertools
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
from mxc.structural import demand as dm  # noqa: E402

PROCESSED, INTERIM, MCS = ROOT / "data/processed", ROOT / "data/interim", ROOT / "data/raw/mcs"
OUT = ROOT / "assets/p2/m7"
T20 = runpy.run_path(str(ROOT / "scripts/20_twin.py"), run_name="lib")
BLUES = ["#d6e6f8", "#a9cbf0", "#6fa6e6", "#2a78d6", "#1d58a3", "#123a6e"]
REGION = {1: "Lake Geneva", 2: "Espace Mittelland", 3: "Northwestern", 4: "Zurich", 5: "Eastern", 6: "Central",
          7: "Ticino"}
CANTON_REGION = {**{c: 1 for c in ("GE", "VD", "VS")}, **{c: 2 for c in ("BE", "FR", "JU", "NE", "SO")},
                 **{c: 3 for c in ("AG", "BL", "BS")}, "ZH": 4,
                 **{c: 5 for c in ("AI", "AR", "GL", "GR", "SG", "SH", "TG")},
                 **{c: 6 for c in ("LU", "NW", "OW", "SZ", "UR", "ZG")}, "TI": 7}
CLOSE = (pd.Timestamp("2020-03-16"), pd.Timestamp("2020-06-15"))
PRE = (pd.Timestamp("2020-01-06"), pd.Timestamp("2020-03-01"))
FRANC = 1.05 / 1.20                                      # EUR/CHF 1.20 -> 1.05: abroad basket 12.5% cheaper in CHF


# ----------------------------------------------------------------------------- model side
def model_side():
    mkt, theta = T20["load"]()
    a = dm.to_arrays(mkt)
    stores = pd.read_parquet(PROCESSED / "stores.parquet", columns=["store_id", "canton"]).set_index("store_id")
    s_canton = stores.canton.reindex(mkt.store_id).to_numpy()
    p_store = np.asarray(theta.p)[mkt.store_banner]

    def run(th):
        s_pair, s_ab, logsum = dm.choice_probs(th, a)
        n_lr = np.asarray(a["hh"])[:, None] * np.asarray(a["type_w"])
        q_pair = (n_lr[mkt.pair_loc] * np.asarray(s_pair)).sum(1)
        rev = np.bincount(mkt.pair_store, q_pair * p_store[mkt.pair_store], mkt.J)
        dom_loc = np.bincount(mkt.pair_loc, q_pair * p_store[mkt.pair_store], mkt.L)
        ab_loc = (n_lr * np.asarray(s_ab)).sum(1) * th.p_abroad
        cs = (n_lr * np.asarray(logsum) / np.asarray(dm.alpha_types(th, a["type_y"]))[None, :]).sum(1)
        return rev, dom_loc, ab_loc, cs

    rev0, dom0, ab0, cs0 = run(theta)
    rev1, _, _, _ = run(theta._replace(p_abroad=1e6))                     # border closed
    revF, domF, abF, csF = run(theta._replace(p_abroad=theta.p_abroad * FRANC))   # franc shock (abroad cheaper)
    # pre-2015 baseline: abroad was 1/0.875 dearer; the shock takes it to today's price.
    rev15, dom15, ab15, cs15 = run(theta._replace(p_abroad=theta.p_abroad / FRANC))

    canton = pd.DataFrame({"canton": s_canton, "rev0": rev0, "rev1": rev1}).groupby("canton").sum()
    loc = pd.DataFrame({"canton": mkt.canton, "hh": mkt.hh, "dom": dom0, "ab": ab0, "border_min": mkt.border_min})
    res = loc.groupby("canton").apply(lambda g: pd.Series({"hh": g.hh.sum(),
                                                           "abroad_share_residents": g.ab.sum() / (g.ab + g.dom).sum()}),
                                      include_groups=False)
    canton = canton.join(res)
    canton["pred_closure_pct"] = 100 * np.log(canton.rev1 / canton.rev0)
    canton["region"] = canton.index.map(CANTON_REGION)
    region = canton.groupby("region")[["rev0", "rev1"]].sum()
    region["pred"] = np.log(region.rev1 / region.rev0)
    region["name"] = region.index.map(REGION)
    nat = {"abroad_share_model": ab0.sum() / (ab0.sum() + dom0.sum()),
           "pred_national_pct": 100 * np.log(rev1.sum() / rev0.sum())}

    # franc shock by border distance band (pre-2015 price -> today's price)
    bands = [0, 10, 20, 30, 45, 60, 1e9]
    lab = ["0-10", "10-20", "20-30", "30-45", "45-60", "60+"]
    f = pd.DataFrame({"band": pd.cut(mkt.border_min, bands, labels=lab, right=False), "hh": mkt.hh,
                      "dom0": dom15, "ab0": ab15, "dom1": dom0, "ab1": ab0, "cs0": cs15, "cs1": cs0})
    fb = f.groupby("band", observed=True).sum()
    fb["abroad_share_before"] = fb.ab0 / (fb.ab0 + fb.dom0)
    fb["abroad_share_after"] = fb.ab1 / (fb.ab1 + fb.dom1)
    fb["gain_pct_of_spend"] = 100 * (fb.cs1 - fb.cs0) / (fb.dom0 + fb.ab0)   # money-metric gain / grocery spend
    fb["hh_share"] = fb.hh / fb.hh.sum()
    fb = fb[["hh_share", "abroad_share_before", "abroad_share_after", "gain_pct_of_spend"]]
    nat["franc_abroad_before"] = ab15.sum() / (ab15.sum() + dom15.sum())
    nat["franc_gain_pct"] = 100 * (cs0.sum() - cs15.sum()) / (dom15.sum() + ab15.sum())
    nat["franc_domestic_rev_change_pct"] = 100 * np.log(rev0.sum() / rev15.sum())
    return canton, region, nat, fb


# ----------------------------------------------------------------------------- A. COVID
def weekly_food():
    d = pd.read_csv(MCS / "DEBIT_Merchanttype_Grossregion.csv", parse_dates=["Date"])
    d = d[(d.Type == "POS") & (d["Merchant category"] == "Retail: Food, beverage, tobacco")]
    nb = ["DEU", "FRA", "ITA", "AUT"]
    d = d[(d.Region != 99) | d.Country.isin(nb)]           # Swiss regions + the four neighbours (as in Pillar 1)
    d["wk"] = d.Date - pd.to_timedelta(d.Date.dt.weekday, unit="D")
    w = d.groupby(["wk", "Region"]).agg(v=("Scaled Value", "sum"), days=("Date", "nunique")).reset_index()
    v = w.pivot(index="wk", columns="Region", values="v")
    days = w.pivot(index="wk", columns="Region", values="days")
    return v[(days == 7).all(1)]                          # full weeks only


def covid(region, v, model_share):
    dom = v[[1, 2, 3, 4, 5, 6, 7]]
    iso = dom.index.isocalendar()
    key = pd.MultiIndex.from_arrays([iso.year.to_numpy().astype(int), iso.week.to_numpy().astype(int)])
    lv = np.log(dom.set_axis(key))
    base = lv.loc[2019]
    pred = region.pred.reindex([1, 2, 3, 4, 5, 6, 7]).to_numpy()
    pc = pred - pred.mean()
    rows = []
    for (y, wk), r in lv.iterrows():
        if y == 2019 or wk not in base.index:
            continue
        g = (r - base.loc[wk]).to_numpy()
        rows.append({"year": y, "isoweek": wk, "beta": ((g - g.mean()) @ pc) / (pc @ pc), "national_yoy": g.mean(),
                     **{f"g_{i}": g[i - 1] for i in range(1, 8)}})
    wb = pd.DataFrame(rows)
    wb["week_start"] = [pd.Timestamp.fromisocalendar(int(y), int(w), 1) for y, w in zip(wb.year, wb.isoweek)]
    # abroad share of Swiss debit food spending, and the share of it that stopped during the closure
    ab = v[99] / v.sum(1)
    wb["abroad_share"] = ab.reindex(wb.week_start).to_numpy()
    ab19 = (v[99] / v.sum(1)).set_axis(key)
    a19 = ab19.loc[2019]
    ab_week = ab.groupby(ab.index).first()
    inclose = (wb.week_start >= CLOSE[0]) & (wb.week_start < CLOSE[1])
    inpre = (wb.week_start >= PRE[0]) & (wb.week_start < PRE[1])
    k0 = 1 - (wb.loc[inclose, "abroad_share"].to_numpy() / a19.reindex(wb.loc[inclose, "isoweek"]).to_numpy()).mean()
    # dose-response over all weeks 2020-22: beta_w = c + b × stopped_w, stopped_w = 1 - abroad share / same week 2019.
    # The model predicts b = 2019 abroad share / model abroad share (if all abroad spending stops, beta = that ratio).
    import statsmodels.api as sm
    wb["stopped"] = 1 - wb.abroad_share.to_numpy() / a19.reindex(wb.isoweek).to_numpy()
    ok = wb[["stopped", "beta"]].notna().all(1)
    dr = sm.OLS(wb.beta[ok], sm.add_constant(wb.stopped[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": 8})
    # The model is calibrated to the 2022 abroad share; in 2019 (the comparison year) shoppers went abroad more.
    # Expected slope = (share of abroad spending that stopped) × (2019 abroad share / model abroad share).
    ratio = float(a19.mean()) / model_share
    k = k0 * ratio
    # per-region closure effect (demeaned), relative to the same weeks of 2019 and net of the pre-closure gap
    G = wb[[f"g_{i}" for i in range(1, 8)]].to_numpy()
    Gd = G - G.mean(1, keepdims=True)
    obs = Gd[inclose.to_numpy()].mean(0) - Gd[inpre.to_numpy()].mean(0)
    reg = pd.DataFrame({"region": [REGION[i] for i in range(1, 8)], "pred_closure_pct": 100 * pred,
                        "pred_demeaned_pct": 100 * pc, "pred_scaled_pct": 100 * k * pc,
                        "observed_closure_minus_pre_pct": 100 * obs})
    slope = (obs @ pc) / (pc @ pc)
    # randomisation inference over all 5,040 assignments of predictions to regions
    perms = np.array([(obs @ (pc[list(p)])) / (pc @ pc) for p in itertools.permutations(range(7))])
    p_val = (np.abs(perms) >= abs(slope) - 1e-12).mean()
    p_one = (perms >= slope - 1e-12).mean()
    corr = np.corrcoef(obs, pc)[0, 1]
    loo = {}
    for i in range(7):                                    # leave one region out
        m_ = np.arange(7) != i
        o, q = obs[m_] - obs[m_].mean(), pred[m_] - pred[m_].mean()
        loo[REGION[i + 1]] = (o @ q) / (q @ q)
    reg["slope_without_region"] = [loo[REGION[i]] for i in range(1, 8)]
    pre_beta = wb.loc[inpre, "beta"]
    out = {"share_abroad_stopped": k0, "ratio_2019_to_model_share": ratio, "expected_slope": k,
           "dose_slope": dr.params["stopped"], "dose_se": dr.bse["stopped"], "dose_const": dr.params["const"],
           "dose_weeks": int(ok.sum()), "slope_closure": slope, "slope_over_expected": slope / k, "perm_p_two_sided": p_val,
           "perm_p_one_sided": p_one, "corr_obs_pred": corr, "beta_pre_mean": pre_beta.mean(),
           "beta_close_mean": wb.loc[inclose, "beta"].mean(), "abroad_share_2019": float(a19.mean()),
           "abroad_share_closure": float(wb.loc[inclose, "abroad_share"].mean()),
           "slope_without_ticino": loo["Ticino"], "slope_loo_min": min(loo.values()), "slope_loo_max": max(loo.values())}
    del ab_week
    return wb, reg, out


# ----------------------------------------------------------------------------- B. duty-free 2025
def dutyfree(canton):
    d = pd.read_csv(MCS / "ACQ_POS_Canton.csv", parse_dates=["Date"])
    d = d[d.Canton != "9999"]
    d["m"] = d.Date.dt.to_period("M")
    last = d.Date.max()
    m = d.groupby(["m", "Canton"])["Scaled Value"].sum().unstack()
    if last < (last + pd.offsets.MonthEnd(0)):
        m = m.iloc[:-1]                                    # drop the incomplete last month
    lm = np.log(m)
    yoy = (lm - lm.shift(12)).dropna(how="all")
    x = canton.pred_closure_pct.reindex(yoy.columns).to_numpy() / 100
    w = canton.hh.reindex(yoy.columns).to_numpy()
    xc = x - np.average(x, weights=w)
    rows = []
    for per, r in yoy.iterrows():
        y = r.to_numpy()
        yc = y - np.average(y, weights=w)
        rows.append({"month": per.to_timestamp(), "beta": (w * yc * xc).sum() / (w * xc * xc).sum(),
                     "national_yoy": np.average(y, weights=w)})
    mb = pd.DataFrame(rows)
    b = mb.set_index("month").beta

    def step(y):                                          # mean beta Jan-Mar y minus Oct-Dec y-1
        return b[f"{y}-01-01":f"{y}-03-01"].mean() - b[f"{y - 1}-10-01":f"{y - 1}-12-01"].mean()

    # a level shift on 1 Jan Y lifts year-on-year growth for 12 months: step up in Jan Y, step down in Jan Y+1.
    # e_Y = (step_Y - step_Y+1) / 2 removes any smooth trend (here: post-COVID return of shopping abroad).
    est = {y: (step(y) - step(y + 1)) / 2 for y in (2022, 2023, 2024, 2025)}
    db = b.diff()["2022-01-01":"2024-12-01"].dropna()
    sigma = db.std(ddof=1) / np.sqrt(2)                   # month-to-month noise of beta
    se = sigma / np.sqrt(3)                               # Var e = (1/4) × 4 × sigma²/3
    eff = est[2025]
    naive_pre = b["2023-01-01":"2024-12-01"].mean()
    naive_post = b["2025-01-01":"2025-12-01"].mean()
    n = pd.read_csv(MCS / "ACQ_NOGA_CardholderOrigin.csv", parse_dates=["Date"])
    n = n[(n["Cardholder origin"] == "Domestic") & (n.Date.dt.year == 2024)]
    phi = n.loc[n["Merchant category"] == "Retail: Food, beverage, tobacco", "Scaled Value"].sum() / n["Scaled Value"].sum()
    out = {"dutyfree_naive_2023_24": naive_pre, "dutyfree_naive_2025": naive_post,
           "dutyfree_naive_diff": naive_post - naive_pre, "dutyfree_effect": eff, "dutyfree_se": se,
           **{f"dutyfree_placebo_{y}": est[y] for y in (2022, 2023, 2024)},
           "food_share_pos_phi": phi, "implied_f": eff / phi,
           "implied_f_lo": (eff - 1.96 * se) / phi, "implied_f_hi": (eff + 1.96 * se) / phi}
    return mb, out


# ----------------------------------------------------------------------------- figures
def figures(canton, region, reg, wb, mb, fb, s):
    viz.setup()
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cant = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)[["ref", "geometry"]]
    cant = cant[cant.ref.isin(CANTON_REGION)].copy()
    cant["region"] = cant.ref.map(CANTON_REGION)

    # 1. map: model prediction by canton | observed by Grossregion
    gdf = cant.merge(canton[["pred_closure_pct"]], left_on="ref", right_index=True)
    reg_g = cant.dissolve("region").reset_index()
    reg_g["obs"] = reg_g.region.map(dict(zip(range(1, 8), reg.observed_closure_minus_pre_pct)))
    reg_g["pred"] = reg_g.region.map(dict(zip(range(1, 8), reg.pred_scaled_pct)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = [0, 1, 2, 4, 6, 10, max(25, gdf.pred_closure_pct.max() + 1)]
    cmap, norm = ListedColormap(BLUES), BoundaryNorm(bins, len(BLUES))
    gdf.plot(ax=axes[0], column="pred_closure_pct", cmap=cmap, norm=norm, edgecolor=viz.SURFACE, linewidth=0.6)
    for _, r in gdf.iterrows():
        if r.geometry.area > 1.2e9 or r.pred_closure_pct >= 12:
            c = r.geometry.representative_point()
            axes[0].text(c.x, c.y, f"{r.ref}\n+{r.pred_closure_pct:.0f}%", ha="center", va="center", fontsize=7,
                         color="white" if r.pred_closure_pct >= 6 else viz.INK)
    axes[0].set_title("Model: domestic grocery sales gain\nif the border closes, by canton", fontsize=10.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes[0], orientation="horizontal", fraction=0.05, pad=0.02)
    cb.set_ticks(bins[:-1] + [bins[-1]])
    cb.set_ticklabels([f"{b:.0f}%" for b in bins])
    lim = 4 * np.ceil(abs(reg_g[["obs", "pred"]].to_numpy()).max() / 4)
    dcmap = plt.get_cmap("RdBu_r")
    dn = BoundaryNorm(np.linspace(-lim, lim, 9), dcmap.N)
    for ax, col, ttl in [(axes[1], "pred", "Model, by Grossregion\n(pp vs average region)"),
                         (axes[2], "obs", "Observed, closure Mar-Jun 2020\n(pp vs average region)")]:
        reg_g.plot(ax=ax, column=col, cmap=dcmap, norm=dn, edgecolor=viz.SURFACE, linewidth=0.8)
        for _, r in reg_g.iterrows():
            c = r.geometry.representative_point()
            ax.text(c.x, c.y, f"{r[col]:+.1f}", ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color="white" if abs(r[col]) > 0.6 * lim else viz.INK)
        ax.set_title(ttl, fontsize=10.5)
    sm = plt.cm.ScalarMappable(cmap=dcmap, norm=dn)
    cb = fig.colorbar(sm, ax=axes[1:], orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.6)
    cb.set_label("percentage points relative to the average region", fontsize=8.5)
    for ax in axes:
        viz.map_axes(ax, cant)
    fig.suptitle(f"Where the model said shoppers would come home, they did: regional correlation "
                 f"{s['corr_obs_pred']:.2f} [data]", x=0.02, y=1.0, ha="left", fontsize=12, fontweight="bold")
    viz.source(axes[0], "Model: Pillar 1 demand with the abroad option removed (store revenue by merchant canton) "
                        "× expected slope [calibrated].\nObserved: MCS / SIX debit-card POS at food retailers by merchant Grossregion, "
                        "weekly log change vs same ISO week 2019, closure 16 Mar-15 Jun minus Jan-Feb 2020 [data].")
    viz.save(fig, 2, "2_7", "exposure_map")

    # 2. weekly slope: dose-response
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    k = s["expected_slope"]
    r = s["ratio_2019_to_model_share"]
    for a_ in (ax, ax2):
        a_.axvspan(*CLOSE, color=viz.GRID, alpha=0.8, lw=0)
    ax.axhline(0, color=viz.INK_2, lw=0.8)
    c_fix = np.nanmean(wb.beta - r * wb.stopped)
    ax.plot(wb.week_start, wb.beta, color=viz.BLUE, lw=1.6, label="observed slope")
    ax.plot(wb.week_start, c_fix + r * wb.stopped, color=viz.ORANGE, lw=1.4, ls="--",
            label=f"model: {c_fix:.2f} + {r:.2f} × share of abroad spending stopped")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_ylabel("slope of regional sales change\non model exposure")
    ax.text(CLOSE[0] + (CLOSE[1] - CLOSE[0]) / 2, ax.get_ylim()[1], "borders\nclosed", ha="center", va="top",
            fontsize=8, color=viz.INK_2)
    ax.set_title(f"Regional sales follow the shutdown of shopping abroad week by week: response "
                 f"{s['dose_slope']:.2f} (±{1.96 * s['dose_se']:.2f}) vs {r:.2f} predicted [data]", fontsize=10.5)
    ax2.plot(wb.week_start, 100 * wb.stopped, color=viz.INK_2, lw=1.4)
    ax2.axhline(0, color=viz.INK_2, lw=0.8)
    ax2.set_ylabel("abroad spending stopped\n(% vs same week 2019)")
    viz.source(ax2, "Each week: cross-region OLS of ln(V 2020-22 / V same ISO week 2019) on predicted closure gain, both "
                    "demeaned across the 7 Grossregionen. Response: OLS of the weekly slope on the share stopped, "
                    f"{s['dose_weeks']} weeks, HAC SE. Constant = drift. MCS / SIX debit POS, food retail [data].")
    viz.save(fig, 2, "2_7", "covid_event")

    # 3. scatter
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    x, y = reg.pred_scaled_pct, reg.observed_closure_minus_pre_pct
    lo, hi = min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1
    ax.plot([lo, hi], [lo, hi], color=viz.MUTED, lw=1.2, ls="--", label="observed = predicted")
    ax.scatter(x, y, s=70, color=viz.BLUE, edgecolors=viz.SURFACE, linewidths=1.5, zorder=3)
    off = {"Northwestern": (-8, 6, "right"), "Lake Geneva": (6, -12, "left"), "Espace Mittelland": (-8, 6, "right")}
    for xi, yi, nm in zip(x, y, reg.region):
        dx, dy, ha = off.get(nm, (6, 4, "left"))
        ax.annotate(nm, (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, ha=ha)
    ax.set_xlabel("model prediction (pp vs average region)")
    ax.set_ylabel("observed closure effect (pp vs average region)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(f"7 regions: slope {s['slope_over_expected']:.2f} vs 1 predicted, correlation {s['corr_obs_pred']:.2f}\n"
                 f"(randomisation p = {s['perm_p_one_sided']:.3f}, 5,040 permutations; without Ticino slope "
                 f"{s['slope_without_ticino']:.2f}) [data]",
                 fontsize=10.5)
    viz.source(ax, "Observed: mean weekly log change vs 2019 in the closure minus Jan-Feb 2020, demeaned across regions. "
                   "Model × expected slope (share of abroad spending that stopped × 2019/2022 abroad share).")
    viz.save(fig, 2, "2_7", "covid_scatter")

    # 4. duty-free
    fig, ax = plt.subplots(figsize=(10, 4.6))
    mm = mb[mb.month >= "2022-01-01"]
    for y in (2025, 2026):
        ax.axvline(pd.Timestamp(f"{y}-01-01"), color=viz.ORANGE if y == 2025 else viz.MUTED, lw=1.2, ls="--")
    ax.axhline(0, color=viz.INK_2, lw=0.8)
    ax.plot(mm.month, mm.beta, color=viz.BLUE, lw=1.6, marker="o", ms=3)
    b = mb.set_index("month").beta
    for a0, a1 in [("2024-10-01", "2024-12-01"), ("2025-01-01", "2025-03-01"), ("2025-10-01", "2025-12-01"),
                   ("2026-01-01", "2026-03-01")]:
        v = b[a0:a1].mean()
        ax.plot([pd.Timestamp(a0) - pd.Timedelta(days=12), pd.Timestamp(a1) + pd.Timedelta(days=12)], [v, v],
                color=viz.INK, lw=2.6, solid_capstyle="butt")
    ax.text(pd.Timestamp("2025-01-10"), ax.get_ylim()[0] + 0.02, "allowance\nCHF 300 → 150", fontsize=8.5,
            color=viz.INK_2, va="bottom")
    ax.text(pd.Timestamp("2022-02-01"), ax.get_ylim()[1] - 0.02, "post-COVID return of shopping abroad:\n"
            "border cantons grow slower, slope < 0", fontsize=8, color=viz.INK_2, va="top")
    ax.set_ylabel("slope of canton year-on-year\ncard sales growth on exposure")
    lo, hi = 100 * s["implied_f_lo"], 100 * s["implied_f_hi"]
    ax.set_title(f"No break on 1 Jan 2025: implied change in food shopping abroad {100 * s['implied_f']:+.0f}% "
                 f"(95% CI {lo:+.0f}% to {hi:+.0f}%) [data]", fontsize=11)
    viz.source(ax, "Worldline-acquired POS, all categories, domestic + foreign cards, by merchant canton (MCS); monthly "
                   "ln(V_m / V_m-12) on model canton exposure, household-weighted. Black bars: Oct-Dec and Jan-Mar means; "
                   "effect = (step into Jan 2025 - step into Jan 2026) / 2, scaled by food share of card POS.")
    viz.save(fig, 2, "2_7", "dutyfree")

    # 5. franc
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.set_axisbelow(True)
    a2.set_axisbelow(True)
    xs = np.arange(len(fb))
    a1.bar(xs - 0.2, 100 * fb.abroad_share_before, 0.38, color=viz.MUTED, label="EUR/CHF 1.20 (before)")
    a1.bar(xs + 0.2, 100 * fb.abroad_share_after, 0.38, color=viz.BLUE, label="EUR/CHF 1.05 (after)")
    a1.set_xticks(xs, fb.index)
    a1.set_xlabel("drive minutes to the border")
    a1.set_ylabel("share of grocery spend abroad (%)")
    a1.legend(fontsize=8.5)
    a1.set_title("Abroad share", fontsize=10.5)
    a2.bar(xs, fb.gain_pct_of_spend, 0.6, color=viz.BLUE)
    for xi, v in zip(xs, fb.gain_pct_of_spend):
        a2.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    a2.set_xticks(xs, fb.index)
    a2.set_xlabel("drive minutes to the border")
    a2.set_ylabel("gain, % of grocery spend")
    a2.set_title("Shopper gain from cheaper abroad baskets", fontsize=10.5)
    fig.suptitle(f"Franc shock 2015: model gain {fb.gain_pct_of_spend.iloc[0]:.1f}% of grocery spend within 10 min of the border "
                 f"vs {fb.gain_pct_of_spend.iloc[4]:.1f}% beyond 45 min [calibrated]",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    viz.source(a1, "Pillar 1 model with the abroad basket price × 1.20/1.05 (before) vs today. Gain = Δ logsum / α. "
                   "Published comparison: cost of living fell 2.8% near the border vs 1.7% far away (Burstein, Lein & "
                   "Vogel, JIE 2024), all goods and channels.")
    viz.save(fig, 2, "2_7", "franc")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    canton, region, nat, fb = model_side()
    print(canton.round(4).to_string())
    print(region.round(4).to_string())
    print(nat)
    print(fb.round(4).to_string())
    v = weekly_food()
    wb, reg, sc = covid(region, v, nat["abroad_share_model"])
    print(reg.round(3).to_string())
    print({k: round(float(x), 4) for k, x in sc.items()})
    mb, sd = dutyfree(canton)
    print({k: round(float(x), 4) for k, x in sd.items()})
    s = {**nat, **sc, **sd}
    canton.round(5).to_csv(OUT / "2_7_canton_exposure.csv")
    reg.round(4).to_csv(OUT / "2_7_region_covid.csv", index=False)
    wb.round(5).to_csv(OUT / "2_7_covid_weekly.csv", index=False)
    mb.round(5).to_csv(OUT / "2_7_dutyfree_monthly.csv", index=False)
    fb.round(5).to_csv(OUT / "2_7_franc_bands.csv")
    pd.Series(s).to_frame("value").round(5).to_csv(OUT / "2_7_summary.csv")
    figures(canton, region, reg, wb, mb, fb, s)


if __name__ == "__main__":
    main()
