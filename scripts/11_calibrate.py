"""Phase 2 step 2: calibrate banner qualities delta to 2025 sales shares and the card-data abroad share.

Run:  conda activate keystone && python scripts/11_calibrate.py      (needs 10_demand.py first)
Outputs (assets/p1/m2/):
  1_2_abroad_share.csv      Swiss debit-card grocery spending abroad vs at home, 2019-2022 (MCS)
  1_2_targets.csv           shares of total grocery spending used as targets
  1_2_fit.csv               target vs model after calibration
  1_2_delta.csv             calibrated delta per banner, in utility units and in CHF per basket
  1_2_convergence.csv       contraction history
  1_2_canton_shares.csv     model shares by canton: merging parties, discounters, abroad
  1_2_params.json           calibrated demand parameters (input to step 1.3 onward)
  1_2_convergence.png, 1_2_brand_premium.png, 1_2_merging_share_map.png
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from mxc import viz  # noqa: E402
from mxc.structural import calibrate as cb  # noqa: E402
from mxc.structural import demand as dm  # noqa: E402

PROCESSED, INTERIM, RAW = ROOT / "data/processed", ROOT / "data/interim", ROOT / "data/raw"
OUT = ROOT / "assets/p1/m2"
ABROAD_YEAR = 2022   # latest full year; 2019-21 are distorted by the cash-to-card shift and COVID
MERGING = ["Migros", "Migros Partner", "Migrolino", "Denner", "Coop", "Coop Pronto"]
SRC = "Sources: OpenStreetMap, FSO STATPOP 2024, ESTV 2023, MCS/SIX debit data, watson.ch (2025 shares)."


def load_market():
    z = np.load(PROCESSED / "market.npz", allow_pickle=True)
    return dm.Market(**{k: z[k] for k in z.files})


def location_shares(theta, a, banners):
    """(L,) revenue share of the given banners among all grocery spending (incl. abroad) per location."""
    s_pair, s0, _ = dm.choice_probs(theta, a)
    b = a["store_banner"][a["pair_store"]]
    w = a["type_w"]
    rev_pair = (s_pair * w[a["pair_loc"]]).sum(1) * theta.p[b]
    L = a["hh"].shape[0]
    tot = jax.ops.segment_sum(rev_pair, a["pair_loc"], num_segments=L) + (s0 * w).sum(1) * theta.p_abroad
    sel = jnp.isin(b, jnp.array([dm.BANNERS.index(x) for x in banners]))
    return np.asarray(jax.ops.segment_sum(jnp.where(sel, rev_pair, 0.0), a["pair_loc"], num_segments=L) / tot)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mkt = load_market()
    a = dm.to_arrays(mkt)

    # --- 1. abroad share from card data
    cards = cb.abroad_share_from_cards(RAW / "mcs/DEBIT_Merchanttype_Grossregion.csv")
    cards.round(4).to_csv(OUT / "1_2_abroad_share.csv")
    abroad = float(cards.loc[ABROAD_YEAR, "abroad_share"])
    print(f"abroad share of grocery card spending, {ABROAD_YEAR}: {abroad:.4f}")

    # --- 2. targets and calibration
    tgt = cb.targets(abroad)
    tgt.rename("target_share").round(5).to_frame().to_csv(OUT / "1_2_targets.csv")
    theta0 = dm.prior_params()
    theta, hist = cb.calibrate(theta0, a, tgt)
    hist.to_csv(OUT / "1_2_convergence.csv", index=False)
    print(f"converged in {len(hist) - 1} iterations; max |ln gap| = {hist.max_abs_log_gap.iloc[-1]:.2e}")

    model = np.asarray(cb.model_revenue_shares(theta, a, cb.player_index()))
    fit = pd.DataFrame({"target": tgt.to_numpy(), "model": model}, index=tgt.index)
    fit["prior_delta0"] = hist.iloc[0][list(tgt.index)].to_numpy()
    fit["abs_error_pp"] = (fit.model - fit.target).abs() * 100
    fit.round(5).to_csv(OUT / "1_2_fit.csv")
    print(fit.round(4).to_string())

    # --- 3. delta table: utility units, CHF per basket at the reference income, split within players
    q_store, _ = dm.quantities(theta, a)
    rev_b = np.asarray(jax.ops.segment_sum(q_store * theta.p[a["store_banner"]], a["store_banner"],
                                           num_segments=len(dm.BANNERS)))
    pidx = cb.player_index()
    rev_player = np.bincount(pidx, weights=rev_b)
    d = np.asarray(theta.delta)
    dtab = pd.DataFrame({
        "player": [list(cb.PLAYERS)[i] for i in pidx], "nest": [dm.BANNER_NEST[b] for b in dm.BANNERS],
        "price_index": np.asarray(theta.p), "delta": d,
        "delta_vs_migros_chf": (d - d[dm.BANNERS.index("Migros")]) / theta.alpha_ref,
        "share_within_player": rev_b / rev_player[pidx],
        "stores": np.bincount(mkt.store_banner, minlength=len(dm.BANNERS)),
    }, index=dm.BANNERS)
    dtab.round(4).to_csv(OUT / "1_2_delta.csv")
    print(dtab.round(3).to_string())

    # --- 4. local shares
    loc = pd.read_parquet(PROCESSED / "locations.parquet", columns=["loc_id", "x", "y"]).set_index("loc_id").loc[mkt.loc_id]
    merging = location_shares(theta, a, MERGING)
    disc = location_shares(theta, a, ["Aldi", "Lidl"])
    s_pair, s0, _ = dm.choice_probs(theta, a)
    abroad_loc = np.asarray((s0 * a["type_w"]).sum(1))
    df = pd.DataFrame({"canton": mkt.canton, "hh": mkt.hh, "merging": merging, "aldi_lidl": disc, "abroad_baskets": abroad_loc})
    can = df.groupby("canton").apply(lambda g: pd.Series({
        "households": g.hh.sum(), "merging_parties_share": np.average(g.merging, weights=g.hh),
        "aldi_lidl_share": np.average(g.aldi_lidl, weights=g.hh),
        "abroad_basket_share": np.average(g.abroad_baskets, weights=g.hh)}), include_groups=False)
    can = can.sort_values("merging_parties_share", ascending=False)
    can.round(4).to_csv(OUT / "1_2_canton_shares.csv")
    print(can.round(3).to_string())

    json.dump({"delta": dict(zip(dm.BANNERS, map(float, d))), "alpha_ref": theta.alpha_ref, "eta": theta.eta,
               "gamma": theta.gamma, "kappa": theta.kappa, "sigma": theta.sigma,
               "price_index": dict(zip(dm.BANNERS, map(float, np.asarray(theta.p)))), "p_abroad": theta.p_abroad,
               "abroad_share_target": abroad, "abroad_year": ABROAD_YEAR},
              open(OUT / "1_2_params.json", "w"), indent=2)

    figures(hist, dtab, fit, loc, merging, df)


def figures(hist, dtab, fit, loc, merging, df):
    viz.setup()
    # convergence
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.semilogy(hist["iter"], hist.max_abs_log_gap, color=viz.BLUE)
    ax.set_xlabel("iteration")
    ax.set_ylabel("largest |ln target − ln model|")
    ax.set_title(f"The contraction converges in {len(hist) - 1} iterations")
    viz.source(ax, "Rule: delta += 0.75 × ln(target / model), all banners of a player together; abroad fixed at 0.")
    viz.save(fig, 1, "1_2", "convergence")

    # brand premium in CHF per basket vs Migros, one bar per player (banners of a player share delta)
    t = dtab.groupby("player").delta_vs_migros_chf.first().sort_values()
    col = [viz.BLUE if p in ("Migros", "Denner") else viz.ORANGE if p == "Coop" else viz.MUTED for p in t.index]
    names = {"Migros": "Migros (+ Partner, Migrolino)", "Coop": "Coop (+ Pronto)",
             "Others": "Others (Volg, Landi, Spar, Manor)"}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh([names.get(p, p) for p in t.index], t.to_numpy(), color=col, height=0.6)
    ax.axvline(0, color=viz.INK_2, linewidth=0.8)
    for y, v in enumerate(t.to_numpy()):
        ax.text(v - 0.6 if v < -0.5 else 0.6, y, "reference" if abs(v) < 0.5 else f"{v:+.0f}", va="center",
                ha="right" if v < -0.5 else "left", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(t.min() - 6, 6)
    ax.set_xlabel("brand quality vs Migros, CHF per basket:  (δ − δ_Migros) / α_ref")
    ax.set_title("Shoppers value Coop like Migros; discounters and small formats far less")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=viz.BLUE, label="Migros group"), Patch(color=viz.ORANGE, label="Coop group"),
                       Patch(color=viz.MUTED, label="rivals")], loc="upper left")
    viz.source(ax, "δ absorbs all that price and drive time miss: range, store size, habit. α_ref = 0.03 (income CHF 62,500).")
    viz.save(fig, 1, "1_2", "brand_premium")

    # merging parties' combined share by location
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cantons = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)
    bins = [0, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap, norm = ListedColormap(ramp), BoundaryNorm(bins, len(ramp))
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    viz.map_axes(ax, cantons)
    o = np.argsort(merging)
    ax.scatter(loc.x.to_numpy()[o], loc.y.to_numpy()[o], c=merging[o], cmap=cmap, norm=norm, s=1.2, linewidths=0, zorder=2)
    nat = np.average(merging, weights=df.hh)
    hi = (merging >= 0.80) @ df.hh / df.hh.sum()
    ax.set_title(f"Migros + Coop would hold {nat:.0%} of grocery spending, and ≥ 80% for {hi:.0%} of households")
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal",
                       fraction=0.04, pad=0.02, aspect=40, ticks=bins[1:-1])
    cb_.ax.set_xticklabels([f"{b:.0%}" for b in bins[1:-1]], fontsize=8)
    cb_.set_label("merging parties' share of local grocery spending (incl. abroad), model [calibrated]", fontsize=8.5,
                  color=viz.INK_2)
    cb_.outline.set_visible(False)
    viz.source(ax, "Migros group = Migros, Partner, Migrolino, Denner; Coop group = Coop, Coop Pronto. " + SRC)
    viz.save(fig, 1, "1_2", "merging_share_map")


if __name__ == "__main__":
    main()
