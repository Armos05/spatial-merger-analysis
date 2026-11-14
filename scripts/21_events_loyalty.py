"""Pillar 2 step 2: events in the twin (staggered Aldi/Lidl openings at real locations, a Spar closure wave) and a
loyalty-card panel (individual households, their trips, what Cumulus and Supercard each see). Everything is [sim].

The events twin and a counterfactual twin (the 120 entrants never open, no closures) use the same seed as 2.1, so
their shocks are identical: the gap between them is the exact effect of the events, week by week.

Run:  conda activate keystone && python scripts/21_events_loyalty.py      (needs 10-13; reuses 20_twin.py helpers)
Data (data/processed/twin/events/, not in git): store_week, truth_week, category, weekly, store, households, trips,
  cumulus, supercard, entry_truth, closure_truth (.parquet)
Outputs (assets/p2/m2/):
  2_2_events.csv                 every entry and closure: store, banner, canton, week, date
  2_2_entry_truth_summary.csv    true sales loss of nearby stores by distance band and player (answer key for T1)
  2_2_entry_sources.csv          where each entrant's sales come from (true diversion to the entrant)
  2_2_closure_truth.csv          where a closed store's shoppers go
  2_2_loyalty_config.csv         panel settings and segment definitions
  2_2_loyalty_summary.csv        households, trips, and what each card programme sees, by segment
  2_2_events_map.png, 2_2_entry_timeline.png, 2_2_entry_distance.png, 2_2_entry_sources.png,
  2_2_event_example.png, 2_2_loyalty_coverage.png, 2_2_coverage_map.png
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from mxc import viz  # noqa: E402
from mxc.structural import calibrate as cb  # noqa: E402
from mxc.structural import demand as dm  # noqa: E402
from mxc.twin import events as evm  # noqa: E402
from mxc.twin import generate as tw  # noqa: E402
from mxc.twin import loyalty as ly  # noqa: E402

T20 = runpy.run_path(str(ROOT / "scripts/20_twin.py"), run_name="lib")
PROCESSED, INTERIM = ROOT / "data/processed", ROOT / "data/interim"
EV = PROCESSED / "twin/events"
OUT = ROOT / "assets/p2/m2"
BLUES = ["#d6e6f8", "#a9cbf0", "#6fa6e6", "#2a78d6", "#1d58a3", "#123a6e"]
BANDS = [0, 1, 2, 3, 5, 7.5, 10, 15, 20]
PLAYER = {b: p for p, bs in cb.PLAYERS.items() for b in bs}
GROUPS = {"Migros": "Migros group", "Denner": "Migros group", "Coop": "Coop group", "Aldi": "Aldi / Lidl",
          "Lidl": "Aldi / Lidl", "Others": "Others"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    EV.mkdir(parents=True, exist_ok=True)
    mkt, theta = T20["load"]()
    a = dm.to_arrays(mkt)
    stores = pd.read_parquet(PROCESSED / "stores.parquet", columns=["store_id", "canton", "lon", "lat"])
    stores = stores.set_index("store_id").loc[mkt.store_id]
    pts = gpd.GeoSeries(gpd.points_from_xy(stores.lon, stores.lat), crs=4326).to_crs(2056)
    xy_km = np.c_[pts.x, pts.y] / 1000
    sbn = np.asarray(a["store_banner"])
    banner = np.array(dm.BANNERS)[sbn]
    group = np.array([GROUPS[PLAYER[b]] for b in banner])
    cfg = tw.TwinConfig()
    nest = np.asarray(mkt.store_nest)

    # 1. events and the two twins (same shocks)
    open_w, close_w, ev = evm.draw_events(cfg.seed, sbn, cfg.weeks)
    never = np.where(open_w > 0, 10 ** 6, 0)                       # counterfactual: entrants never open, no closures
    base = tw.generate(theta, a, stores.canton.to_numpy(), nest, cfg, store_id=mkt.store_id, open_week=never)
    world = tw.generate(theta, a, stores.canton.to_numpy(), nest, cfg, store_id=mkt.store_id,
                        open_week=open_w, close_week=close_w)
    for k in ("store_week", "truth_week", "category", "weekly", "store"):
        world[k].to_parquet(EV / f"{k}.parquet", index=False)
    dates = world["arrays"]["dates"]
    ev = ev.assign(store_id=mkt.store_id[ev.store], canton=stores.canton.to_numpy()[ev.store],
                   date=dates[ev.week].date)
    ev.to_csv(OUT / "2_2_events.csv", index=False)
    print(ev.groupby("event").size())

    # 2. answer key at normal times
    xi_perm = world["arrays"]["xi_perm"]
    et, src = evm.entry_truth(theta, a, xi_perm, ev, xy_km)
    ct, csum = evm.closure_truth(theta, a, xi_perm, ev, xy_km)
    et["group"], ct["group"] = group[et.store], group[ct.store]
    et.to_parquet(EV / "entry_truth.parquet", index=False)
    ct.to_parquet(EV / "closure_truth.parquet", index=False)
    et["band"] = pd.cut(et.dist_km, BANDS)
    summ = et.groupby(["band", "group"], observed=True).apply(
        lambda g: pd.Series({"pairs": len(g), "mean_loss_pct": g.loss_pct.mean(),
                             "weighted_loss_pct": 100 * -g.dq.sum() / g.q_base.sum(),
                             "take_share_total": g.take_share.sum() / g.entrant.nunique()}), include_groups=False)
    summ.round(4).to_csv(OUT / "2_2_entry_truth_summary.csv")
    take = et.groupby(["entrant", "group"]).take_share.sum().unstack(fill_value=0)
    src = src.set_index("entrant").join(take)
    src["beyond_20km"] = src.take_all_stores - src.take_within_radius
    src.insert(0, "store_id", mkt.store_id[src.index])
    src.insert(1, "banner", banner[src.index])
    src.round(4).to_csv(OUT / "2_2_entry_sources.csv")
    cdiv = ct.groupby(["closed", "group"]).diversion.sum().unstack(fill_value=0)
    csum = csum.set_index("closed").join(cdiv)
    csum.insert(0, "store_id", mkt.store_id[csum.index])
    csum.round(4).to_csv(OUT / "2_2_closure_truth.csv")
    print("entrant sales come from (mean):\n", src[list(take.columns) + ["take_abroad", "beyond_20km"]].mean().round(3))
    print("closed Spar shoppers go to (mean):\n", csum.drop(columns=["store_id", "q_closed"]).mean().round(3))

    # realised effect in the twin vs the answer key, stores within 3 km of each entrant.
    # r_t = Σ q_world / Σ q_counterfactual over the entrant's near stores (shocks cancel: both worlds share them).
    # jump  = r(week before opening) − r(opening week): the clean effect of this entrant alone.
    # window = mean r (26 weeks before) − mean r (26 weeks after): also picks up other entrants opening nearby.
    qb, qw = base["arrays"]["q"], world["arrays"]["q"]
    near = et[et.dist_km <= 3]
    real, jump, struct = [], [], []
    for e, g in near.groupby("entrant"):
        w0 = open_w[e]
        r = qw[:, g.store].sum(1) / qb[:, g.store].sum(1)
        real.append(100 * (r[max(0, w0 - 26):w0].mean() - r[w0:w0 + 26].mean()))
        jump.append(100 * (r[w0 - 1] - r[w0]))
        struct.append(100 * -g.dq.sum() / g.q_base.sum())
    check = {"entrants with stores within 3 km": len(real),
             "answer key: loss of stores within 3 km, mean over entrants, %": float(np.mean(struct)),
             "twin: opening-week jump, mean over entrants, %": float(np.mean(jump)),
             "correlation across entrants (jump vs answer key)": float(np.corrcoef(struct, jump)[0, 1]),
             "twin: 26-week before/after window, mean over entrants, %": float(np.mean(real)),
             "correlation across entrants (window vs answer key)": float(np.corrcoef(struct, real)[0, 1])}
    print(check)
    pd.Series(check).round(3).to_csv(OUT / "2_2_entry_check.csv", header=["value"])
    print(check)

    # 3. loyalty panel
    lcfg = ly.LoyaltyConfig()
    hh = ly.draw_households(theta, a, lcfg)
    hh.insert(1, "loc_id", mkt.loc_id[hh["loc"]])
    hh.insert(2, "canton", mkt.canton[hh["loc"]])
    trips = ly.simulate_trips(theta, a, hh, world["arrays"], lcfg)
    trips["store_id"] = np.where(trips.store >= 0, mkt.store_id[trips.store.clip(0)], "abroad")
    hh.to_parquet(EV / "households.parquet", index=False)
    trips.to_parquet(EV / "trips.parquet", index=False)
    cols = ["hh_id", "week", "store_id", "banner", "spend"] + [f"dept_{d}" for d in ly.DEPARTMENTS]
    home = hh.set_index("hh_id").loc_id
    for card in ("cumulus", "supercard"):
        v = trips.loc[trips[f"seen_{card}"], cols].rename(columns={"hh_id": "card_id"})
        v.insert(1, "home_loc_id", home.loc[v.card_id].to_numpy())
        v.to_parquet(EV / f"{card}.parquet", index=False)
    ly.config_table(lcfg).to_csv(OUT / "2_2_loyalty_config.csv", index=False)

    t = trips.merge(hh[["hh_id", "segment", "canton"]], on="hh_id")
    t["seen_merged"] = t.seen_cumulus | t.seen_supercard
    t["abroad"] = t.store < 0

    def cover(g):
        sp = g.spend.sum()
        return pd.Series({"households": g.hh_id.nunique(), "trips": len(g),
                          "trips_per_hh_week": len(g) / g.hh_id.nunique() / cfg.weeks,
                          "spend_per_trip": g.spend.mean(), "abroad_trip_share": g.abroad.mean(),
                          "seen_by_cumulus_trips": g.seen_cumulus.mean(), "seen_by_supercard_trips": g.seen_supercard.mean(),
                          "seen_by_merged_trips": g.seen_merged.mean(),
                          "seen_by_cumulus_spend": g.spend[g.seen_cumulus].sum() / sp,
                          "seen_by_supercard_spend": g.spend[g.seen_supercard].sum() / sp,
                          "seen_by_merged_spend": g.spend[g.seen_merged].sum() / sp,
                          "true_alpha_mean": hh.set_index("hh_id").alpha_i.loc[g.hh_id.unique()].mean()})
    ls = pd.concat([t.groupby("segment").apply(cover, include_groups=False),
                    cover(t).to_frame("all").T])
    ls.round(4).to_csv(OUT / "2_2_loyalty_summary.csv")
    print(ls.round(3).to_string())
    can = t.groupby("canton")[["seen_cumulus", "seen_merged"]].mean()

    figures(ev, et, src, take.columns, qb, qw, world, open_w, xy_km, banner, group, ls, can, check)


def figures(ev, et, src, take_cols, qb, qw, world, open_w, xy_km, banner, group, ls, can, check):
    viz.setup()
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cantons = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)
    dates = world["arrays"]["dates"]
    X, Y = xy_km[:, 0] * 1000, xy_km[:, 1] * 1000

    # 1. map of events
    fig, ax = plt.subplots(figsize=(9, 6))
    viz.map_axes(ax, cantons)
    ax.scatter(X, Y, s=1.5, color=viz.MUTED, linewidths=0, zorder=2, label="other stores")
    ent = ev[ev.event == "entry"]
    yrs = dates[ent.week].year
    for y, colr in zip([2022, 2023, 2024], [BLUES[2], BLUES[3], BLUES[5]]):
        s = ent.store[yrs == y]
        ax.scatter(X[s], Y[s], s=26, color=colr, edgecolors=viz.SURFACE, linewidths=0.8, zorder=4,
                   label=f"Aldi/Lidl opens {y} ({len(s)})")
    c = ev.store[ev.event == "closure"]
    ax.scatter(X[c], Y[c], s=30, marker="X", color=viz.ORANGE, edgecolors=viz.SURFACE, linewidths=0.6, zorder=4,
               label=f"Spar closes {dates[ev.week[ev.event == 'closure'].iloc[0]].date()} ({len(c)})")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, markerscale=1)
    ax.set_title("120 simulated discounter openings at real Aldi/Lidl sites, and one closure wave [sim]", loc="left")
    viz.source(ax, "Locations are real OSM stores; which ones are 'new' and when they open is random. 2_2_events.csv")
    viz.save(fig, 2, "2_2", "events_map")

    # 2. timeline
    fig, ax = plt.subplots(figsize=(8, 3.2))
    q = pd.Series(1, index=pd.DatetimeIndex(dates[ent.week])).resample("QS").sum()
    ax.bar(q.index, q.values, width=70, color=viz.BLUE)
    ax.set_ylabel("openings per quarter")
    ax.set_title("Openings are staggered over two years: a clean staggered-adoption design [sim]")
    viz.source(ax, "Opening weeks drawn uniformly between week 40 and week 130 of the 156-week panel.")
    viz.save(fig, 2, "2_2", "entry_timeline")

    # 3. true loss by distance
    fig, ax = plt.subplots(figsize=(8, 4.2))
    mids = [(lo + hi) / 2 for lo, hi in zip(BANDS[:-1], BANDS[1:])]
    for gname, colr in zip(["Aldi / Lidl", "Migros group", "Coop group", "Others"],
                           [viz.AQUA, viz.BLUE, viz.ORANGE, viz.MUTED]):
        g = et[et.group == gname]
        w = g.groupby(pd.cut(g.dist_km, BANDS), observed=False).apply(
            lambda x: 100 * -x.dq.sum() / x.q_base.sum() if len(x) else np.nan, include_groups=False)
        ax.plot(mids, w.values, marker="o", color=colr, label=gname)
    ax.set_xlabel("distance from the new store, km (straight line)")
    ax.set_ylabel("true loss of baskets, %")
    ax.legend(frameon=False)
    g1 = et[(et.dist_km < 1) & (et.group == "Aldi / Lidl")]
    ax.set_title(f"Losses fade with distance; discounters within 1 km lose most ({100 * -g1.dq.sum() / g1.q_base.sum():.1f}%)")
    viz.source(ax, "Answer key at normal times: model with vs without the entrant, all other entrants closed. [sim]")
    viz.save(fig, 2, "2_2", "entry_distance")

    # 4. where the entrant's sales come from
    parts = list(take_cols) + ["beyond_20km", "take_abroad"]
    lab = {"take_abroad": "abroad", "beyond_20km": "stores > 20 km"}
    m = src[parts].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colmap = {"Migros group": viz.BLUE, "Coop group": viz.ORANGE, "Aldi / Lidl": viz.AQUA}
    ax.barh([lab.get(k, k) for k in m.index], 100 * m.values, color=[colmap.get(k, viz.MUTED) for k in m.index],
            height=0.6)
    for i, v in enumerate(m.values):
        ax.text(100 * v + 0.6, i, f"{100 * v:.0f}%", va="center", fontsize=9)
    ax.set_xlabel("share of the new store's baskets taken from ..., % (mean over 120 entrants)")
    ax.grid(axis="y", visible=False)
    top = m.index[-1]
    ax.set_title(f"{100 * m[top]:.0f}% of a new discounter's sales come from the {lab.get(top, top)}")
    viz.source(ax, "True diversion to the entrant (2_2_entry_sources.csv). This is what T1 must recover. [sim]")
    viz.save(fig, 2, "2_2", "entry_sources")

    # 5. one event, observed vs truth
    near = et[(et.dist_km < 2) & (et.group == "Migros group")].sort_values("loss_pct", ascending=False).iloc[0]
    e, k = int(near.entrant), int(near.store)
    w0 = open_w[e]
    win = np.arange(max(0, w0 - 30), min(len(dates), w0 + 31))
    sw = world["store_week"]
    obs = sw[sw.store == k].set_index("week").baskets.reindex(win)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(dates[win], obs.values, color=viz.INK_2, linewidth=1, marker="o", markersize=3, label="observed baskets")
    ax.plot(dates[win], qb[win, k], color=viz.MUTED, linewidth=2, label="truth without the entrant")
    ax.plot(dates[win], qw[win, k], color=viz.BLUE, linewidth=2, label="truth with the entrant")
    ax.axvline(dates[w0], color=viz.ORANGE, linewidth=1.2)
    ax.text(dates[w0], ax.get_ylim()[1], f"  {banner[e]} opens {near.dist_km:.1f} km away", color=viz.ORANGE,
            va="top", fontsize=9)
    ax.set_ylabel("baskets per week")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    typical = et[et.dist_km <= 3]
    ax.set_title(f"The hardest-hit store: this {banner[k]} loses {near.loss_pct:.0f}%; a typical store within 3 km loses "
                 f"{typical.loss_pct.median():.1f}%")
    viz.source(ax, "Same shocks in both worlds, so the gap between the two truth lines is the exact effect. [sim]")
    viz.save(fig, 2, "2_2", "event_example")

    # 6. loyalty coverage by segment (spend)
    segs = list(ly.SEGMENTS) + ["all"]
    L = ls.loc[segs]
    cm, cs = 100 * L.seen_by_cumulus_spend, 100 * L.seen_by_supercard_spend
    rest = 100 - cm - cs
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    yl = [s.replace("_", " ") for s in segs][::-1]
    ax.barh(yl, cm[::-1], color=viz.BLUE, height=0.6, label="seen by Cumulus (Migros)")
    ax.barh(yl, cs[::-1], left=cm[::-1], color=viz.ORANGE, height=0.6, label="seen by Supercard (Coop)")
    ax.barh(yl, rest[::-1], left=(cm + cs)[::-1], color=viz.MUTED, height=0.6, label="seen by neither")
    for i, (a_, b_) in enumerate(zip(cm[::-1], (cm + cs)[::-1])):
        ax.text(b_ + 1, i, f"merged {b_:.0f}%", va="center", fontsize=8.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("share of panel grocery spend, %")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22), fontsize=8.5)
    ax.grid(axis="y", visible=False)
    A_ = L.loc["all"]
    ax.set_title(f"Cumulus sees {A_.seen_by_cumulus_spend:.0%} of grocery spend, Supercard {A_.seen_by_supercard_spend:.0%}; "
                 f"merged, {A_.seen_by_merged_spend:.0%}")
    viz.source(ax, "20,000 simulated households; neither = Denner, discounters, others, abroad, no card or no swipe. [sim]")
    viz.save(fig, 2, "2_2", "loyalty_coverage")

    # 7. map: share of trips seen, Migros alone vs merged, by home canton
    cc = cantons[cantons.ref.notna()].copy()
    cc = cc.merge(can.reset_index(), left_on="ref", right_on="canton", how="left")
    v = np.r_[cc.seen_cumulus, cc.seen_merged] * 100
    bins = np.round(np.nanquantile(v, np.linspace(0, 1, 7)), 0)
    bins[0], bins[-1] = np.nanmin(v) - 0.5, np.nanmax(v) + 0.5
    cmap, norm = ListedColormap(BLUES), BoundaryNorm(bins, len(BLUES))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, ttl in zip(axes, ["seen_cumulus", "seen_merged"], ["Migros today (Cumulus)", "Merged firm"]):
        cc.plot(column=cc[col] * 100, ax=ax, cmap=cmap, norm=norm, edgecolor=viz.SURFACE, linewidth=0.6)
        ax.set_axis_off()
        ax.set_title(f"{ttl}", fontsize=10.5)
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, orientation="horizontal",
                       fraction=0.05, pad=0.03, aspect=50, ticks=bins[1:-1])
    cb_.ax.set_xticklabels([f"{b:.0f}" for b in bins[1:-1]], fontsize=8)
    cb_.set_label("share of panel households' grocery trips recorded on the card, % (by home canton)", fontsize=8.5)
    cb_.outline.set_visible(False)
    fig.suptitle(f"The merged firm's cards would record {can.seen_merged.mean():.0%} of trips (canton mean), "
                 f"vs {can.seen_cumulus.mean():.0%} for Migros alone [sim]",
                 x=0.01, y=0.97, ha="left", fontweight="bold")
    viz.save(fig, 2, "2_2", "coverage_map")


if __name__ == "__main__":
    main()
