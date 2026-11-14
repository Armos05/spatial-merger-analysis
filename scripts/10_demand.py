"""Phase 2 step 1: build the spatial nested-logit demand system on real Swiss geography.

This step assembles the choice sets (every location's stores within 25 minutes, plus abroad) and the
income types. It also evaluates demand at the prior parameters, to check the machinery at full
scale. Banner qualities delta are still 0 here; step 2 (11_calibrate.py) fits them to market shares.

Run:  conda activate keystone && python scripts/10_demand.py
Outputs:
  data/processed/market.npz            choice-set arrays (input to every later Pillar 1 step)
  assets/p1/1_1_market_summary.csv     size of the market and the choice sets
  assets/p1/1_1_overlap.csv            households reaching Migros-group and Coop-group stores
  assets/p1/1_1_prior_shares.csv       national shares at prior parameters, delta = 0 [uncalibrated]
  assets/p1/1_1_store_map.png          where each group's stores are
  assets/p1/1_1_overlap_map.png        where only the merging groups are in reach
  assets/p1/1_1_reach.png              households within x minutes of each group
  assets/p1/1_1_income_map.png         median taxable-unit net income by municipality (drives alpha)
"""
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
import pyproj  # noqa: E402

from mxc import viz  # noqa: E402
from mxc.structural import demand as dm  # noqa: E402

PROCESSED, INTERIM = ROOT / "data/processed", ROOT / "data/interim"
OUT = ROOT / "assets/p1"
SRC = "Sources: OpenStreetMap (ODbL), FSO STATPOP 2024, ESTV 2023; own road-network drive times."
GROUPS = {"Migros group": ["Migros"], "Coop group": ["Coop"], "Aldi + Lidl": ["Aldi", "Lidl"]}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    loc = pd.read_parquet(PROCESSED / "locations.parquet")
    stores = pd.read_parquet(PROCESSED / "stores.parquet", columns=["store_id", "banner", "group", "canton"])
    dt = pd.read_parquet(PROCESSED / "drive_times.parquet", columns=["loc_id", "store_id", "drive_min"])
    muni = pd.read_parquet(PROCESSED / "municipalities.parquet")

    mkt, info = dm.build_market(loc, stores, dt, muni)
    np.savez_compressed(PROCESSED / "market.npz", **asdict(mkt))
    print(f"market built in {time.time() - t0:.0f}s: {info}")

    # --- size of the market and the choice sets
    per_loc = np.bincount(mkt.pair_loc, minlength=mkt.L)
    near = np.full(mkt.L, np.inf)
    np.minimum.at(near, mkt.pair_loc, mkt.pair_time)
    hh = mkt.hh
    wq = lambda x, q: float(np.interp(q, np.cumsum(hh[np.argsort(x)]) / hh.sum(), np.sort(x)))  # noqa: E731
    summary = pd.DataFrame([
        ("shopper locations", info["locations_kept"]),
        ("locations dropped (no store within 25 min)", info["locations_dropped"]),
        ("households dropped", info["households_dropped"]),
        ("households in market", int(hh.sum())),
        ("stores (12 banners)", info["stores"]),
        ("location-store pairs (<= 25 min)", info["pairs"]),
        ("stores in reach, household median", wq(per_loc, 0.5)),
        ("stores in reach, household p10", wq(per_loc, 0.1)),
        ("nearest store minutes, household median", round(wq(near, 0.5), 1)),
        ("nearest store minutes, household p90", round(wq(near, 0.9), 1)),
        ("border minutes, household median", round(wq(mkt.border_min, 0.5), 1)),
    ], columns=["item", "value"])
    summary.to_csv(OUT / "1_1_market_summary.csv", index=False)
    print(summary.to_string(index=False))

    # --- overlap: share of households with both groups' stores nearby (where the merger bites)
    rows, only_10 = [], None
    for m in [5, 10, 15]:
        sel = mkt.pair_time <= m
        grp = mkt.store_group[mkt.pair_store[sel]]
        has = lambda g: np.bincount(mkt.pair_loc[sel][grp == g], minlength=mkt.L) > 0  # noqa: E731
        both = has("Migros") & has("Coop")
        only = both & ~(has("Aldi") | has("Lidl") | has("fenaco") | has("Spar") | has("Manor"))  # 2 -> 1
        if m == 10:
            only_10 = only
        rows.append({"within_min": m, "households_with_migros_group_and_coop_group": hh[both].sum() / hh.sum(),
                     "households_with_both_groups_and_no_rival": hh[only].sum() / hh.sum()})
    overlap = pd.DataFrame(rows).round(3)
    overlap.to_csv(OUT / "1_1_overlap.csv", index=False)
    print("\n", overlap.to_string(index=False))

    # --- full-scale evaluation at prior parameters (delta = 0): checks the machinery, not a result
    theta = dm.prior_params()
    a = dm.to_arrays(mkt)
    t1 = time.time()
    shares = dm.banner_shares(theta, a)
    print(f"\nshares at prior parameters, delta = 0 [uncalibrated] ({time.time() - t1:.1f}s):")
    out = shares.rename("share_prior_delta0").round(4).to_frame()
    out.to_csv(OUT / "1_1_prior_shares.csv")
    print(out.to_string())
    assert abs(shares.sum() - 1) < 1e-9

    figures(mkt, only_10)


def nearest_by_group(mkt):
    """(L, groups) drive minutes from each location to the nearest store of each group (inf if none in reach)."""
    out = {}
    for name, owners in GROUPS.items():
        sel = np.isin(mkt.store_group[mkt.pair_store], owners)
        m = np.full(mkt.L, np.inf)
        np.minimum.at(m, mkt.pair_loc[sel], mkt.pair_time[sel])
        out[name] = m
    return out


def figures(mkt, only_10):
    viz.setup()
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cantons = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)
    loc = pd.read_parquet(PROCESSED / "locations.parquet", columns=["loc_id", "x", "y"]).set_index("loc_id")
    xy = loc.loc[mkt.loc_id]
    s = pd.read_parquet(PROCESSED / "stores.parquet", columns=["store_id", "lon", "lat", "group"])
    sx, sy = pyproj.Transformer.from_crs(4326, 2056, always_xy=True).transform(s.lon.to_numpy(), s.lat.to_numpy())

    # 1. stores by group
    fig, axes = plt_subplots(1, 3, (13, 3.6))
    for ax, (name, owners) in zip(axes, GROUPS.items()):
        viz.map_axes(ax, cantons)
        sel = s.group.isin(owners).to_numpy()
        ax.scatter(sx[sel], sy[sel], s=2.5, color=viz.GROUP_COLOR[name], linewidths=0, zorder=2)
        ax.set_title(f"{name}: {sel.sum():,} stores", fontsize=10.5)
    fig.suptitle("Migros and Coop groups blanket the country; Aldi and Lidl cover the Plateau",
                 x=0.01, ha="left", fontweight="bold")
    viz.source(axes[0], "Migros group = Migros, Partner, Migrolino, Denner. Coop group = Coop, Coop Pronto. " + SRC)
    viz.save(fig, 1, "1_1", "store_map")

    # 2. where only the merging groups are in reach
    fig, ax = plt_subplots(1, 1, (8.5, 5.6))
    viz.map_axes(ax, cantons)
    ax.scatter(xy.x[~only_10], xy.y[~only_10], s=0.4, color=viz.MUTED, linewidths=0, zorder=2,
               label="other locations")
    ax.scatter(xy.x[only_10], xy.y[only_10], s=3, color=viz.ORANGE, linewidths=0, zorder=3,
               label="both groups within 10 min, no rival (Aldi, Lidl, Volg, Landi, Spar, Manor)")
    share = mkt.hh[only_10].sum() / mkt.hh.sum()
    ax.set_title(f"For {share:.1%} of households the merger is two-to-one within 10 minutes")
    ax.legend(loc="lower left", markerscale=4, fontsize=8.5)
    viz.source(ax, "Each dot is a shopper location (H3 cell, ~0.7 km2). " + SRC)
    viz.save(fig, 1, "1_1", "overlap_map")

    # 3. reach: share of households within x minutes of each group's nearest store
    near = nearest_by_group(mkt)
    grid = np.linspace(0, 25, 251)
    fig, ax = plt_subplots(1, 1, (7.5, 4.2))
    at10 = {}
    for name, m in near.items():
        cdf = np.array([(mkt.hh * (m <= g)).sum() / mkt.hh.sum() for g in grid])
        ax.plot(grid, cdf, color=viz.GROUP_COLOR[name], label=name)
        at10[name] = cdf[np.searchsorted(grid, 10)]
    ax.axvline(10, color=viz.INK_2, linewidth=0.8, linestyle=":")
    ax.text(10.4, 0.42, "within 10 min\n" + "\n".join(f"{k}: {v:.0%}" for k, v in at10.items()),
            fontsize=9, color=viz.INK, va="top")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("drive minutes to the nearest store of the group")
    ax.set_ylabel("share of households")
    ax.set_title("Almost every household is minutes from both merging groups")
    ax.legend(loc="lower right")
    viz.source(ax, "Free-flow drive times. " + SRC)
    viz.save(fig, 1, "1_1", "reach")

    # 4. income: median net income per taxpayer by municipality (ESTV 2023) -> price sensitivity alpha
    from matplotlib.colors import BoundaryNorm, ListedColormap
    muni = pd.read_parquet(PROCESSED / "municipalities.parquet")
    mun = adm[adm.admin_level.astype(str) == "8"].copy()
    mun["bfs"] = pd.to_numeric(mun.bfs, errors="coerce")
    mun = mun.to_crs(2056).merge(muni[["median_net_income"]], left_on="bfs", right_index=True, how="left")
    bins = [0, 45_000, 50_000, 55_000, 60_000, 65_000, 70_000, 80_000, 1e9]
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
    cmap, norm = ListedColormap(ramp), BoundaryNorm(bins, len(ramp))
    fig, ax = plt_subplots(1, 1, (8.5, 5.8))
    viz.map_axes(ax, cantons)
    mun[mun.median_net_income.isna()].plot(ax=ax, color=viz.GRID, linewidth=0, zorder=1)
    mun.dropna(subset=["median_net_income"]).plot(ax=ax, column="median_net_income", cmap=cmap, norm=norm,
                                                  linewidth=0.1, edgecolor=viz.SURFACE, zorder=2)
    cantons.boundary.plot(ax=ax, color=viz.SURFACE, linewidth=0.8, zorder=3)
    nat = float(np.average(muni.median_net_income.dropna(), weights=muni.loc[muni.median_net_income.notna(), "taxpayers"]))
    lo = mun.dropna(subset=["median_net_income"]).nsmallest(1, "median_net_income").iloc[0]
    hi = mun.dropna(subset=["median_net_income"]).nlargest(1, "median_net_income").iloc[0]
    ax.set_title("Incomes are highest around Zurich, Zug and Lake Geneva's north shore")
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=40,
                      ticks=bins[1:-1], spacing="uniform")
    cb.ax.set_xticklabels([f"{b // 1000:.0f}k" for b in bins[1:-1]], fontsize=8)
    cb.set_label("median net income per taxpayer, CHF / year (municipality)", fontsize=8.5, color=viz.INK_2)
    cb.outline.set_visible(False)
    viz.source(ax, f"Taxpayer-weighted average of municipal medians: CHF {nat:,.0f}. Range: {lo['name']} "
                   f"CHF {lo.median_net_income:,.0f} to {hi['name']} CHF {hi.median_net_income:,.0f}. "
                   "Grey = no ESTV row (recent mergers). Source: ESTV direct federal tax 2023; OSM boundaries.")
    viz.save(fig, 1, "1_1", "income_map")


def plt_subplots(r, c, size):
    import matplotlib.pyplot as plt
    return plt.subplots(r, c, figsize=size)


if __name__ == "__main__":
    main()
