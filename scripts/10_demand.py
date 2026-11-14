"""Phase 2 step 1: build the spatial nested-logit demand system on real Swiss geography.

This step assembles the choice sets (every location's stores within 25 minutes, plus abroad) and the
income types. It also evaluates demand at the prior parameters, to check the machinery at full
scale. Banner qualities delta are still 0 here; step 2 (11_calibrate.py) fits them to market shares.

Run:  conda activate keystone && python scripts/10_demand.py
Outputs:
  data/processed/market.npz                  choice-set arrays (input to every later Pillar 1 step)
  outputs/tables/demand_market_summary.csv   size of the market and the choice sets
  outputs/tables/demand_overlap.csv          households reaching Migros-group and Coop-group stores
  outputs/tables/demand_prior_shares.csv     national shares at prior parameters, delta = 0 [uncalibrated]
"""
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mxc.structural import demand as dm  # noqa: E402

PROCESSED, TABLES = ROOT / "data/processed", ROOT / "outputs/tables"


def main():
    t0 = time.time()
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
    summary.to_csv(TABLES / "demand_market_summary.csv", index=False)
    print(summary.to_string(index=False))

    # --- overlap: share of households with both groups' stores nearby (where the merger bites)
    rows = []
    for m in [5, 10, 15]:
        sel = mkt.pair_time <= m
        grp = mkt.store_group[mkt.pair_store[sel]]
        has = lambda g: np.bincount(mkt.pair_loc[sel][grp == g], minlength=mkt.L) > 0  # noqa: E731
        both = has("Migros") & has("Coop")
        only = (has("Migros") | has("Coop")) & ~(has("Aldi") | has("Lidl") | has("fenaco") | has("Spar") | has("Manor"))
        rows.append({"within_min": m, "households_with_migros_group_and_coop_group": hh[both].sum() / hh.sum(),
                     "households_with_only_migros_or_coop_group": hh[only].sum() / hh.sum()})
    overlap = pd.DataFrame(rows).round(3)
    overlap.to_csv(TABLES / "demand_overlap.csv", index=False)
    print("\n", overlap.to_string(index=False))

    # --- full-scale evaluation at prior parameters (delta = 0): checks the machinery, not a result
    theta = dm.prior_params()
    a = dm.to_arrays(mkt)
    t1 = time.time()
    shares = dm.banner_shares(theta, a)
    print(f"\nshares at prior parameters, delta = 0 [uncalibrated] ({time.time() - t1:.1f}s):")
    out = shares.rename("share_prior_delta0").round(4).to_frame()
    out.to_csv(TABLES / "demand_prior_shares.csv")
    print(out.to_string())
    assert abs(shares.sum() - 1) < 1e-9


if __name__ == "__main__":
    main()
