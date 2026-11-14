"""Step 2: shopper locations, income, drive times to stores, drive time to the border.

Shopper locations = populated hectares (STATPOP 2024) grouped into H3 resolution-8 cells
(~0.74 km2), split by municipality so each location has exactly one municipality's income.

Run:  conda activate keystone && python scripts/02_geography.py
Outputs:
  data/processed/locations.parquet       loc_id, h3, bfs, canton, pop, households, x, y,
                                         mean/median net income, border_min
  data/processed/municipalities.parquet  income distribution by municipality (ESTV 2023)
  data/processed/drive_times.parquet     loc_id, store_id, drive_min, road_km, straight_km (<= LIMIT_MIN)
  data/processed/border_crossings.parquet
  outputs/tables/detour_factor.csv       road km / straight km, by canton
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd  # noqa: E402
import h3  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyproj  # noqa: E402
import shapely  # noqa: E402

from mxc.data import demographics as dm  # noqa: E402
from mxc.data import roads  # noqa: E402

LIMIT_MIN = 25.0      # choice-set cutoff (drive minutes); generous, distance disutility does the work
H3_RES = 8
RNG = np.random.default_rng(20261114)

RAW, INTERIM, PROCESSED, TABLES = (ROOT / "data/raw", ROOT / "data/interim",
                                   ROOT / "data/processed", ROOT / "outputs/tables")
t0 = time.time()
log = lambda msg: print(f"[{time.time()-t0:5.0f}s] {msg}")  # noqa: E731

# ---------------------------------------------------------------- boundaries
admin = gpd.read_file(INTERIM / "admin_boundaries.gpkg").to_crs("EPSG:2056")
ch = admin[admin.admin_level == 2].geometry.union_all()
cantons = admin[(admin.admin_level == 4) & admin.iso.fillna("").str.startswith("CH-")]
cantons = cantons.assign(canton=cantons.iso.str[3:])[["canton", "geometry"]]
munis = admin[(admin.admin_level == 8) & admin.bfs.notna()][["bfs", "name", "geometry"]]
munis["bfs"] = munis.bfs.astype(int)

# ---------------------------------------------------------------- income (ESTV 2023)
income = dm.read_estv_income(RAW / "estv/statistik-dbst-np-gemeinden-2023-500.xlsx")
income.to_parquet(PROCESSED / "municipalities.parquet")
log(f"ESTV municipalities: {len(income)}")

# ---------------------------------------------------------------- hectares -> locations
hect = dm.read_statpop_hectares(RAW / "bfs/statpop2024/STATPOP2024.csv")
hg = gpd.GeoDataFrame(hect, geometry=gpd.points_from_xy(hect.x, hect.y), crs="EPSG:2056")
hg = gpd.sjoin(hg, munis[["bfs", "geometry"]], how="left", predicate="within").drop(columns="index_right")
miss = hg.bfs.isna()
if miss.any():  # hectare centres on lake shores / boundary slivers: nearest municipality
    near = gpd.sjoin_nearest(hg[miss].drop(columns="bfs"), munis[["bfs", "geometry"]], how="left")
    hg.loc[miss, "bfs"] = near.groupby(level=0).bfs.first()
hg["bfs"] = hg.bfs.astype(int)
to_wgs = pyproj.Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
lon, lat = to_wgs.transform(hg.x.to_numpy(), hg.y.to_numpy())
hg["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(lat, lon)]
log(f"hectares: {len(hg):,} (pop {hg['pop'].sum():,}); assigned to nearest municipality: {int(miss.sum())}")

hg["px"], hg["py"] = hg.x * hg["pop"], hg.y * hg["pop"]
loc = hg.groupby(["h3", "bfs"], as_index=False).agg(
    pop=("pop", "sum"), households=("households", "sum"), px=("px", "sum"), py=("py", "sum"),
    n_hect=("pop", "size"), x_mean=("x", "mean"), y_mean=("y", "mean"))
w = loc["pop"].replace(0, np.nan)
loc["x"] = (loc.px / w).fillna(loc.x_mean)      # population-weighted centroid
loc["y"] = (loc.py / w).fillna(loc.y_mean)
loc = loc.drop(columns=["px", "py", "x_mean", "y_mean"])
loc = loc[loc["pop"] > 0].reset_index(drop=True)
loc.insert(0, "loc_id", np.arange(len(loc)))

lg = gpd.GeoDataFrame(loc, geometry=gpd.points_from_xy(loc.x, loc.y), crs="EPSG:2056")
lg = gpd.sjoin_nearest(lg, cantons, how="left").drop(columns="index_right")
lg = lg[~lg.loc_id.duplicated()]
loc["canton"] = lg.set_index("loc_id").canton.reindex(loc.loc_id).to_numpy()

loc = loc.merge(income[["mean_net_income", "median_net_income"]], left_on="bfs", right_index=True, how="left")
no_inc = loc.mean_net_income.isna()
canton_mean = loc[~no_inc].groupby("canton").apply(
    lambda d: np.average(d.mean_net_income, weights=d["pop"]), include_groups=False)
loc["income_imputed"] = no_inc
loc.loc[no_inc, "mean_net_income"] = loc.loc[no_inc, "canton"].map(canton_mean)
loc.loc[no_inc, "median_net_income"] = np.nan
log(f"locations: {len(loc):,}; population without own ESTV income (canton mean used): "
    f"{loc.loc[no_inc, 'pop'].sum():,} in {loc.loc[no_inc, 'bfs'].nunique()} municipalities")

# ---------------------------------------------------------------- road graph
gpath = INTERIM / "road_graph.npz"
if gpath.exists():
    g = roads.load_graph(gpath)
else:
    g = roads.build_graph(RAW / "osm/switzerland-latest.osm.pbf")
    roads.save_graph(g, gpath)
m_time, m_len = roads.csr(g, "time_min"), roads.csr(g, "length_m")
main = roads.largest_component_mask(m_time)
log(f"road graph: {len(g['x']):,} nodes; largest component {main.mean():.1%}")

stores = gpd.read_parquet(PROCESSED / "stores.parquet")
loc["node"], loc["snap_m"] = roads.snap(g, loc.x.to_numpy(), loc.y.to_numpy(), main)
stores["node"], stores["snap_m"] = roads.snap(g, stores.geometry.x.to_numpy(), stores.geometry.y.to_numpy(), main)
log(f"snap distance (m) median / p95: locations {loc.snap_m.median():.0f} / {loc.snap_m.quantile(.95):.0f}; "
    f"stores {stores.snap_m.median():.0f} / {stores.snap_m.quantile(.95):.0f}")

# ---------------------------------------------------------------- border crossings
# A crossing = road edge with one end inside Switzerland and one outside, excluding
# Liechtenstein and Buesingen (both inside the Swiss customs area: shopping there is not "abroad").
shapely.prepare(ch)
inside = shapely.contains_xy(ch, g["x"], g["y"])
cross = inside[g["e_from"]] != inside[g["e_to"]]
out_node = np.where(inside[g["e_from"]], g["e_to"], g["e_from"])[cross]
in_node = np.where(inside[g["e_from"]], g["e_from"], g["e_to"])[cross]
olon, olat = to_wgs.transform(g["x"][out_node], g["y"][out_node])
li = (olon > 9.47) & (olon < 9.64) & (olat > 47.04) & (olat < 47.28)
buesingen = (olon > 8.65) & (olon < 8.73) & (olat > 47.68) & (olat < 47.72)
keep = ~li & ~buesingen & main[in_node]
xing = np.unique(in_node[keep])
pd.DataFrame({"node": xing, "x": g["x"][xing], "y": g["y"][xing]}).to_parquet(PROCESSED / "border_crossings.parquet")
border_t = roads.nearest_source_time(m_time, xing)
loc["border_min"] = border_t[loc.node.to_numpy()]
log(f"border crossing points: {len(xing)}; median drive time to border {loc.border_min.median():.1f} min")

# ---------------------------------------------------------------- drive times stores -> locations
# Dijkstra from each store (symmetric graph), keep locations within LIMIT_MIN.
loc_nodes = loc.node.to_numpy()
rows = []
st_nodes = stores.node.to_numpy()
for s in range(0, len(stores), 16):
    D = roads.times_from_sources(m_time, st_nodes[s:s + 16], loc_nodes, limit=LIMIT_MIN, batch=16)
    si, li_ = np.nonzero(np.isfinite(D))
    rows.append(pd.DataFrame({"store_idx": si + s, "loc_id": li_, "drive_min": D[si, li_]}))
dt = pd.concat(rows, ignore_index=True)
dt["store_id"] = stores.store_id.to_numpy()[dt.store_idx]
sx, sy = stores.geometry.x.to_numpy()[dt.store_idx], stores.geometry.y.to_numpy()[dt.store_idx]
dt["straight_km"] = np.hypot(sx - loc.x.to_numpy()[dt.loc_id], sy - loc.y.to_numpy()[dt.loc_id]) / 1000
dt = dt.drop(columns="store_idx")
dt["drive_min"] = dt.drive_min.astype("float32")
dt["straight_km"] = dt.straight_km.astype("float32")
log(f"drive-time pairs within {LIMIT_MIN:.0f} min: {len(dt):,}")

# ---------------------------------------------------------------- detour factor (validation)
# Road km / straight km on a random sample of 300 locations x their stores (2-25 km apart).
samp = RNG.choice(len(loc), 300, replace=False, p=(loc["pop"] / loc["pop"].sum()).to_numpy())
st_idx = {sid: i for i, sid in enumerate(stores.store_id)}
road_km = roads.times_from_sources(m_len, loc_nodes[samp], st_nodes, limit=40_000, batch=16) / 1000
recs = []
for k, lid in enumerate(samp):
    pairs = dt[dt.loc_id == lid]
    j = pairs.store_id.map(st_idx).to_numpy()
    recs.append(pd.DataFrame({"loc_id": lid, "store_id": pairs.store_id.to_numpy(),
                              "straight_km": pairs.straight_km.to_numpy(), "road_km": road_km[k, j]}))
det = pd.concat(recs)
det = det[(det.straight_km >= 2) & np.isfinite(det.road_km)]
det["detour"] = det.road_km / det.straight_km
det = det.merge(loc[["loc_id", "canton"]], on="loc_id")
tab = det.groupby("canton").detour.describe(percentiles=[.1, .5, .9])[["count", "10%", "50%", "mean", "90%"]]
tab.loc["CH"] = det.detour.describe(percentiles=[.1, .5, .9])[["count", "10%", "50%", "mean", "90%"]]
tab.round(3).to_csv(TABLES / "detour_factor.csv")
log(f"detour factor CH: median {det.detour.median():.2f}, mean {det.detour.mean():.2f}, "
    f"p10-p90 {det.detour.quantile(.1):.2f}-{det.detour.quantile(.9):.2f} (n={len(det):,} pairs)")

# road km for every pair is not stored (time is what enters utility); detour stats above suffice.
loc.to_parquet(PROCESSED / "locations.parquet")
dt.to_parquet(PROCESSED / "drive_times.parquet")
stores.to_parquet(PROCESSED / "stores.parquet")          # now with graph node + snap distance
log("done")
