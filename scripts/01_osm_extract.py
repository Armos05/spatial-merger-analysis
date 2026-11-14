"""Step 1 of the data pipeline: stores + boundaries from the OSM extract, counts by canton.

Run:  conda activate keystone && python scripts/01_osm_extract.py
Outputs:
  data/interim/osm_shops_raw.parquet        all grocery-type shops (before brand cleaning)
  data/interim/admin_boundaries.gpkg        country / cantons / municipalities
  data/processed/stores.parquet             tracked banners, deduplicated, with canton + municipality
  outputs/tables/store_counts_by_canton.csv
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd  # noqa: E402
import pandas as pd      # noqa: E402

from mxc.data import osm  # noqa: E402

PBF = ROOT / "data/raw/osm/switzerland-latest.osm.pbf"
INTERIM, PROCESSED, TABLES = ROOT / "data/interim", ROOT / "data/processed", ROOT / "outputs/tables"
for d in (INTERIM, PROCESSED, TABLES):
    d.mkdir(parents=True, exist_ok=True)

t0 = time.time()
shops_path = INTERIM / "osm_shops_raw.parquet"
if shops_path.exists():
    shops = pd.read_parquet(shops_path)
else:
    shops = osm.extract_shops(PBF)
    shops.to_parquet(shops_path)
print(f"shops extracted: {len(shops):,}  ({time.time()-t0:.0f}s)")

admin_path = INTERIM / "admin_boundaries.gpkg"
if admin_path.exists():
    admin = gpd.read_file(admin_path)
else:
    admin = osm.extract_admin(PBF)
    admin.to_file(admin_path, driver="GPKG")
print(f"admin areas: {admin.admin_level.value_counts().to_dict()}  ({time.time()-t0:.0f}s)")

# Swiss cantons / municipalities only (the extract also clips neighbouring countries' areas).
cantons = admin[(admin.admin_level == 4) & admin.iso.fillna("").str.startswith("CH-")].to_crs(osm.CRS_CH)
cantons = cantons.assign(canton=cantons.iso.str[3:])[["canton", "name", "geometry"]]
munis = admin[admin.admin_level == 8].to_crs(osm.CRS_CH)
munis = munis[munis.bfs.notna()][["bfs", "name", "geometry"]].rename(columns={"name": "municipality"})
print(f"Swiss cantons: {len(cantons)}   municipalities with BFS number: {len(munis)}")

stores = osm.clean_stores(shops)
stores = gpd.sjoin(stores, cantons[["canton", "geometry"]], how="left", predicate="within").drop(columns="index_right")
stores = gpd.sjoin(stores, munis, how="left", predicate="within").drop(columns="index_right")
stores = stores[stores.canton.notna()]                    # drop stores just outside Switzerland
stores.to_parquet(PROCESSED / "stores.parquet")

order = ["Migros", "Migros Partner", "Migrolino", "Denner", "Coop", "Coop Pronto",
         "Aldi", "Lidl", "Volg", "Landi", "Spar", "Manor"]
counts = (stores.groupby(["canton", "banner"]).size().unstack(fill_value=0)
          .reindex(columns=order, fill_value=0))
counts.loc["CH total"] = counts.sum()
counts.to_csv(TABLES / "store_counts_by_canton.csv")
pd.set_option("display.width", 200)
print(counts)
print(f"done in {time.time()-t0:.0f}s")
