"""OpenStreetMap extraction: grocery stores and administrative boundaries.

Reads the Geofabrik Switzerland extract (.osm.pbf) locally with pyosmium,
so no Overpass server, Docker or osmium-tool is needed.

Data label: [real] (OpenStreetMap contributors, ODbL).
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import osmium
import pandas as pd
from shapely import wkb

# Swiss national grid (LV95), metres. Used for all distance work.
CRS_CH = "EPSG:2056"
CRS_WGS = "EPSG:4326"

# Shop types that can be a grocery outlet. Brand matching is applied on top.
GROCERY_SHOPS = {"supermarket", "convenience", "department_store", "general", "food", "deli"}

KEEP_TAGS = ["name", "brand", "brand:wikidata", "operator", "shop",
             "addr:postcode", "addr:city", "addr:street", "opening_hours"]

# Non-grocery formats of the same groups, excluded even if tagged as shops above.
NON_GROCERY = re.compile(
    r"vitality|bau\s*\+?\s*hobby|do\s*it|sportxx|melectronics|micasa|interdiscount|"
    r"restaurant|take\s*it|fust|import\s*parfumerie|ochsner|bike\s*world|obi\b|"
    r"tankstelle\s*only", re.I)

# Order matters: most specific first.
BANNER_RULES: list[tuple[str, re.Pattern]] = [
    ("Migrolino",       re.compile(r"migrolino", re.I)),
    ("Migros Partner",  re.compile(r"migros[\s-]*partner|\bvoi\b", re.I)),
    ("Coop Pronto",     re.compile(r"coop\s*pronto|\bpronto\b", re.I)),
    ("Denner",          re.compile(r"\bdenner\b", re.I)),
    ("Aldi",            re.compile(r"\baldi\b", re.I)),
    ("Lidl",            re.compile(r"\blidl\b", re.I)),
    ("Volg",            re.compile(r"\bvolg\b", re.I)),
    ("Landi",           re.compile(r"\blandi\b", re.I)),
    ("Spar",            re.compile(r"\bspar\b", re.I)),
    ("Manor",           re.compile(r"\bmanor\b", re.I)),
    ("Migros",          re.compile(r"\bmigros\b", re.I)),
    ("Coop",            re.compile(r"\bcoop\b", re.I)),
]

# Ownership groups used later in the ownership matrix.
GROUP = {
    "Migros": "Migros", "Migros Partner": "Migros", "Migrolino": "Migros", "Denner": "Migros",
    "Coop": "Coop", "Coop Pronto": "Coop",
    "Aldi": "Aldi", "Lidl": "Lidl",
    "Volg": "fenaco", "Landi": "fenaco",
    "Spar": "Spar", "Manor": "Manor",
}


def classify_banner(brand: str | None, name: str | None, operator: str | None) -> str | None:
    """Banner from brand + name + operator read together, most specific rule first.

    Reading them together matters: brand="Coop", name="Coop Pronto" must give Coop Pronto.
    None = not a tracked chain.
    """
    text = " | ".join(t for t in (brand, name, operator) if isinstance(t, str) and t)  # NaN-safe
    if not text or NON_GROCERY.search(text):
        return None
    for banner, pat in BANNER_RULES:
        if pat.search(text):
            return banner
    return None


def extract_shops(pbf: Path) -> pd.DataFrame:
    """All grocery-type shops (nodes + building areas) with a representative point."""
    wkbf = osmium.geom.WKBFactory()
    rows = []
    shop_filter = osmium.filter.KeyFilter("shop")
    fp = (osmium.FileProcessor(str(pbf))
          .with_locations()
          .with_areas(shop_filter)
          .with_filter(shop_filter))
    for obj in fp:
        shop = obj.tags.get("shop")
        if shop not in GROCERY_SHOPS:
            continue
        if obj.is_node():
            lon, lat, kind, oid = obj.location.lon, obj.location.lat, "node", obj.id
        elif obj.is_area():
            try:
                geom = wkb.loads(wkbf.create_multipolygon(obj), hex=True)
            except Exception:  # broken geometry in OSM
                continue
            pt = geom.representative_point()
            lon, lat, kind, oid = pt.x, pt.y, "area", obj.orig_id()
        else:
            continue  # plain ways/relations: covered by the area version
        rec = {k: obj.tags.get(k) for k in KEEP_TAGS}
        rec.update(osm_type=kind, osm_id=oid, lon=lon, lat=lat)
        rows.append(rec)
    return pd.DataFrame(rows)


def extract_admin(pbf: Path, levels=("2", "4", "8")) -> gpd.GeoDataFrame:
    """Administrative boundaries: 2 = country, 4 = canton, 8 = municipality."""
    wkbf = osmium.geom.WKBFactory()
    rows = []
    afilter = osmium.filter.TagFilter(("boundary", "administrative"))
    fp = (osmium.FileProcessor(str(pbf))
          .with_areas(afilter)
          .with_filter(osmium.filter.EntityFilter(osmium.osm.AREA))
          .with_filter(afilter))
    for obj in fp:
        lvl = obj.tags.get("admin_level")
        if lvl not in levels:
            continue
        try:
            geom = wkb.loads(wkbf.create_multipolygon(obj), hex=True)
        except Exception:
            continue
        rows.append({
            "admin_level": int(lvl),
            "name": obj.tags.get("name"),
            "ref": obj.tags.get("ref"),                       # canton code, e.g. ZH
            "iso": obj.tags.get("ISO3166-2"),                 # e.g. CH-ZH
            "bfs": obj.tags.get("swisstopo:BFS_NUMMER"),      # municipality number
            "osm_id": obj.orig_id(),
            "geometry": geom,
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_WGS)


def clean_stores(shops: pd.DataFrame, dedupe_m: float = 40.0) -> gpd.GeoDataFrame:
    """Classify banners, drop untracked shops, remove node/area duplicates of the same store."""
    df = shops.copy()
    df["banner"] = [classify_banner(b, n, o) for b, n, o in zip(df["brand"], df["name"], df["operator"])]
    df = df[df["banner"].notna()].copy()
    df["group"] = df["banner"].map(GROUP)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=CRS_WGS).to_crs(CRS_CH)

    # Same banner within dedupe_m metres = one store mapped twice (node + building). Keep the area.
    gdf = gdf.sort_values("osm_type").reset_index(drop=True)          # 'area' sorts before 'node'
    keep = pd.Series(True, index=gdf.index)
    for banner, sub in gdf.groupby("banner"):
        near = gpd.sjoin_nearest(sub, sub, how="inner", max_distance=dedupe_m,
                                 distance_col="d", exclusive=True)
        drop = near[near.index > near["index_right"]].index           # drop the later one of each pair
        keep.loc[drop] = False
    out = gdf[keep].copy()
    out["store_id"] = out["osm_type"].str[0] + out["osm_id"].astype(str)
    return out
