"""Drivable road network from the OSM extract -> drive times with scipy shortest paths.

A light, Docker-free substitute for OSRM: free-flow speeds by road class (no congestion).
Data label: road geometry [real]; speeds [assumed], see SPEED_KMH.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import osmium
import pyproj
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

# Free-flow car speeds (km/h) by OSM highway class. Assumption; varied in sensitivity analysis.
SPEED_KMH = {
    "motorway": 100, "motorway_link": 60, "trunk": 80, "trunk_link": 50,
    "primary": 60, "primary_link": 40, "secondary": 50, "secondary_link": 40,
    "tertiary": 40, "tertiary_link": 30, "unclassified": 30, "residential": 25,
    "living_street": 10, "road": 25,
}
FAST = {"motorway", "motorway_link", "trunk", "trunk_link"}   # not used for snapping points
NO_ACCESS = {"no", "private"}

_to_ch = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)


def build_graph(pbf: Path) -> dict:
    """Read drivable ways; return node coordinates (LV95, m) and undirected edges with length and time."""
    node_index: dict[int, int] = {}
    lon, lat, snappable = [], [], []
    e_from, e_to, e_speed = [], [], []

    hw_filter = osmium.filter.KeyFilter("highway")
    fp = (osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY)
          .with_locations()
          .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
          .with_filter(hw_filter))
    for w in fp:
        hw = w.tags.get("highway")
        if hw not in SPEED_KMH or w.tags.get("access") in NO_ACCESS or w.tags.get("area") == "yes":
            continue
        speed = SPEED_KMH[hw]
        prev = None
        for n in w.nodes:
            if not n.location.valid():
                prev = None
                continue
            i = node_index.get(n.ref)
            if i is None:
                i = len(lon)
                node_index[n.ref] = i
                lon.append(n.location.lon)
                lat.append(n.location.lat)
                snappable.append(hw not in FAST)
            elif hw not in FAST:
                snappable[i] = True
            if prev is not None:
                e_from.append(prev)
                e_to.append(i)
                e_speed.append(speed)
            prev = i

    x, y = _to_ch.transform(np.asarray(lon), np.asarray(lat))
    e_from, e_to = np.asarray(e_from, np.int64), np.asarray(e_to, np.int64)
    length_m = np.hypot(x[e_from] - x[e_to], y[e_from] - y[e_to])
    time_min = length_m / (np.asarray(e_speed, float) * 1000 / 60)
    return {"x": x, "y": y, "snappable": np.asarray(snappable), "e_from": e_from,
            "e_to": e_to, "length_m": length_m, "time_min": time_min}


def save_graph(g: dict, path: Path) -> None:
    np.savez_compressed(path, **g)


def load_graph(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def csr(g: dict, weight: str = "time_min") -> sparse.csr_matrix:
    """Sparse adjacency. Stored one-way (i<j); use with directed=False (oneway restrictions ignored)."""
    n = len(g["x"])
    w = np.maximum(g[weight], 1e-6)  # zero-length edges would vanish from a sparse matrix
    a, b = np.minimum(g["e_from"], g["e_to"]), np.maximum(g["e_from"], g["e_to"])
    # Same node pair on two ways: keep the faster edge (COO would otherwise sum duplicates).
    order = np.lexsort((w, b, a))
    a, b, w = a[order], b[order], w[order]
    first = np.r_[True, (a[1:] != a[:-1]) | (b[1:] != b[:-1])]
    return sparse.csr_matrix((w[first], (a[first], b[first])), shape=(n, n))


def largest_component_mask(m: sparse.csr_matrix) -> np.ndarray:
    from scipy.sparse.csgraph import connected_components
    _, lab = connected_components(m, directed=False)
    return lab == np.bincount(lab).argmax()


def snap(g: dict, xs: np.ndarray, ys: np.ndarray, mask: np.ndarray | None = None):
    """Nearest snappable graph node for each point. Returns (node index, snap distance m)."""
    ok = g["snappable"] if mask is None else (g["snappable"] & mask)
    idx = np.flatnonzero(ok)
    tree = cKDTree(np.c_[g["x"][idx], g["y"][idx]])
    d, j = tree.query(np.c_[xs, ys])
    return idx[j], d


def times_from_sources(m: sparse.csr_matrix, sources: np.ndarray, targets: np.ndarray,
                       limit: float, batch: int = 8) -> np.ndarray:
    """Drive time (min) from each source node to each target node; inf beyond `limit` minutes.

    Returns a dense (n_sources, n_targets) array. Batched to bound memory.
    """
    out = np.full((len(sources), len(targets)), np.inf, dtype=np.float32)
    for s in range(0, len(sources), batch):
        d = dijkstra(m, directed=False, indices=sources[s:s + batch], limit=limit)
        out[s:s + batch] = d[:, targets]
    return out


def nearest_source_time(m: sparse.csr_matrix, sources: np.ndarray, limit: float = np.inf) -> np.ndarray:
    """Multi-source Dijkstra: time from every node to its nearest source node."""
    return dijkstra(m, directed=False, indices=sources, limit=limit, min_only=True)
