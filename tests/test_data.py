import numpy as np
import pytest

from mxc.data import demographics as dm
from mxc.data import osm, roads


@pytest.mark.parametrize("brand,name,expected", [
    ("Coop", "Coop Pronto", "Coop Pronto"),          # brand + name read together
    ("Migros", "Migros", "Migros"),
    (None, "VOI Migros-Partenaire", "Migros Partner"),
    ("Denner", "Denner Satellit", "Denner"),
    ("ALDI SUISSE", None, "Aldi"),
    ("Coop", "Coop Vitality", None),                  # pharmacy, not grocery
    (None, "Cooperativa agricola", None),             # 'coop' must be a whole word
    (np.nan, "Migrolino", "Migrolino"),               # NaN tags are tolerated
])
def test_classify_banner(brand, name, expected):
    assert osm.classify_banner(brand, name, None) == expected


def test_grouped_median_interpolates_within_class():
    counts = np.zeros(10)
    counts[4] = 10            # all taxpayers in 50,001-75,000
    assert dm._grouped_median(counts) == pytest.approx(50_001 + 0.5 * (75_000 - 50_001))


def tiny_graph():
    # 0 --1km motorway-- 1 --1km residential-- 2 ; 0 --3km residential-- 2
    g = {"x": np.array([0.0, 1000.0, 2000.0]), "y": np.zeros(3), "snappable": np.array([True, False, True]),
         "e_from": np.array([0, 1, 0]), "e_to": np.array([1, 2, 2]),
         "length_m": np.array([1000.0, 1000.0, 3000.0])}
    speeds = np.array([100.0, 25.0, 25.0])
    g["time_min"] = g["length_m"] / (speeds * 1000 / 60)
    return g


def test_shortest_time_prefers_motorway_route():
    g = tiny_graph()
    t = roads.times_from_sources(roads.csr(g), np.array([0]), np.array([2]), limit=np.inf)[0, 0]
    assert t == pytest.approx(0.6 + 2.4)       # 1 km @100 + 1 km @25, beats 3 km @25 = 7.2 min


def test_snap_skips_motorway_nodes():
    node, d = roads.snap(tiny_graph(), np.array([1000.0]), np.array([10.0]))
    assert node[0] in (0, 2)                   # node 1 is motorway-only


def test_duplicate_edges_keep_fastest():
    g = tiny_graph()
    g["e_from"], g["e_to"] = np.r_[g["e_from"], 1], np.r_[g["e_to"], 0]   # same pair, reversed, slower
    g["time_min"] = np.r_[g["time_min"], 50.0]
    assert roads.csr(g)[0, 1] == pytest.approx(0.6)


def test_limit_returns_inf_beyond_cutoff():
    t = roads.times_from_sources(roads.csr(tiny_graph()), np.array([0]), np.array([2]), limit=1.0)
    assert np.isinf(t[0, 0])
