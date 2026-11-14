"""Regenerate every dataset, table and figure, in order. One command:  python scripts/run_all.py"""
import runpy
import sys
from pathlib import Path

STEPS = [
    "01_osm_extract.py",   # stores, boundaries, store counts by canton
    "02_geography.py",     # shopper locations, income, drive times, border times, detour check
    # later phases append here: 10_demand.py, 20_screening.py, 30_simulation.py, ...
]

here = Path(__file__).parent
for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    sys.argv = [str(here / step)]
    runpy.run_path(str(here / step), run_name="__main__")
