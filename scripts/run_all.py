"""Regenerate every dataset, table and figure, in order. One command:  python scripts/run_all.py"""
import runpy
import sys
from pathlib import Path

STEPS = [
    "01_osm_extract.py",   # stores, boundaries, store counts by canton
    "02_geography.py",     # shopper locations, income, drive times, border times, detour check
    "10_demand.py",        # Pillar 1 step 1: choice sets, income types, demand at priors
    "11_calibrate.py",     # Pillar 1 step 2: delta fitted to 2025 sales shares + card-data abroad share
    "12_diversion.py",     # Pillar 1 step 3: store Jacobian, diversion (store, player, canton), elasticities
    "13_supply.py",        # Pillar 1 step 4: cooperative conduct λμ and marginal costs from 2024 accounts
    "14_screening.py",     # Pillar 1 step 5: national + store GUPPI/CMCR, Koh no-price check
    "15_simulation.py",    # Pillar 1 step 6: post-merger equilibrium, ΔCS by location / canton / income
    "16_scenarios.py",     # Pillar 1 step 7: efficiencies 0-10% × λμ survives / dilutes / disappears
    "17_outcomes.py",      # Pillar 1 step 8: outcome table A-D, G + divestiture remedy curve
    "20_twin.py",          # Pillar 2 step 1: digital twin (store-week + store-category-week panels) + answer key [sim]
    "21_events_loyalty.py",  # Pillar 2 step 2: Aldi/Lidl openings, Spar closures, loyalty-card panel [sim]
    "22_t1_entry.py",       # Pillar 2 step 3: T1 entry analysis — TWFE vs stacked event study vs pooled exposure [sim]
    # later steps append here: 20_screening.py, 30_simulation.py, ...
]

here = Path(__file__).parent
for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    sys.argv = [str(here / step)]
    runpy.run_path(str(here / step), run_name="__main__")
