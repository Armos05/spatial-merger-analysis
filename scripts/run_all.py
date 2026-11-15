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
    "23_t2_elasticity.py",  # Pillar 2 step 4: T2 elasticities — OLS vs IV vs DML-PLR vs DML-PLIV [sim]
    "24_t3_local.py",       # Pillar 2 step 5: T3 store-level elasticities (causal IV forest) + local GUPPIs [sim]
    "25_scorecard.py",     # Pillar 2 step 6: validation scorecard T1-T3 [sim]
    "26_border.py",        # Pillar 2 step 7: border evidence on real card data (COVID, duty-free 2025, franc) 
    "30_experiments.py",   # Pillar 3 step 1: coupon A/B tests → uplift (T-learner) → α̂ per household [sim]
    "31_prediction.py",    # Pillar 3 step 2: prediction quality, Cumulus only vs pooled; learning curve [sim]
    "32_pricing.py",       # Pillar 3 step 3: {uniform, personal} × {competition, merger}; interaction [sim]
    "33_ring_fence.py",    # Pillar 3 step 4: local-DP ring-fence, privacy-vs-competition curve [sim]
    "34_audit.py",         # Pillar 3 step 5: WEKO test-account audit + power calculation [sim]
    "35_outcomes_data.py", # Pillar 3 step 6: outcome-table rows with the data channel (B+, B++, E, E', F)
    "40_sensitivity.py",   # Phase 5 step 1: implicit-differentiation sensitivities, tornado, σ map
    "41_montecarlo.py",    # Phase 5 step 2: Monte Carlo, 90% intervals on the outcome table
    # later steps append here: 20_screening.py, 30_simulation.py, ...
]

here = Path(__file__).parent
for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    sys.argv = [str(here / step)]
    runpy.run_path(str(here / step), run_name="__main__")
