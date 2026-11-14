# Data register
*Every input to the analysis, with its source, licence, status label and the file that holds it. Labels: **[real]** observed public data · **[calibrated]** fitted to published aggregates · **[assumed]** stated assumption, varied in sensitivity · **[sim]** simulated (method validation only). Raw files are not committed (`data/raw/`, see `.gitignore`); `python scripts/run_all.py` rebuilds everything else.*

Last updated: 14 Nov 2026.

## Raw inputs

| # | Data | Source | Vintage | Licence / terms | Label | Raw file |
|---|---|---|---|---|---|---|
| R1 | Store locations, banners, shop type | OpenStreetMap, Geofabrik Switzerland extract (MD5 checked) | snapshot 14 Nov 2026 | ODbL, attribution "© OpenStreetMap contributors" | [real] | `data/raw/osm/switzerland-latest.osm.pbf` |
| R2 | Canton (26) and municipality (2,121) boundaries | same OSM extract (admin_level 4 / 8, BFS numbers) | 14 Nov 2026 | ODbL | [real] | same |
| R3 | Drivable road network (4.18 m nodes) | same OSM extract | 14 Nov 2026 | ODbL | [real] geometry | same |
| R4 | Population and private households per hectare | FSO STATPOP 2024 geodata (ag-b-00.03-vz2024statpop) | 31 Dec 2024 | FSO open use, source must be cited | [real] | `data/raw/bfs/statpop2024/` |
| R5 | Population by municipality (cross-check) | FSO PXWeb px-x-0102010000_101 | 2024, 2025 | FSO open use | [real] | `data/raw/bfs/statpop_municipality_2024_2025.csv` |
| R6 | Income distribution by municipality: taxpayers in 10 net-income classes, total net and taxable income | ESTV, direct federal tax statistics, natural persons, municipalities, file 500 (all taxpayers, normal cases) | tax year 2023 (published 5 Nov 2026) | federal open data | [real] | `data/raw/estv/statistik-dbst-np-gemeinden-2023-500.xlsx` |
| R7 | Swiss debit-card spending by country (DEU/FRA/ITA/AUT) × merchant category; domestic by merchant canton | Monitoring Consumption Switzerland (HSG), SIX issuing data | daily, 2019-01-01 to 2023-01-01 | free use, cite MCS / SIX | [real] | `data/raw/mcs/DEBIT_*.csv` |
| R8 | Card spending at Swiss merchants by canton; by cardholder origin × category | MCS, Worldline acquiring data | daily, 2019-01-01 to 2026-11-08 | free use, cite MCS / Worldline | [real] | `data/raw/mcs/ACQ_*.csv` |
| R10 | Calibration targets: domestic food-retail sales shares 2025: Migros 33.5%, Migros group incl. Denner 42.6% (so Denner ≈ 9.1%), Coop 33.7%, Lidl 8.4%, Aldi 7.3%, others ≈ 8% (residual) | watson.ch, 29 Oct 2026 ("Marktanteile"; source not named) | 2025 | press | [real], source unnamed | `assets/p1/1_2_targets.csv` (step 1.2) |
| R11 | Cross-check, 2025 sales (CHF bn): Migros supermarkets incl. online 12.65; Denner 3.85; Coop supermarkets incl. coop.ch 12.40; Aldi ≈ 2.5 and Lidl ≈ 2.7 (2024 NielsenIQ estimates) | Migros and Coop annual reports 2025; NielsenIQ via watson (7 Feb 2026), relayed by rappn.ch | 2024–25 | public reports / press | [real] / [estimate] | same |
| R9 | Published store counts (validation) | Blick, blue News, cash.ch, Statista (see table file) | 2025–Feb 2026 | press | [real] | `outputs/tables/store_count_validation.csv` |

## Derived datasets

| # | Dataset | Built by | How | Label | File |
|---|---|---|---|---|---|
| D1 | Stores: 4,565 outlets of 12 tracked banners, deduplicated, with canton, municipality, ownership group, road-graph node | `scripts/01_osm_extract.py`, `02_geography.py` | brand + name + operator read together; same-banner points < 40 m merged | [real] | `data/processed/stores.parquet` |
| D2 | Store counts by canton | `01_osm_extract.py` | count | [real] | `outputs/tables/store_counts_by_canton.csv` |
| D3 | Shopper locations: 32,388 = H3 res-8 cell × municipality | `02_geography.py` | hectares grouped; population-weighted centroid; income from municipality | [real] | `data/processed/locations.parquet` |
| D4 | Municipal income distribution | `02_geography.py` | mean = total net income / taxpayers; median interpolated within income class | [real] | `data/processed/municipalities.parquet` |
| D5 | Drive times location → store, all pairs ≤ 25 min (5.3 m pairs) | `02_geography.py` | Dijkstra on R3 with free-flow speeds by road class | geometry [real], speeds [assumed] | `data/processed/drive_times.parquet` |
| D6 | Drive time to nearest border crossing per location | `02_geography.py` | 284 road crossings (edges crossing the national boundary; Liechtenstein and Büsingen excluded as they are in the Swiss customs area); multi-source Dijkstra | [real] / [assumed] speeds | column `border_min` in D3 |
| D7 | Detour factor (road km / straight km) | `02_geography.py` | 82,304 sampled location–store pairs 2–25 km apart | [real] | `outputs/tables/detour_factor.csv` |
| D8 | Demand market: 5.31 m location–store pairs ≤ 25 min, 7 income types per location, 4.27 m households | `scripts/10_demand.py` | flat pair arrays sorted by location; 147 locations without a store in reach dropped | [real] inputs | `data/processed/market.npz` |

## Key numbers from Phase 1 (for the appendix)
- **Population:** 9,123,704 in 347,736 populated hectares → 32,388 shopper locations.
- **Income match:** 11 municipalities (27,823 people, 0.3%) have no ESTV 2023 income because of mergers or boundary changes. They get their canton's population-weighted mean [imputed].
- **Snapping to the road network:** median distance 43 m for locations, 26 m for stores; p95 433 m and 69 m.
- **Detour factor:** median 1.20, mean 1.25, p10–p90 1.10–1.42. The design assumption of ~1.3 is slightly high; we use network drive times directly, so the factor is only a check.
- **Drive-time validation, free flow** (city centre to city centre):

  | Route | Our estimate | Typical real time |
  |---|---|---|
  | Zurich–Winterthur | 18 min | ~22 |
  | Bern–Thun | 21 min | ~25 |
  | Lausanne–Geneva | 42 min | ~45 |
  | Basel–Zurich | 54 min | ~60 |

  Free-flow times run 10–25% fast in cities. A congestion multiplier is in the sensitivity analysis.
- **Border reach:** 23% / 51% / 73% of the population live within 15 / 30 / 45 min of a road border crossing.

## Assumptions to carry into sensitivity analysis
| Assumption | Value | Range to test |
|---|---|---|
| A1 Free-flow speeds (km/h) | motorway 100 · trunk 80 · primary 60 · secondary 50 · tertiary 40 · unclassified 30 · residential 25 | congestion multiplier 1.0–1.3 on urban roads |
| A2 Choice-set cutoff | 25 min drive | 15 / 20 / 30 |
| A3 Taxpayer ≈ household | married couples file jointly | compare with STATPOP households (HPTOT) |
| A4 Oneway restrictions | ignored | — (small effect on minutes) |
| A5 OSM completeness | Denner partners under-mapped (−6%) | reweight Denner by published count |
| A6 Basket price index (Migros = 100) | Migros Partner 103, Coop 102, Denner 88, Aldi 85, Lidl 85, Migrolino 115, Coop Pronto 115, Volg 108, Landi 100, Spar 108, Manor 110; abroad 70 | ±5 per banner; abroad 60–80. Screening (Koh 2025) needs only margins and revenue diversion |
| A7 Demand priors | α_ref 0.03, η 0.5, γ 0.06/min, κ 0.04/min, σ 0.25 | σ 0–0.6, η 0–1, γ/α CHF 1–3 per min; δ calibrated in step 2 |
| A8 Income types | ESTV taxpayer classes merged into 7; representative incomes 20k, 35k, 45k, 62.5k, 87.5k, 150k, 300k | top-class value 200k–500k |
