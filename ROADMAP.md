# Roadmap — Migros acquires Coop (case study)
*Tracks the finalised design (`DESIGN.md` v0.2). Tick items as they are done. Freeze analysis Sunday night; write Monday; send Tuesday morning.*

**Geography decision (14 Nov):**
- **Canton map** is the backbone: stores counted per canton from OpenStreetMap; real canton data (population, income, card spending) for calibration; canton maps as the main visuals.
- **Shoppers live in municipalities** inside each canton (~2,100, real population and income). That keeps local catchments small enough for local GUPPIs: a canton like Zurich or Bern is far bigger than a grocery market.
- **Shoppers are simulated from real data:** synthetic households drawn from real municipal population and income, shopping at real store locations.

Legend: `[x]` done · `[ ]` to do · *(cut)* = first to drop if behind

---

## Phase 0 — Framing and learning (Thu–Sat AM)
- [x] Topic chosen: Migros × Coop, "a merger of stores and shopper data"
- [x] Keystone does merger work (Merger Advisory page; "value of combined data")
- [x] Learning modules 1–7 (`LEARNING_MODULES.md`)
- [x] References: Ellickson–Grieco–Khvastunov (EGK), CMA Sainsbury's/Asda, Keystone "Next Frontier", Koh 2025 (+ summary)
- [x] Design v0.2: three pillars + digital twin (`DESIGN.md`)
- [x] Step 0 engagement brief: client = Migros (acquirer), target = Coop (`report/00_engagement_brief.md`)
- [x] Step 1 ghost memo: answer paragraph with placeholders + final exhibit titles (`report/01_ghost_memo.md`)
- [ ] Self-test on EGK (intro + model) — 8 questions (`EGK_SELF_TEST.md` written; tick when answered ≥6/8)

## Phase 1 — Environment and data (Sat)
- [x] Fresh Python environment: conda env `keystone` (Python 3.11, conda-forge, `environment.yml`) — jax, numpy, scipy, pandas, geopandas, h3, osmnx, pyosmium, statsmodels, scikit-learn, econml, linearmodels, pyfixest, matplotlib, contextily; import + smoke tests pass (14 Nov)
- [x] Repo skeleton (`src/mxc/{data,structural,twin,causal,datamerger,extensions}`, `tests/` (13 passing), `scripts/run_all.py`, `pyproject.toml`)
- [x] **OSM stores:** Migros, Coop, Denner, Aldi, Lidl (+ Migrolino, Coop Pronto, Volg, Spar) with coordinates, banner, format
- [x] Clean brands; store counts per canton (`outputs/tables/store_counts_by_canton.csv`) vs published counts (`store_count_validation.csv`: all 5 main banners within ±7%)
- [x] **Canton boundaries** (26) + **municipality** boundaries (2,121 with BFS number), from the OSM extract (`data/interim/admin_boundaries.gpkg`)
- [x] Population (STATPOP 2024 hectare grid → 32,388 shopper locations) and income (ESTV 2023: full net-income distribution per municipality)
- [x] Drive times location → store from the OSM road network (scipy Dijkstra, free-flow speeds; 5.3 m pairs ≤ 25 min) — better than straight-line × detour factor; no Docker
- [x] Detour factor checked on 82k pairs: median 1.20 (p10–p90 1.10–1.42); city-pair drive times vs known routes (`DATA_REGISTER.md`)
- [x] ~~OSRM~~ not needed: own road-graph drive times cover it
- [x] 284 road border crossings (LI and Büsingen excluded: Swiss customs area); drive time to border per location (51% of people within 30 min)
- [x] Facts check DESIGN §2: 9 verified (incl. SIEC test adopted Dec 2025, duty-free cut, Denner 2007 conditions lapsed, alcohol vote, 2016 pilot); still open: share year, margins, Supercard members, upstream structure
- [x] Card data found (MCS/SIX/Worldline): COVID closure 2020 + duty-free cut 2025 → **real**; franc shock 2015 → calibrated (no card data before 2019)
- [x] Data register (`DATA_REGISTER.md`)

## Phase 2 — Pillar 1: structural merger analysis (Sat PM – Sat night)
- [x] Demand: spatial nested logit, income-varying α, drive time; nests full-service / discount / convenience / other (`src/mxc/structural/demand.py`, `scripts/10_demand.py`; 6 tests incl. exact match to textbook formulas and the toy)
- [ ] Calibrate to national shares (+ canton shares where available) and margins; fit table
- [ ] Diversion ratios: store, banner, canton
- [ ] Supply: cooperative objective (λμ), cost recovery
- [ ] Koh-style no-price identification (revenue diversion + margins → GUPPI, CMCR)
- [ ] Local GUPPIs + CMCR for every Migros/Denner–Coop overlap; flag vs 2.75% / 3.25%
- [ ] Merger simulation (convergence-checked); CS / compensating variation in CHF per household per year
- [ ] Scenarios: efficiencies; λμ survives / dilutes / disappears
- [ ] Outcome table rows A–D
- [ ] Unit tests: derivatives vs finite differences; equilibrium residual; zero-diversion; λμ = 1

## Phase 3 — Pillar 2: digital twin + causal/ML validation (Sun AM)
- [ ] Twin generator: synthetic households from real municipal data → weekly store × category sales (endogenous prices, promotions, seasonality, cost shocks, noise, missing weeks)
- [ ] Simulated Aldi/Lidl openings at real locations; loyalty-card trips for a household sample
- [ ] T1 pooled store-level entry analysis vs traditional regression
- [ ] T2 DML elasticities vs OLS
- [ ] T3 causal forest local elasticities → compare with true local diversion
- [ ] T4 ML shopper segmentation *(cut)*
- [ ] T5 RDD around price changes *(cut)*
- [ ] Validation scorecard (bias, RMSE, coverage)
- [ ] Border evidence (real data if available): franc-shock synthetic DiD + event study; 2025 duty-free RDD; COVID out-of-sample check

## Phase 4 — Pillar 3: data as an asset (Sun PM)
- [ ] A/B coupon experiments in the twin → heterogeneous α_i (causal forest / T-learner)
- [ ] Prediction quality: Cumulus only vs Cumulus + Supercard
- [ ] Pricing: {uniform, personalised} × {competition, merger}; interaction term
- [ ] Differential-privacy ring-fence: privacy-vs-competition curve *(text only if behind)*
- [ ] WEKO audit design + power calculation
- [ ] Outcome table rows E–F

## Phase 5 — Uncertainty and freeze (Sun PM – Sun night)
- [ ] JAX implicit-differentiation sensitivities → tornado chart
- [ ] Monte Carlo (10k draws) → 90% intervals on every headline number
- [ ] Appendix extensions from toys: bargaining, entry *(paragraphs only)*
- [ ] Stretch: Q-learning pricing or product matching *(only if ahead)*
- [ ] All figures and tables generated by `scripts/run_all.py`; tests green
- [ ] **FREEZE — no new analysis after Sunday night**

## Phase 6 — Writing (Mon)
- [ ] Memo (4–6 pages): one-paragraph answer, outcome table, 6–8 exhibits with "so what" titles
- [ ] Technical appendix: model, data register, scorecard, sensitivities, toy sanity checks
- [ ] Repo README (one command to reproduce) + public repo/link
- [ ] One-page web summary
- [ ] CV, one page MBB-style: PhD, research and publications on top; programming; entrepreneurship; notable professors
- [ ] Application answers (Greenhouse)
- [ ] Email to Greg Crawford; LinkedIn messages to Julian and Theresa

## Phase 7 — Send (Tue AM)
- [ ] Fresh-eyes reread against the Module 7.9 checklist
- [ ] Links work in a private browser window
- [ ] Submit application → send email → send messages
