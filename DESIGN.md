# Migros × Coop: a merger of stores *and* of shopper data
### Design document — v0.2 (14 Nov 2026)

> Working document. Facts marked **[verify]** must be checked against a primary source before they appear in the memo. Anything simulated is labelled **[sim]** everywhere it appears — in code, figures and text. v0.1 is kept as `DESIGN_v0.1_archive.md`.

**Standard:** the report must read as a client-ready Keystone deliverable, not coursework. Toy models from `LEARNING_MODULES.md` appear only as appendix sanity checks.

---

## 0. Engagement and question

**Framing (hypothetical):** Keystone is retained by counsel to **Migros (acquirer)** for a **pre-notification antitrust risk assessment** of its proposed acquisition of **Coop (target)**, before filing with the Swiss Competition Commission (WEKO/COMCO). Coop's Supercard data is "target data". Full brief: `report/00_engagement_brief.md`.

Why this framing: it is a service Keystone sells (Merger Advisory page: "antitrust risk assessments" before agency review), and it forces an even-handed analysis — where the risk is robust, where it is not, and which remedies would clear it.

**Questions the client needs answered**
1. Where, and by how much, would prices rise? (local markets, not one national number)
2. How much does cross-border shopping discipline a merged retailer, and where does that discipline stop?
3. Will WEKO see pooling Cumulus + Supercard data as a separate harm, and how large is it?
4. How robust is each finding to the uncertain parameters — what would the other side's economist attack?
5. Which remedies (divestitures, data ring-fence, commitments) would resolve the concerns, and at what cost?
6. Which of these questions could be answered better on the parties' own data — and is the pipeline ready for it?

**The answer must fit in one paragraph** at the top of the memo (template; numbers filled Sunday night):
> "On public data, we estimate the merger would raise grocery prices by X–Y% on average and by up to Z% in N regions more than ~K km from the border, costing households CHF A–B per year. Cross-border shopping constrains prices only near the border. Pooling loyalty data adds CHF C of harm through more precise personalised pricing. Local screening flags M overlap areas above the CMA 2.75% threshold. A remedy package of [divestitures in flagged areas + data ring-fence] resolves most of the risk. Our ML pipeline, validated on a simulated replica of the market, is ready to run on the parties' transaction data to sharpen these estimates."

Every number comes with a range (Monte Carlo over parameters), never a single point.

---

## 1. Report architecture — three pillars

| Pillar | Question it answers | Methods | Why it is not econ 101 |
|---|---|---|---|
| **1. Structural merger analysis** | How much do prices and welfare change, where, and under which assumptions? | spatial nested logit with heterogeneous price sensitivity; cooperative objective; identification without price data (Koh 2025); local GUPPIs + compensating marginal cost reductions; full equilibrium simulation; differentiable simulator for uncertainty | frontier IO: cooperative conduct, no-price identification, local screening at scale, gradient-based sensitivity |
| **2. Data-driven validation ("digital twin")** | How would we measure substitution on the parties' data, and do the ML methods beat the traditional ones? | simulated replica of Swiss grocery on real geography; pooled store-level entry analysis; double machine learning; causal forests; synthetic DiD / RDD on real border events; ML shopper segmentation | mirrors Keystone's own "Next Frontier in Merger Review" agenda and *quantifies* its claims against a known truth |
| **3. Data as an asset** | Is combining loyalty data a separate harm, and can a remedy fix it? | personalised pricing with segment targeting; value of pooled data; differential-privacy data ring-fence; audit design with power calculation | "value of combined data" theory of harm (Keystone Merger Advisory page) made quantitative |

Extensions (appendix only): supplier bargaining (Nash-in-Nash), entry thresholds, algorithmic pricing (Q-learning), product-range matching.

---

## 2. Market facts (inputs)

| Fact | Value | Source | Status |
|---|---|---|---|
| Coop share, food & near-food | 43% | Blick (citing market data; year not stated) | [verify] year & definition; Coop gained share in 2025 (Lebensmittel Zeitung) |
| Migros share, food & near-food | 37.4% | same | [verify] year |
| Combined share | ~70–80% depending on definition; **69% of tracked grocery outlets** in our OSM count (3,158 / 4,565) | Greenpeace 2026 (~80%); own count | partly verified (outlets ≠ sales) |
| Denner owned by Migros | WEKO cleared with conditions, 3 Sep 2007: Denner kept as separate brand, **excluded from Cumulus**, supplier protection, measures easing Aldi/Lidl entry; for 7 years or until Aldi + Lidl reach 250 sites | admin.ch press release; Tages-Anzeiger | **verified**; conditions have lapsed (Aldi + Lidl now ~430 stores) [verify exact end date] |
| Aldi Suisse / Lidl Schweiz | 244 stores (Feb 2026) / ~190 | Blick; blue News | **verified**; OSM 239 / 190 |
| Coop / Migros / Denner stores | ~1,000 Coop supermarkets (end 2025); 672 Migros; Denner ~600 own + ~260 partner | cash.ch; Statista; Blick | verified ±7% vs OSM (`outputs/tables/store_count_validation.csv`) |
| Others: Volg/Landi (fenaco), Spar, Manor food, Migrolino, Coop Pronto | OSM: Volg 580, Landi 127, Spar 214, Manor 57, Migrolino 280, Coop Pronto 249 | OSM | counted; published counts [verify] |
| Legal form | both cooperatives; Migros = federation of 10 regional cooperatives | annual reports | [verify] member counts |
| Migros supermarkets don't sell alcohol | all 10 cooperatives rejected lifting the ban (vote ended 4 Jun 2022, turnout 29%; 2/3 majority needed in 9 of 10); Denner and Migrolino do sell alcohol | SRF, Blick, Année Politique Suisse | **verified** |
| Vertical integration upstream | Coop → Bell Food Group; Migros → industrial arm (e.g. Micarna, Elsa) | annual reports | [verify] current structure |
| Loyalty programmes | Cumulus > 3 million cardholders; Supercard count not found | Blick | Cumulus verified (approx.); Supercard [verify] |
| Personalised-discount pilot | Sept 2016, 3 stores (Zurich, Aare, Geneva cooperatives), personalised discounts at the till/app; Migros CEO: "no personalised prices", only personalised offers | NZZ am Sonntag via swissinfo; Horizont | **verified** (2016) |
| Duty-free import allowance cut CHF 300 → 150 per person per day | 1 Jan 2025 (VAT due above it) | admin.ch | **verified** |
| SNB removes EUR/CHF floor | 15 Jan 2015 | — | known |
| COVID border closure | entry restrictions from mid-Mar 2020; DE/AT/FR borders reopened 15 Jun 2020 | MCS card data shows foreign food spend −90% Apr–May 2020 | consistent with data; [verify] official dates |
| Swiss merger test | Parliament adopted Cartel Act partial revision 19 Dec 2025: **SIEC test replaces qualified-dominance test**; entry into force expected 2027 | CMS, Bär & Karrer, economiesuisse | **verified** (in-force date [verify]) — a 2027 filing would face SIEC |

**Literature we build on (not redo):** Ellickson–Grieco–Khvastunov (2020) spatial grocery demand; CMA Sainsbury's/Asda (2019) local GUPPI practice; Koh (2025) merger analysis without prices; Burstein–Lein–Vogel cross-border shopping in Switzerland [verify title]; Chen–Choe–Cong–Matsushima (2022) data-driven mergers; Keystone (2024) "Next Frontier in Merger Review". Citations from memory still to verify: Conlon–Mortimer (2021) diversion ratios; Backus–Conlon–Sinkinson (2021) common ownership; Miller–Weinberg (2017).

---

## 3. Pillar 1 — Structural merger analysis

### 3.1 Game structure
```
Stage 0  WEKO          chooses {clear, clear with remedies, prohibit}
Stage 1  Aldi / Lidl   expand store networks                    (appendix: entry thresholds)
Stage 2  Retailers ↔ suppliers  bargain over wholesale prices   (appendix: Nash-in-Nash)
Stage 3  Retailers     set shelf prices (+ personalised prices)  (Bertrand-Nash, cooperative objective)
         Households    choose where to shop, incl. abroad        (spatial nested logit)
```
Solved backwards. Stage 3 is the core of the report; stages 1–2 appear as appendix extensions.

### 3.2 Demand — spatial nested logit with heterogeneous price sensitivity
Markets: H3 hexagons (resolution ~7–8, a few km²) with population from STATPOP; each hexagon's households choose among nearby stores and abroad.
```
u_ijh = δ_b(j) − α_i × p_b(j) − γ × time_jh + ξ_j + ζ_ig + (1 − σ) × ε_ijh

i = household, j = store, h = hexagon, b(j) = banner of store j (Migros, Coop, Denner, Aldi, Lidl, other)
δ_b      = banner quality
α_i      = α_0 + α_inc × income_i   (price sensitivity varies with income — light random coefficients)
time_jh  = travel distance from location h to store j: straight-line × validated detour factor (drive times optional)
ξ_j      = store-specific quality (size, parking), calibrated
g        = nest: {full-service: Migros (+ Partner/VOI), Coop}, {discount: Denner, Aldi, Lidl},
           {convenience: Migrolino, Coop Pronto}, {other: Volg, Landi, Spar, Manor}   (decided 14 Nov)
ownership: Migros group = Migros + Partner + Migrolino + Denner; Coop group = Coop + Coop Pronto;
           Aldi; Lidl; fenaco (Volg, Landi); Spar; Manor.  Outside option = abroad.
σ        = within-nest correlation — high σ ⇒ Migros and Coop are close substitutes (the crux)
```
Outside option (abroad), from Pillar 2's border evidence:
```
u_i0h = δ_0 + β × gap_t − κ × border_time_h + ε_i0h
```
Outputs: store-level and banner-level elasticities, **local diversion ratios** D(j→k) for every store pair.

**As implemented (Phase 2 step 1, 14 Nov).**
- *Income types.* Each location's households are split into 7 income types using its municipality's ESTV 2023 taxpayer distribution (<30k, 30–40k, 40–50k, 50–75k, 75–100k, 100–200k, 200k+). Integrating over the real income distribution is the "light random coefficient".
- *Price sensitivity.* We use α_r = α_ref × (y_r / 62,500)^(−η) rather than linear in income, which keeps α > 0 for the top class. η = 0 means no income effect; η = 1 means α ∝ 1/income.
- *Travel.* Network drive times are used directly, with no detour factor. The choice set is every store within 25 min.
- *Excluded.* 147 locations (1,633 households, 0.04%) with no store within 25 min.
- *ξ_j.* Set to 0 for now. A hook exists for store-format effects.
- *Prices.* Banner basket index with Migros = 100, all [assumed]; abroad = 70 (see DATA_REGISTER A6).

### 3.3 Supply — cooperative conduct
```
W_f = Profit_f + λ_f × μ_f × CS
⇒ p − c = (1 − λ_f μ_f) × [standard multiproduct markup]
```
Same algebra as **common-ownership profit weights** (O'Brien–Salop; Backus–Conlon–Sinkinson) — λμ is a conduct parameter. Identify λμ by matching implied margins to annual-report margins (range, since accounting ≠ economic margins). Merger: ownership matrix changes (Migros + Denner + Coop), μ becomes the union of memberships, λ post-merger is a scenario (survives / dilutes / disappears).

### 3.4 Identification without shelf prices (Koh 2025)
Swiss banner-level basket prices are not public at local level. Following Koh, GUPPIs and **compensating marginal cost reductions (CMCR)** are identified from **revenues, margins and revenue diversion ratios**; simulation is reported in **percentage price changes**. Our demand model supplies revenue diversion; annual reports supply margins.

### 3.5 Local screening
For every overlapping Migros/Coop/Denner store pair within a drive-time catchment:
```
GUPPI_j = D(j→k) × margin_k × p_k / p_j
CMCR    = cost reduction that would exactly offset the upward pricing pressure
```
Decision rule benchmark: CMA Sainsbury's/Asda thresholds — 2.75% (supermarkets), 3.25% (convenience). Output: map of flagged areas and count of stores in them (= the divestiture list).

### 3.6 Simulation and scenarios
Full post-merger equilibrium (damped fixed point with convergence check, or Morrow–Skerlos ζ-map), consumer surplus via log-sum, compensating variation in CHF per household per year. Scenarios: efficiencies 0–10%; λμ survives / dilutes / disappears; with/without data pooling (Pillar 3); remedy packages.

### 3.7 Uncertainty — differentiable simulator (JAX)
- Entire pipeline (shares → equilibrium → CS) written in JAX.
- **Implicit differentiation through the equilibrium** (implicit function theorem) gives exact d(harm)/d(parameter) for σ, α, λμ, β, κ → tornado chart of what drives the answer.
- **Vectorised Monte Carlo** (10k parameter draws) → 90% intervals on every headline number.

---

## 4. Pillar 2 — Data-driven validation on a "digital twin"

### 4.1 Why
Keystone's "Next Frontier" paper argues that merger review should measure substitution the way firms do: granular transaction data + ML + causal design. We have no client data. So we build a **digital twin** — a simulated replica of Swiss grocery on **real geography** — where the true diversion ratios and elasticities are known, then show which methods recover them. Framing in the memo: *"pipeline validated on simulated data; ready to run on the parties' data on day one."*

**Honesty rule:** every Pillar 2 result is labelled "method validation on simulated data [sim]". No twin output is presented as a finding about Switzerland.

### 4.2 Twin specification [sim on real geography]
- **Stores:** real OSM locations of Migros, Coop, Denner, Aldi, Lidl, others (~3,000+).
- **Households:** synthetic, placed by STATPOP population on H3 cells; income from municipality data; true α_i, brand tastes and segment membership drawn from known distributions.
- **Sales panel:** store × product category (~20) × week (~3 years), generated from the Pillar 1 demand model with the *true* parameters.
- **Realism / messiness:** prices respond to demand shocks (endogeneity), promotions, seasonality, cost shocks (instruments), missing weeks, measurement noise.
- **Events:** staggered Aldi/Lidl store openings at real locations with simulated dates; one store closure wave.
- **Loyalty data:** card-level trips for a sample of households, partially observed by each chain.
- Scale: ~10M rows; seeded and reproducible.

### 4.3 The tests (each = traditional method vs data-driven method vs truth)
| ID | Keystone technique | Traditional | Data-driven | Recovers |
|---|---|---|---|---|
| T1 | Entry / intrusion analysis | store-month revenue regression on entry dummy | store-by-store event studies vs random control stores, pooled; staggered-adoption estimator | local diversion to entrant, by distance |
| T2 | Demand estimation | OLS log-quantity on log-price | **double machine learning** (DML) with high-dimensional controls + cost-shock instrument | own-price elasticity (bias of OLS shown) |
| T3 | Localised effects | one national elasticity | **causal forest** (Athey–Wager): elasticity per store/hexagon vs distance to rival | local elasticities → local GUPPIs |
| T4 | Shopper segmentation | segments by judgement | unsupervised clustering of loyalty baskets (k-means / Gaussian mixture) | true segments; merger harm by segment |
| T5 | "Trade quantity for quality" | — | RDD around price changes in the twin | elasticity from a narrow window |

**Validation scorecard (appendix table):** for each test — bias, RMSE, 95% coverage, and precision vs the traditional method.

### 4.4 Real-data causal evidence: the border
These use **real** data where it exists (decided by the data check):
| Event | Design | Identifies |
|---|---|---|
| Franc shock, 15 Jan 2015 | **synthetic DiD** (Arkhangelsky, Athey et al. 2021), plain DiD/event study as robustness | β (response to price gap), κ (distance decay) |
| Duty-free cut, 1 Jan 2025 | RDD in time + placebo year (difference-in-discontinuities) | sensitivity to the allowance (a policy lever) |
| COVID border closure 2020 | out-of-sample test of the demand model | validation |

Data candidates: Monitoring Consumption Switzerland (card spending by canton/category), FSO retail turnover, customs counts. If too coarse → calibrate to Burstein–Lein–Vogel and label **[calibrated]**.

**Data check result (14 Nov):** Monitoring Consumption Switzerland (MCS) public files are downloaded (`data/raw/mcs/`).
- **COVID closure 2020 → real.** SIX debit-card issuing data, daily 2019-01 to 2023-01, covers Swiss cards' spending by country (DEU/FRA/ITA/AUT) and merchant category, plus domestic spending by merchant canton. Swiss debit spend at German food retailers fell ~90% in Apr–May 2020. **This is now the primary border experiment.**
- **Duty-free cut 2025 → real.** Worldline acquiring data, POS by canton, runs daily 2019-01 to 2026-11. It supports a difference-in-discontinuities at 1 Jan 2025 (border vs interior cantons, placebo 1 Jan 2024).
- **Franc shock 2015 → no card data (it starts in 2019).** Calibrate to literature and label [calibrated]; it moves to robustness.

---

## 5. Pillar 3 — Data as an asset (Cumulus + Supercard)

**Theory of harm:** today each chain sees only its own customers' baskets, and competition disciplines targeted pricing; after the merger one firm sees ~all purchase histories **and** faces no close rival. The sign depends on what the data reveal (Learning Module 5):
- **brand-preference data** → poaching discounts help shoppers under competition (Thisse–Vives); they vanish after the merger.
- **price-sensitivity data** → targeted price increases hurt shoppers even under competition, more after the merger.
- The harm is the **interaction** of data pooling and loss of the rival; estimate both channels.

**Build (on the twin's loyalty data):**
1. Randomised coupon experiments (A/B tests) → heterogeneous treatment effects (causal forest / T-learner) → predicted α_i per household.
2. Prediction quality: one chain's data vs pooled data.
3. Feed predicted α_i into pricing: {uniform, personalised} × {competition, merger}.
4. **Remedy: differential-privacy ring-fence / data access** — share data with rivals under privacy level ε; plot how much targeting value survives → privacy-vs-competition curve.
5. **Audit design for WEKO:** paired synthetic shopper profiles; power calculation for a given price gap.

---

## 6. Extensions (appendix only)
- **Supplier bargaining (Nash-in-Nash):** toy result (Module 6) — suppliers lose ~22%, almost all through volume; buyer power needs survival threats / private label / size-dependent bargaining weight. One paragraph + one table.
- **Entry thresholds (Bresnahan–Reiss):** toy result — entry offsets at most ~22% of harm. One paragraph; the twin's T1 replaces the rest.
- **Algorithmic pricing (stretch):** Q-learning agents (Calvano et al. 2020) pre/post merger — do learned prices exceed Nash? Only if ahead.
- **Product-range matching (stretch):** multilingual embeddings on Open Food Facts → Migros–Coop assortment overlap (informs σ).

---

## 7. Outcome table and remedies

| Scenario | Δ price (avg / far from border) | Δ CS (CHF/household/yr) | Areas above 2.75% GUPPI | Δ retailer profit | Δ total welfare |
|---|---|---|---|---|---|
| A. Status quo | 0 | 0 | — | 0 | 0 |
| B. Merger, no remedy | | | | | |
| C. B + cost efficiencies | | | | | |
| D. C + divest flagged stores (incl. Denner?) | | | | | |
| E. C + data ring-fence (DP) | | | | | |
| F. D + E | | | | | |
| G. Prohibit | 0 | 0 | — | 0 | 0 |

All cells as 90% intervals.

---

## 8. Exhibits (ghost memo — each must have a "so what" title)
1. One-paragraph answer + outcome table (§7).
2. Map: local GUPPIs by overlap area vs 2.75% threshold.
3. Harm by drive time to the border (border discipline fades with distance).
4. Event-study / synthetic DiD: franc shock, near vs far.
5. Tornado: which parameters drive the harm (implicit-diff sensitivities).
6. Data merger: harm under {uniform, personalised} × {competition, merger}; DP privacy-vs-competition curve.
7. Digital twin scorecard: traditional vs ML methods vs truth.
8. Remedies: harm remaining under each package.
Final list fixed in `report/01_ghost_memo.md` (14 Nov): 1 outcome table · 2 diversion · 3 GUPPI map · 4 border evidence + reach · 5 data harm 2×2 + DP curve · 6 tornado · 7 twin scorecard · 8 remedy waterfall.

---

## 9. Data plan and register

| Data | Source | Status |
|---|---|---|
| Store locations by banner | OpenStreetMap (Overpass, `shop=supermarket`, `brand=*`) | real |
| Population grid | FSO STATPOP (hectare grid) → H3 | real |
| Municipality income | FSO / tax statistics | real [verify] |
| Travel distance | straight-line × detour factor, validated on ~200 osmnx road routes; OSRM via Homebrew optional (no Docker) | real / calibrated |
| Border crossings / foreign shopping towns | OSM / swisstopo | real |
| Market shares | published Nielsen figures | real (aggregate) |
| Gross margins, member counts | Migros/Coop annual reports | real [verify] |
| Price level by banner | published basket comparisons | real [verify] |
| Cross-border spending | Monitoring Consumption Switzerland / FSO / literature | real if available, else [calibrated] |
| Weekly sales, loyalty data, entry dates | digital twin | [sim] |
| Supplier costs, bargaining weights | — | [sim], swept |

**Data register (appendix):** every input listed as observed / calibrated / assumed / simulated, with source.

---

## 10. Validation and tests
- Calibration fit: shares, store counts, margins.
- Out-of-sample: COVID closure prediction vs data.
- Twin scorecard: bias / RMSE / coverage per method.
- Unit tests: logit derivatives vs finite differences; equilibrium residual max |target − price| < 1e-8 (convergence check — toy showed damping 0.5 fails at extreme parameters); zero diversion ⇒ zero price change; λμ = 1 ⇒ marginal-cost pricing; JAX implicit gradients vs finite differences.

---

## 11. Repo structure
```
case-study/
  DESIGN.md  README.md  pyproject.toml
  src/mxc/
    data/        osm.py, statpop.py, h3grid.py, drivetime.py, borders.py, calibration.py, register.py
    structural/  demand.py, supply.py, equilibrium.py, merger.py, guppi.py, uncertainty.py   (JAX)
    twin/        generate.py, events.py, loyalty.py
    causal/      entry_pooled.py, dml.py, causal_forest.py, segmentation.py, rdd.py, sdid.py
    datamerger/  personalization.py, privacy.py, audit.py
    extensions/  bargaining.py, entry.py, (algocollusion.py), (matching/)
  scripts/run_all.py          → figures/, tables/
  tests/
  report/  00_engagement_brief.md, 01_ghost_memo.md, memo.md → memo.pdf
  figures/  tables/
```
Environment: fresh (not the broken Anaconda) — jax, numpy, scipy, pandas, geopandas, h3, osmnx, statsmodels/linearmodels, scikit-learn, econml, matplotlib. Pinned versions, fixed seeds, one command reproduces everything.

---

## 12. Alignment map

**Keystone "Next Frontier in Merger Review" (2024) → this case**
| Their technique | Here |
|---|---|
| Granular entry analysis, pooled store-level estimates, localised diversion | T1 + local GUPPIs (§3.5) |
| Demand with thousands of controls (ML) | T2 DML |
| Localised elasticities | T3 causal forest |
| "Trade quantity for quality" (RDD) | T5 + 2025 duty-free RDD |
| Pricing algorithms in simulation | Extension: Q-learning |
| Efficiencies via DML on supply chain | memo: how we would verify efficiencies on client data |
| ML customer segmentation; effects by customer group | T4 + Pillar 3 |
| Card-transaction / third-party data | border evidence (§4.4); memo paragraph on what card data would add |

**Job-fair keywords → this case**
| Keyword | Where |
|---|---|
| Predictive modelling | causal forest / T-learner α_i; DML |
| Game theory | Bertrand with cooperative conduct; Nash-in-Nash; entry; WEKO decision |
| A/B testing | coupon experiments; audit power calculation |
| RDD | 2025 duty-free cut; twin T5 |
| Simulation strategies | merger simulation; digital twin; Monte Carlo |
| Testing strategies | twin scorecard; unit tests; out-of-sample COVID test |
| Damage calculation | compensating variation in CHF with intervals |
| Quantitative analysis / report drafting | memo + appendix |

---

## 13. Schedule and cut-lines (revised 14 Nov)

| When | Work | Done = |
|---|---|---|
| Sat AM | Step 0 brief, Step 1 ghost memo; environment; facts check; data feasibility | brief + exhibit list fixed; `import jax, geopandas, h3` works |
| Sat PM | Data pipeline (OSM, STATPOP→H3, drive times, borders); Pillar 1 demand + calibration | calibrated demand, diversion ratios |
| Sat night | Pillar 1 supply, simulation, local GUPPIs, scenarios | outcome rows A–D, GUPPI map |
| Sun AM | Pillar 2 twin generator + T1–T3; border evidence (SDID/RDD) | scorecard draft; border exhibit |
| Sun PM | Pillar 3 data merger + DP remedy; T4; uncertainty (Monte Carlo + tornado) | rows E–F; all numbers with intervals |
| Sun night | Freeze: figures, tables, tests green, one-command run | **no new analysis after this** |
| Mon | Memo, appendix, README, web summary; CV; application; emails/messages | send-ready |
| Tue AM | Fresh-eyes reread (Module 7.9 checklist) → send | — |

**Cut order if behind at Sunday noon:** (1) extensions → toy paragraphs only; (2) T5 and T4 → drop; (3) DP remedy → text only; (4) SDID → calibrated border parameters with stated method.
**Never cut:** Pillar 1 core (demand, simulation, local GUPPIs, uncertainty), T1–T3 scorecard, the one-paragraph answer, reproducibility.

---

## 14. Open items (Saturday morning)
1. Which regional cross-border / card-spending data exist? (real vs calibrated in §4.4)
2. Burstein–Lein–Vogel: exact paper and estimates usable as priors.
3. Annual-report margins and member counts.
4. Published basket price comparisons by banner.
5. Status of the Swiss Cartel Act revision (dominance vs SIEC test) — matters for how the memo states the legal standard.
6. Verify the from-memory citations (§2).
7. Framing guardrail: hypothetical merger; public data + clearly labelled simulation; no Keystone slide material copied.

---

## Change log
- **v0.2 (14 Nov):** framing → pre-notification risk assessment for the parties; three-pillar architecture; digital-twin validation (Pillar 2) aligned with Keystone's "Next Frontier" paper; Koh (2025) no-price identification + CMCR; H3/drive-time local markets; income-varying α; bargaining and entry moved to appendix; schedule shifted to Sat–Tue; math in plain text.
- v0.1 (12–13 Nov): four-stage game, WEKO framing, CS layer (§14b) — archived.
