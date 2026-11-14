# Ghost memo (Step 1): DRAFT skeleton
*Written before the analysis, on purpose. Every headline below is a **hypothesis** stated as the finding we expect. Placeholders in [brackets] are filled from `scripts/run_all.py` on Sunday night. If a result comes out the other way, the headline is rewritten to say so (see "If wrong" under each exhibit). Nothing here is a finding yet.*

Labels used throughout: **[real]** public data · **[calibrated]** fitted to published aggregates · **[assumed]** stated assumption with a range · **[sim]** method validation on simulated data.

---

**DRAFT: hypothetical engagement** *(case study on public data)*

**To:** External antitrust counsel to Migros-Genossenschafts-Bund
**From:** [Author], economic advisory team
**Date:** [Tue 17 Nov 2026]
**Re:** Proposed acquisition of Coop Genossenschaft: pre-notification antitrust risk assessment, Swiss grocery retail

---

## 1. The answer

> On public data, we estimate that the merger as structured would raise grocery prices by **[X–Y]%** on average across Switzerland, and by up to **[Z]%** in **[N]** local areas more than **[K]** minutes' drive from the border. That would cost households **CHF [A–B]** per year on average (90% intervals). Cross-border shopping constrains prices only near the border; in the Mittelland and Alpine valleys there is no substitute of comparable reach. Combining Cumulus and Supercard adds a separate, smaller harm of **CHF [C]** per household per year through more precise targeted pricing. It arises only when the data reveal price sensitivity and the rival disappears at the same time. Local screening flags **[M]** of **[M_total]** overlap areas above the 2.75% GUPPI benchmark used by the CMA for supermarkets. A package of **[S]** store divestitures in those areas plus a privacy-protected data ring-fence removes **[R]%** of the estimated consumer harm. We regard that as the minimum credible offer to WEKO. Our measurement pipeline, validated on a simulated replica of the Swiss market, is ready to run on the parties' transaction data; doing so would narrow these ranges before filing.

**What we recommend Migros do now**
1. Treat the transaction as **not clearable unconditionally**. Plan the filing around remedies from day one. *[flip if B-row harm interval includes ~0]*
2. Prepare a divestiture list of **[S]** stores in **[regions]**, with buyers that are credible discounters or regional chains. *[check: does Denner belong in the package?]*
3. Offer **data commitments up front** (ring-fence + audit). They cost little in profit and remove a theory of harm WEKO is likely to pursue.
4. Commission the **day-one data request** in §7, before the first WEKO meeting.

---

## 2. Outcome table: Exhibit 1
**Title (hypothesis):** *"Unremedied, the merger harms households in every scenario we can defend; divestitures plus a data ring-fence remove most of the harm."*

| Scenario | Δ price, national avg | Δ price, > [K] min from border | Δ consumer surplus (CHF/household/yr) | Overlap areas > 2.75% GUPPI | Δ retailer profit (CHF m/yr) | Δ total welfare (CHF m/yr) |
|---|---|---|---|---|---|---|
| A. Status quo | 0 | 0 | 0 | n/a | 0 | 0 |
| B. Merger, no remedy | [ ] | [ ] | [ ] | [M] | [ ] | [ ] |
| C. B + cost efficiencies ([e]% marginal cost) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| D. C + divest flagged stores | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| E. C + data ring-fence | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| F. D + E (recommended package) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| G. Prohibition | 0 | 0 | 0 | n/a | 0 | 0 |

All cells are 90% intervals from [10,000] Monte Carlo draws over the uncertain parameters (Exhibit 6). How each cell is computed:
- Δ consumer surplus is compensating variation from the log-sum: CV_i = [ ln Σ_j e^(V_ij, post) − ln Σ_j e^(V_ij, pre) ] / α_i, averaged over households.
- Δ total welfare = Δ consumer surplus × number of households + Δ retailer profit.

**Source:** `mxc.structural.simulate` → `outputs/tables/outcome_table.csv`.
**If wrong:** if row C's interval includes zero, the headline becomes "efficiencies of [e]% would offset the harm; the case turns on verifying them". The recommendation then shifts to efficiency evidence (DML on the parties' cost data, §7).

---

## 3. Where Migros and Coop compete

### Exhibit 2: diversion
**Title (hypothesis):** *"Coop is the closest substitute for Migros almost everywhere: [D_MC]% of shoppers leaving a Migros store would go to Coop, against [D_disc]% to all discounters combined."*

- **Content:** two panels.
  - **Left:** bar chart of banner-level diversion ratios from Migros, and from Coop, to each rival and to abroad.
  - **Right:** median store-level diversion D(Migros j → nearest Coop k) against the distance between them.
- **Method:**
  - Spatial nested logit [calibrated]:
    ```
    u_ijh = δ_b(j) − α_i × p_b(j) − γ × time_jh + ξ_j + ζ_ig + (1 − σ) × ε_ijh
    ```
    - [real] store locations (OSM) and municipal population and income
    - [calibrated] to national shares and margins
  - Diversion from the model:
    ```
    D(j→k) = −(∂s_k/∂p_j) / (∂s_j/∂p_j)
    ```
- **Key number to watch:** σ, the within-nest correlation. High σ means Migros and Coop are close substitutes.
- **If wrong:** if discounters take more diversion than Coop in most areas, the headline becomes "Denner, Aldi and Lidl now constrain both chains" and the whole risk assessment softens. In that case, say so first.

### Exhibit 3: local screening map
**Title (hypothesis):** *"The risk is local: [M] of [M_total] overlap areas exceed the 2.75% GUPPI benchmark, concentrated in [regions], far from any border."*

- **Content:** choropleth map of Switzerland at municipality level.
  - Colour shows the maximum GUPPI among Migros/Denner–Coop store pairs in the catchment.
  - Canton outlines are drawn on top, and flagged areas are hatched.
  - Inset: histogram of GUPPIs, with lines at 2.75% (supermarkets) and 3.25% (convenience).
- **Method:** prices are not observed, so we use revenue-based identification (Koh 2025). Margins come from annual reports [real, range].
  ```
  GUPPI_j = D_R(j→k) × m_k
  ```
  Here D_R is the revenue diversion ratio and m_k is the percentage margin. Also compute CMCR, the marginal-cost cut that would offset the pressure.
- **Output:** list of flagged areas and the stores in them. This list *is* the divestiture candidate list (Exhibit 8).
- **If wrong:** if almost no areas are flagged, the headline becomes "local overlaps are benign; any concern is national or data-driven". Pillar 3 then carries the memo.

---

## 4. How far the border disciplines prices

### Exhibit 4: border evidence and its reach
**Title (hypothesis):** *"Cross-border shopping caps price increases within ~[K] minutes of the border, and does almost nothing beyond it."*

- **Content:** two panels.
  - **Left [real]:** event study / synthetic DiD of domestic card spending in border vs interior cantons during the **COVID border closure (Mar–Jun 2020)**, when Swiss spending at foreign food retailers fell ~90%. Weekly coefficients with 95% intervals. Source: Monitoring Consumption Switzerland (SIX/Worldline). Robustness: duty-free cut 1 Jan 2025 (difference-in-discontinuities); franc shock 2015 [calibrated, no card data before 2019].
  - **Right [model]:** simulated post-merger price rise by drive time to the nearest border crossing (bins of 10 minutes), with a 90% band.
- **Method:**
  - Outside option, with the response to the price gap β and the decay with distance κ identified from the left panel:
    ```
    u_i0h = δ_0 + β × gap_t − κ × border_time_h + ε_i0h
    ```
  - **Robustness:**
    - diff-in-discontinuities around the duty-free cut (1 Jan 2025)
    - COVID border closure as an out-of-sample check
- **Why it matters to the client:** this is Migros's best argument ("we compete with Konstanz, Lörrach and Annemasse"), and the exhibit shows exactly where it holds and where it does not.
- **If wrong:** if the decay with distance is slow (κ small), the border constrains most of the population. The headline then flips in Migros's favour, and Exhibit 3's flagged set shrinks.

---

## 5. Combining Cumulus and Supercard

### Exhibit 5: the data theory of harm
**Title (hypothesis):** *"Pooling loyalty data adds harm only through the combination of better targeting and a missing rival: about CHF [C] per household per year, which a ring-fence largely removes."*

- **Content:** two panels.
  - **Left:** a 2 × 2 grid of consumer surplus, {uniform, personalised pricing} × {competition, merger}, plus the interaction term:
    ```
    interaction = ΔCS(personalised, merger) − ΔCS(personalised, competition) − ΔCS(uniform, merger)
    ```
  - **Right:** privacy-vs-competition curve: targeting value retained (y) against the differential-privacy level ε of the ring-fence (x).
- **Method [sim]:**
  1. Randomised coupon experiments on the twin's loyalty data.
  2. Causal forest / T-learner to estimate each household's α_i.
  3. Predictive accuracy of one chain's data vs pooled data.
  4. Feed the predictions into the pricing game.
- **Data type matters:** brand-preference data and price-sensitivity data push in opposite directions (Thisse–Vives). Report both.
- **Includes:** WEKO audit design. Paired synthetic shopper profiles, with a power calculation for detecting a CHF [x] targeted price gap.
- **If wrong:** if the interaction is near zero, the headline becomes "data pooling is not a separate harm on these assumptions". The ring-fence is then a low-cost goodwill commitment, not a requirement.

---

## 6. How robust the answer is

### Exhibit 6: sensitivity (tornado)
**Title (hypothesis):** *"The conclusion turns on how close Migros and Coop are (σ) and on border reach (κ); it is robust to cooperative conduct (λμ) and to efficiencies up to [e]%."*

- **Content:** a tornado chart of Δ consumer surplus as each parameter moves across its plausible range. Each bar is labelled with the exact derivative d(harm)/d(parameter), from implicit differentiation through the equilibrium in JAX.
- **Parameters:**
  - σ, nesting
  - α_0 and α_inc, price sensitivity and how it varies with income
  - γ, travel cost
  - β and κ, border response and its decay with distance
  - λμ, cooperative weight
  - e, efficiencies
- **Also shown:**
  - break-even efficiency: the marginal-cost cut at which consumer surplus is unchanged
  - λμ scenarios: survives / dilutes / disappears after the merger

### Opposing-economist table (in text, not an exhibit)
| Their attack | Our test | Result |
|---|---|---|
| "Logit-type demand overstates Migros–Coop closeness" | vary σ over full range; report harm at the lower bound | [ ] |
| "Accounting margins overstate economic margins" | margins as a range; GUPPIs at low end | [ ] |
| "Cooperatives don't maximise profit" | λμ cooperative objective; harm under all three post-merger scenarios | [ ] |
| "Border shopping disciplines everyone" | Exhibit 4 real-data evidence; κ range | [ ] |
| "Discounters will expand" | entry-threshold appendix + twin entry test | [ ] |
| "Efficiencies are large" | break-even efficiency vs plausible synergy range | [ ] |

---

## 7. How we would measure this on the parties' data

### Exhibit 7: digital-twin validation scorecard [sim]
**Title (hypothesis):** *"On a simulated replica of the Swiss market, the ML pipeline recovers true local substitution with [x]× lower error than standard methods; it is ready to run on the parties' data."*

- **Content:** a table. One row per test. Columns: bias, RMSE, 95% coverage, and precision relative to the traditional method.

| Test | Traditional method | Data-driven method | Target (known truth in the twin) |
|---|---|---|---|
| T1 entry | store-month regression | pooled store-by-store event studies vs random controls | local diversion to entrant, by distance |
| T2 demand | OLS | double ML + cost-shock instrument | own-price elasticity |
| T3 local effects | one national elasticity | causal forest | local elasticities → local GUPPIs |
| T4 segmentation *(cut if behind)* | judgement | clustering on loyalty baskets | true segments |
| T5 RDD *(cut if behind)* | — | narrow window around price changes | elasticity |

- **Real geography:** the twin uses real store locations and real municipal population; all sales are simulated. Every number in this section is labelled "method validation on simulated data".
- **Day-one data request** (one paragraph in the memo):
  - store × SKU × week sales
  - loyalty-card trips
  - the parties' own pricing rules and cost data, for DML efficiency verification
  - third-party card-spending data for border flows
- **If wrong:** if ML does not beat the traditional method on a test, say so. That honesty is part of the pitch.

---

## 8. What would make it clearable

### Exhibit 8: remedy packages
**Title (hypothesis):** *"Divesting [S] stores in [N] areas plus a data ring-fence removes [R]% of the harm, at a cost to Migros of CHF [P] m/yr in lost profit."*

- **Content:** a waterfall from the row B harm, through row D (divestitures), row E (data ring-fence) and row F (both), to the residual harm. A side table lists stores divested per region, and whether Denner stores are needed.
- **Method:**
  1. Re-run the equilibrium with divested stores assigned to an independent buyer. Buyer types tested: discounter, regional chain.
  2. Re-run the data game under the differential-privacy ring-fence.
- **If wrong:** if divestitures need more than [threshold]% of Coop's stores to fix the harm, the headline becomes "no proportionate remedy exists in [regions]". The advice then turns to deal structure, e.g. excluding regions.

---

## Technical appendix (planned contents)
- **A1. Model.** Demand, the cooperative objective, the equilibrium algorithm and convergence checks.
- **A2. Data register.** Every input labelled [real], [calibrated], [assumed] or [sim], with its source.
- **A3. Calibration fit.** Model vs published national and canton shares; store counts per canton vs published counts.
- **A4. Validation scorecard.** Full T1–T5 results.
- **A5. Sensitivities and Monte Carlo design.**
- **A6. Extensions.** Supplier bargaining (Nash-in-Nash), entry thresholds; algorithmic pricing (stretch).
- **A7. Sanity checks against closed-form toy models.**
- **A8. Literature and the facts check.** Every [verify] resolved.

---

## Placeholder register (where each number will come from)
| Placeholder | Meaning | Produced by |
|---|---|---|
| X–Y, Z, N, K | avg / max price rise; # areas; border reach | `structural.simulate`, `twin.border` |
| A–B | CV per household per year | `structural.welfare` |
| C | data-pooling harm (interaction) | `datamerger.pricing` |
| M, M_total | flagged / all overlap areas | `structural.screening` |
| S, R, P | stores divested; % harm removed; profit cost | `structural.remedies` |
| D_MC, D_disc | diversion Migros→Coop / →discounters | `structural.demand` |
| e | efficiency scenario | `structural.simulate` |
| x | ML vs traditional error ratio | `causal.scorecard` |

## Open questions this skeleton exposed
1. **Denner.** It is already Migros-owned, so does it count as part of the overlap? It does in the ownership matrix. Should Denner stores be divestiture candidates? Decide after Exhibit 3.
2. **Unit of local market.** Municipality-based catchments for screening, cantons for the maps. Confirm that catchment radius = drive-time cutoff of [10/15] minutes.
3. **Legal test.** Whether the threshold language is dominance or SIEC changes the wording of §1, not the numbers. [verify Cartel Act revision status]
4. **Length.** Eight exhibits fit in 6 pages only if Exhibits 4 and 5 stay at two panels each. If over length, Exhibit 7 moves to the appendix and keeps one line in §7.
