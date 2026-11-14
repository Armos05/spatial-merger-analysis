# Engagement brief (Step 0) — DRAFT for review
*Hypothetical engagement, prepared as a case study. Public data and clearly labelled simulation only.*

## Client and situation
- **Client:** Migros-Genossenschafts-Bund (the acquirer), via its external antitrust counsel (hypothetical).
- **Target:** Coop Genossenschaft, including its supermarkets, Coop Pronto, the Supercard loyalty programme and upstream subsidiaries (e.g. Bell Food Group).
- **Situation:** Migros's board is considering acquiring Coop. The combined group would include Denner (already Migros-owned) and both loyalty programmes (Cumulus + Supercard). No filing has been made. Coop's data is treated as **target data**: what Migros would gain by combining it with Cumulus.
- **Authority:** Swiss Competition Commission (WEKO/COMCO); the merger would be notifiable. Legal standard: dominance-based test under the Cartel Act, with a revision towards the EU-style SIEC test [verify status].

## The question
**"What is the antitrust risk of this merger in Swiss grocery retail, where is it concentrated, and what remedy package would make it clearable?"**

## Decisions this informs
1. Whether Migros should proceed with the acquisition as structured.
2. Which local overlaps need a divestiture plan *before* filing.
3. Whether to offer data commitments (ring-fence / access) up front.
4. Which evidence to request from the parties' own data to strengthen the filing.

## Scope
**In:**
- local price effects
- cross-border constraint
- combined-data theory of harm
- robustness and uncertainty
- remedies
- method readiness for the parties' data

**Appendix only:**
- supplier (buyer-power) effects
- entry by discounters
- algorithmic pricing

**Out:**
- non-grocery businesses (electronics, fuel, travel, banking)
- coordinated effects beyond a short note
- legal opinion

## Deliverables
1. Executive memo (4–6 pages) with a one-paragraph answer, an outcome table and 6–8 exhibits.
2. Technical appendix:
   - model
   - data register
   - validation scorecard
   - sensitivity
3. Reproducible code repository: one command regenerates every number and figure.
4. A one-page web summary.

## Constraints and how we handle them
| Constraint | Handling |
|---|---|
| No access to the parties' transaction data | Public data for geography, shares and margins; a digital twin to validate the methods we would run on client data |
| Local shelf prices not public | Identification from revenues, margins and diversion (Koh 2025) |
| Key parameters uncertain (substitution, cooperative conduct, border response) | Ranges from Monte Carlo; sensitivity ranking; scenarios |
| Hypothetical deal | Neutral tone; label every simulated input |

## What success looks like
- Counsel can say in one sentence where the risk is and what fixes it.
- Every number traces to code, a source or a stated assumption.
- An opposing economist cannot find an unflagged assumption that flips the conclusion.
