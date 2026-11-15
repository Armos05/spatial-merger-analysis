# Appendix: extensions not built into the main model

Two channels that merger guidelines ask about are discussed here in stylised form rather than estimated:
suppliers (buyer power) and entry. Both come from small three-firm logit models (Migros, Coop, Aldi plus an
outside option, calibrated to the same prices and margins as Pillar 1). They are sanity checks, not estimates.

## A1. Suppliers: does the merged firm squeeze them? (Nash-in-Nash bargaining)

**Set-up.** One branded supplier sells through Migros and Coop. Each retailer–supplier pair bargains over the
wholesale price w, taking the other pair's deal as given (Nash-in-Nash; Crawford, Lee, Whinston & Yurukoglu,
Econometrica 2018). Bargaining weight 0.5. If talks fail, the product is delisted at that retailer.

**Mechanism.**
- Before the merger, a delisted brand keeps part of its sales, because brand-loyal shoppers move to the other
  chain. The second buyer protects the supplier: if Migros delists, supplier profit falls 4.65 → 2.61 per shopper.
- After the merger there is no second buyer, so failure means zero sales.
- But the merged retailer also loses more from failure (it cannot keep brand-loyal shoppers "in the family"), so
  bargaining leverage barely moves.

**Result (toy).** Supplier profit falls 22.0%:
- volume: (3.648 − 4.645) / 4.645 = −21.5%, because retail prices rise and fewer baskets are sold;
- wholesale price: (3.623 − 3.648) / 4.645 = −0.5%.

The split holds for brand values of CHF 10 to 40 per basket (−22.0% to −20.9%).

**What would create real buyer power** (to test with data the case does not have):
1. supplier survival threats: Swiss dairies or meat processors with fixed costs and ~75% of volume through one
   buyer;
2. a stronger private-label outside option (Migros and Coop own processors);
3. a bargaining weight that rises with buyer size.

**Pass-through check from Pillar 1.** In row C, the 5% cost saving lowers the Migros cost by
0.05 × 75.15 = CHF 3.76. The post-merger Migros price changes from 118.03 to 115.75 (−2.27). Pass-through
= 2.27 / 3.76 = 60%. Even if suppliers were squeezed, at most around half to two thirds of the saving would
reach shoppers.

**Memo line.** Suppliers lose about a fifth of their margin income, mostly because the merger shrinks retail
volume. A wholesale squeeze needs supplier distress, which WEKO should test with supplier cost data.

## A2. Entry: will Aldi and Lidl undo the harm? (Bresnahan–Reiss revealed preference)

**Set-up.**
- In one regional market (10,000 households, 52 baskets a year), Aldi's quality rises with its store count:
  δ(n) = δ(2) + 0.5 × ln(n/2).
- Today Aldi runs 2 stores. Opening the 2nd store paid off but a 3rd would not, so the annual fixed cost of a
  store F lies between CHF 306,819 and 441,394 (revealed preference; Bresnahan & Reiss 1991).

**Result (toy).** After the merger, discounter profits rise and Aldi opens stores while the marginal store
still covers F:

| F | Aldi stores after | Harm per basket (no entry → entry) | Harm offset |
|---|---|---|---|
| upper bound 441,394 | 2 | −11.50 → −11.50 | 0% |
| midpoint 374,107 | 3 | −11.50 → −10.04 | 13% |
| lower bound 306,819 | 4 | −11.50 → −8.97 | 22% |

**Link to the main model.**
- In Pillar 1 (row B), Aldi and Lidl gain +49% and +47% in baskets **without** opening a single store.
  The diversion goes to their existing stores.
- The twin (step 2.3) shows how WEKO could measure entry effects from store-level data: a pooled exposure
  regression recovers the loss of stores within 3 km of a new discounter with bias +0.13 pp. A classic
  store-month TWFE regression has bias +1.15 pp.

**Memo line.** Discounter expansion would offset at most about a fifth of the price rise, and only after the
years that Swiss zoning and land scarcity take. Entry is likely in part, but neither timely nor sufficient.

## A3. Not done (stretch items)
- Algorithmic pricing (Q-learning agents, Calvano et al. AER 2020): would learned post-merger prices exceed the
  Nash prices used here? Not built.
- Product-range overlap from Open Food Facts embeddings (to inform σ): not built. σ is instead treated as the
  main uncertainty in the sensitivity and Monte Carlo work (assets/p1/m9, m10).
