# Summary — "The Next Frontier in Merger Review" (Keystone, April 2024)
Bajari (Keystone Chief Economist, ex-Amazon VP), Calanchi (ex-Compass Lexecon), Akati-Udi. 12 pages.

## Main argument
- Competition authorities are moving in two directions at once: more data science (DOJ under Susan Athey, the UK CMA's DaTA unit, the EU's new data unit), **and** more weight on internal documents and qualitative evidence.
- Both trends have one cause: doubt about "traditional" merger econometrics, whose predictions often look poor after the fact (Kwoka 2013), and a wish to see a **"business perspective"**.
- Keystone's answer is to analyse a merger **the way the firms themselves analyse their business**: huge granular data, machine learning (ML) and causal-inference tools like those Big Tech uses. This applies even in traditional sectors such as **retail**.
- To do this, a consulting team needs three things: (1) Big-Tech-level data engineering (cloud, terabytes), (2) "Big-Tech econometrics" (ML with thousands of controls), (3) deep knowledge of company and third-party datasets.

## Group 1 — upgrading the four traditional tools
- **Entry analysis** (how incumbent A's sales drop when rival B opens nearby → closeness of competition, diversion):
  - Use weekly or daily, SKU × store data instead of monthly store totals → precise estimates even when the effect is subtle.
  - Use third-party **card-transaction** and **mobile-location** data to follow individual shoppers across stores.
  - Estimate the effect store by store against random control stores, then pool → precise network-level **and local** diversion ratios (how the effect fades with distance).
  - Noise in traditional estimates tends to be **biased against the merging parties**: an insignificant result in a competitive market carries little weight.
- **Demand estimation**: the core risk is endogeneity (prices respond to demand). Fixes: control for thousands of variables with ML; "trade quantity for quality" by comparing purchases **seconds before and after** a price change (a regression discontinuity design, RDD); estimate elasticities store by store and relate them to distance to rivals.
- **Merger simulation**: standard models give a static, simplistic picture of price-setting. Real firms (e.g. Walmart) use **pricing algorithms** that react to rivals, so estimate those **best-response functions** and build them into the simulation. Product attributes come from ML (images, review text).
- **Efficiencies**: they must be verifiable. Use the firms' own supply-chain and forecasting models and **double machine learning** to measure how marginal costs (transport, inventory) fall when sales are combined.

## Group 2 — measuring substitution the Big Tech way
- **Search-ranking changes**: when a product drops from #1 to #3, do buyers follow it or switch to the new #1? That reveals substitution.
- **ML customer segmentation** (supervised and unsupervised) instead of drawing segment lines by judgement.
- **Merger effects by customer group**: use AI on granular consumption data (reviews, purchase histories) to predict who is harmed or helped.

## Conclusion
- To persuade an authority with data, advisers will need "Big-Tech-like" analysis that shows substitution the way businesses actually see it.

## What this means for our Migros × Coop case
- **We match their thesis already:** local diversion ratios with distance fade-out (H3 geo layer), an RDD in time (duty-free cut), shopper heterogeneity and loyalty data (Module 5), a reproducible computational pipeline (JAX).
- **Borrow their language:** "business perspective", "localized diversion", "Big-Tech econometrics", "trade quantity for quality".
- **Worth adding if time allows:**
  - one paragraph on what **card-transaction data** would add in a real engagement (Swiss card-spending data, Module 4.8)
  - a short note on **algorithmic pricing / best-response estimation** (connects to stretch item E, Q-learning collusion)
  - **double ML** (DML) as the route to verifying efficiencies
- **Framing:** Keystone often advises **merging parties** (Microsoft/Activision, Roark/Subway), so the memo should be even-handed: show where harm is robust and where it is not (border regions, efficiencies, entry).
