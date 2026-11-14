# References

## 1. Ellickson, Grieco & Khvastunov — "Measuring Competition in Spatial Retail"
- File: `Ellickson_Grieco_Khvastunov_2020_Measuring_Competition_Spatial_Retail_preprint.pdf` (preprint, 26 Mar 2019, 62 pp.)
- Published: RAND Journal of Economics 51(1), 2020, pp. 189–232, doi:10.1111/1756-2171.12310 — check page numbers against the published version before citing.
- Source: authors' site, https://paulgrieco.github.io/files/preprint/groceries.pdf
- What it does: nested-logit store choice aggregated over census tracts; utility depends on distance, car ownership, income; chain fixed effects instead of prices; store- and chain-level diversion ratios; applied to US grocery and two mergers (Whole Foods/Wild Oats, Delhaize/Ahold).
- Use for: §3 demand (distance term, formats as nests), diversion ratios, how to do it without price data.
- Gap we fill: they explicitly do **not** estimate a supply model, so no full merger simulation (p. ~7) — our case adds supply (incl. cooperative objective).
- Read first: Abstract + Introduction (pp. 1–6), model section, Section 6 (mergers).

## 2. UK Competition and Markets Authority — Sainsbury's / Asda, Final Report (25 Apr 2019)
- File: `CMA_2019_Sainsburys_Asda_Final_Report.pdf` (542 pp.)
- Source: gov.uk, https://assets.publishing.service.gov.uk/media/5cc1ec1340f0b64031cfa6f0/Final_reportSA.pdf
- What it is: the real regulatory review of a merger between the UK's 2nd and 3rd largest grocers; **prohibited**. Andrea Coscelli (now Keystone Senior Partner) was CMA Chief Executive at the time.
- Methods: local GUPPIs with a decision rule (supermarket threshold **2.75%**, convenience stores **3.25%**, thresholds include an allowance for efficiencies); diversion from a 20,500-respondent **store exit survey**; weighted-share (WSS) models; national assessment; buyer power (ch. 15); efficiencies (ch. 16); remedies.
- Note: Sainsbury's owned the **Nectar** loyalty programme (acquired Feb 2018) — a direct parallel to Cumulus/Supercard.
- Use for: the structure and language of the memo, GUPPI decision rules, how buyer power, efficiencies and remedies are argued.
- Read first: Summary (pp. 6–31), then ch. 8 (in-store unilateral effects), 15 (buyer power), 16 (efficiencies).

## 3. Keystone — "The Next Frontier in Merger Review" (April 2024)
- File: `Keystone_2024_Next_Frontier_in_Merger_Review.pdf` (12 pp.)
- Authors: Patrick Bajari, Gianmarco Calanchi, Tega Akati-Udi (Keystone). Subtitle: "How to Use AI and Other Big Tech Techniques to Assess Horizontal Mergers – Even in Traditional Sectors".
- Source: https://www.keystone.com/news-publications/the-next-frontier-in-merger-review
- Use for: Keystone's own view of ML/AI in merger review — mirror its vocabulary and framing in the memo and the CS layer (DESIGN §14b). Read first, whole thing.

## 4. Koh — "Merger Analysis with Unobserved Prices" (arXiv v6, 30 Aug 2025)
- File: `Koh_2025_Merger_Analysis_with_Unobserved_Prices_arXiv_v6.pdf` (59 pp.)
- Earlier title "Merger Analysis with Latent Price"; FTC Bureau of Economics Working Paper No. 350 (Feb 2025). arXiv: https://arxiv.org/abs/2404.07684 (FTC PDF link blocked automated download).
- What it does: GUPPIs, compensating marginal cost reductions and merger simulation **without price data**, from revenues, margins and revenue diversion ratios; applied to Albertsons/Safeway (2015, grocery) and Staples/Office Depot (2016). Cites EGK.
- Use for: cutting-edge reference for grocery merger simulation when shelf prices are unobserved — our exact data situation (Swiss basket prices are not public).
- Read first: Abstract, Section 1, the Albertsons/Safeway application.

## Not downloaded (no PDF available)
- Coscelli, Chissell, Bagaria & Akati-Udi (Keystone), "Generative AI and Competitive Disruption: Increasingly Relevant for Merger Analysis?", Feb 2026; also CPI Antitrust Chronicle, 28 Apr 2026. Web only: https://www.keystone.com/news-publications/generative-ai-and-competitive-disruption-increasingly-relevant-for-merger-analysis

## Also worth knowing
- Hosken, Olson & Smith, "Do retail mergers affect competition? Evidence from grocery retailing", JEMS 27(1), 2018 — merger **retrospective** using DiD and synthetic control on 14 US grocery mergers. Free FTC working-paper version: https://www.ftc.gov/sites/default/files/attachments/guarding-consumers-pocketbooks/wp313.pdf
