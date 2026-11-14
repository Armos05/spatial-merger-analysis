# Spatial merger analysis

A pre-notification antitrust risk assessment for a **hypothetical** Swiss grocery merger: Migros (acquirer) and Coop (target), *"a merger of stores and of shopper data"*. Built on real Swiss geography and public data.

> **Disclaimer.** The transaction is hypothetical. This is an independent case study using public data only. It is not affiliated with, endorsed by, or based on information from Migros, Coop or any consultancy. Results labelled **[sim]** come from simulated data and are for method validation only.

## The question
Can the acquirer buy the target and get it cleared? If so, which local areas raise concerns, and which store and data remedies make the deal approvable?

## Three pillars
1. **Structural merger analysis.** A spatial nested-logit demand model on 32k shopper locations and 4.5k stores. It produces local diversion ratios, GUPPI and CMCR screens, and a merger simulation with consumer harm in CHF per household. Cooperative conduct (a weight on members' surplus) is part of the supply model.
2. **Validation with a digital twin and real border events.** Simulated store × week data with known ground truth tests ML and causal methods (pooled entry analysis, DML, causal forests). Real card-spending data around Swiss border events disciplines cross-border substitution.
3. **Data as an asset.** Combining two loyalty-card datasets: what it adds to price-targeting ability, and whether a differential-privacy remedy removes the harm.

## Repository layout
```
DESIGN.md            design of the analysis
ROADMAP.md           build plan, ticked step by step
DATA_REGISTER.md     every input: source, licence, label ([real] / [calibrated] / [assumed] / [sim])
report/              engagement brief and memo
src/mxc/             package: data, structural, twin, causal, datamerger, extensions
scripts/             numbered pipeline steps; run_all.py rebuilds everything
tests/               unit tests
assets/p1, p2, p3    figures and tables per pillar, named <pillar>_<step>_<name> (e.g. 1_1_reach.png)
outputs/tables/      Phase 1 data tables
```

## Reproduce
```
conda env create -f environment.yml
conda activate keystone
pip install -e .
python scripts/run_all.py     # needs the raw inputs listed in DATA_REGISTER.md under data/raw/
pytest
```
Raw and processed data are not committed. Sources and download locations are in `DATA_REGISTER.md`.

## Data credits
© OpenStreetMap contributors (ODbL); Swiss Federal Statistical Office (STATPOP); Swiss Federal Tax Administration (ESTV); Monitoring Consumption Switzerland (HSG, SIX, Worldline).
