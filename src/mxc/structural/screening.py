"""Merger screens without simulation: GUPPI and CMCR (Phase 2 step 5; DESIGN §3.4–3.5).

GUPPI (Farrell & Shapiro 2010; Moresi 2010). For store j of one party, after the merger:

    GUPPI_j = Σ_{k in other party} D(j→k) × (p_k − c_k) / p_j
            = Σ_{k in other party} DR(j→k) × μ_k              (Koh 2025: no price levels needed)

    D(j→k)  = Jac[j,k] / (−Jac[j,j])                  diversion (baskets)
    DR(j→k) = D(j→k) × p_k / p_j                      revenue diversion
    μ_k     = (p_k − c_k) / p_k                       margin (annual reports)

With the cooperative objective, the new term in store j's FOC acts exactly like a cost increase of
Σ D(j→k) m_k, so the GUPPI formula is unchanged when the actual (cooperative) margins are used.

CMCR (Werden 1996): the cost cut that keeps today's prices optimal after the merger.
    National (all banners re-optimise together):  c* = p − m_post,  m_post = −(Ω_post ∘ A)^(−1) (1 − λμ) Q
                                                 CMCR_B = (c_B − c*_B) / c_B
    Store (one-sided; the store alone):          CMCR_j = Σ_k D(j→k) m_k / c_j = GUPPI_j × p_j / c_j
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import demand as dm
from . import supply as su

THRESHOLD = {"supermarket": 0.0275, "convenience": 0.0325}   # CMA, Sainsbury's/Asda (2019) [verify]


def store_guppi(Jac, store_party, markup_store, price_store, party_a="Migros group", party_b="Coop group"):
    """(J,) GUPPI per store of either merging party (NaN for other stores), and the diversion to the other party."""
    other = np.where(store_party == party_a, party_b, np.where(store_party == party_b, party_a, ""))
    J = Jac.shape[0]
    out_g, out_d = np.full(J, np.nan), np.full(J, np.nan)
    own = -Jac.diagonal()
    for src, dst in [(party_a, party_b), (party_b, party_a)]:
        ind = (store_party == dst).astype(float)
        sel = store_party == src
        to_dst = Jac @ ind                                 # Σ_{k in dst} dq_k/dp_j
        val = Jac @ (ind * markup_store)                   # Σ_{k in dst} dq_k/dp_j × m_k
        out_d[sel] = to_dst[sel] / own[sel]
        out_g[sel] = val[sel] / own[sel] / price_store[sel]
    assert (other != "").sum() == np.isfinite(out_g).sum()
    return out_g, out_d


def national_cmcr(A, Q, lm, price, cost):
    """Banner CMCR with joint re-optimisation under post-merger ownership, at today's prices."""
    m_post = su.markups(A, Q, su.OWNER_POST, lm)
    c_star = price - m_post
    return (cost - c_star) / cost, m_post


def banner_guppi(A, price, markup):
    """National GUPPI per banner: Σ_{B' of the other party} D(B→B') m_B' / p_B, uniform banner price rise."""
    party = np.array([su.OWNER_PRE[b] for b in dm.BANNERS])
    D = A / (-np.diag(A))[:, None]
    out = np.full(len(price), np.nan)
    for a_, b_ in [("Migros group", "Coop group"), ("Coop group", "Migros group")]:
        src, dst = party == a_, party == b_
        out[src] = (D[np.ix_(src, dst)] * markup[dst][None, :]).sum(1) / price[src]
    return out


def sparse_rows_sum(M: sp.csr_matrix, mask: np.ndarray) -> np.ndarray:
    return np.asarray(M[mask].sum(0)).ravel()
