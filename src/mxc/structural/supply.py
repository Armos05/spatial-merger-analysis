"""Supply: cooperative conduct and cost recovery (Phase 2 step 4; DESIGN §3.3; design_dummy.md Step 5).

Swiss grocers price nationally: one basket price per banner. Firm f sets the prices of its banners to maximise

    W_f = Σ_{B in f} (p_B − c_B) Q_B  +  λμ_f × CS,          with dCS/dp_B = −Q_B  (Roy's identity)

First-order condition for banner B (A[B,B'] = Σ_{j∈B} Σ_{k∈B'} dq_k/dp_j, the banner Jacobian):

    (1 − λμ_f) Q_B + Σ_B' Omega[B,B'] × (p_B' − c_B') × A[B,B'] = 0

Omega[B,B'] = 1 if B and B' have the same owner. In matrix form:

    m = p − c = −(Omega ∘ A)^(−1) × (1 − λμ) Q

Omega is block-diagonal by firm, so each firm's markups are (1 − λμ_f) × its profit-maximising markups.
"""
from __future__ import annotations

import numpy as np

from . import demand as dm

# Pre-merger owners of the 12 banners. fenaco (Volg, Landi) is an agricultural cooperative, but its
# member weight is set to 0 [assumed]: its members are farmers, not shoppers.
OWNER_PRE = {
    "Migros": "Migros group", "Migros Partner": "Migros group", "Migrolino": "Migros group", "Denner": "Migros group",
    "Coop": "Coop group", "Coop Pronto": "Coop group",
    "Aldi": "Aldi", "Lidl": "Lidl", "Volg": "fenaco", "Landi": "fenaco", "Spar": "Spar", "Manor": "Manor",
}
OWNER_POST = {b: ("Migros-Coop" if o in ("Migros group", "Coop group") else o) for b, o in OWNER_PRE.items()}


def omega(owner: dict) -> np.ndarray:
    o = np.array([owner[b] for b in dm.BANNERS])
    return (o[:, None] == o[None, :]).astype(float)


def markups(A: np.ndarray, Q: np.ndarray, owner: dict, lm: np.ndarray) -> np.ndarray:
    """(B,) CHF markups that satisfy the FOCs: m = −(Omega ∘ A)^(−1) (1 − λμ) Q."""
    return np.linalg.solve(omega(owner) * A, -(1 - lm) * Q)


def foc_residual(A, Q, owner, lm, m) -> np.ndarray:
    return (1 - lm) * Q + (omega(owner) * A) @ m


def lambda_from_margin(m0_B: float, p_B: float, margin: float) -> float:
    """observed markup = (1 − λμ) × profit-max markup  →  λμ = 1 − margin × p / m0."""
    return 1.0 - margin * p_B / m0_B


def local_margin(Jac, q_store, store_banner, store_owner, m_banner, lm_store) -> np.ndarray:
    """Margin a store would pick if it alone could set its own price, given every other price.

    (1 − λμ) q_j + m_j Jac[j,j] + Σ_{k sister of j, k≠j} m_k Jac[j,k] = 0
    →  m_j = −[ (1 − λμ) q_j + Σ_{sisters} m_k Jac[j,k] ] / Jac[j,j]
    Sisters keep their banner's national markup. This shows where national pricing holds local prices down.
    """
    J = Jac.shape[0]
    mk = m_banner[store_banner]
    own = store_owner
    coo = Jac.tocoo()
    rows, cols, vals = coo.row, coo.col, coo.data
    sister = (own[rows] == own[cols]) & (rows != cols)
    recapture = np.bincount(rows[sister], weights=vals[sister] * mk[cols[sister]], minlength=J)
    diag = Jac.diagonal()
    with np.errstate(divide="ignore", invalid="ignore"):
        return -((1 - lm_store) * q_store + recapture) / diag
