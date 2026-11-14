"""Store-level Jacobian, diversion ratios and elasticities (Phase 2 step 3; design_dummy.md Step 4-5).

For households of type r at location h, write m_hr = n_hr × alpha_r (households × price sensitivity).
Nested-logit derivatives of s_k with respect to V_j at one location (lam = 1 - sigma):

    own:          ds_j/dV_j = s_j [ 1/lam − ((1−lam)/lam) P(j|g) − s_j ]
    same nest:    ds_k/dV_j = − s_k [ ((1−lam)/lam) P(j|g) + s_j ]
    other nest:   ds_k/dV_j = − s_k s_j

Since dV_j/dp_j = −alpha_r, summing over locations and types gives the store Jacobian
dq_k/dp_j = Σ_hr m_hr × (− ds_k/dV_j), which splits into three sparse pieces:

    Jac = C + ((1−lam)/lam) × N − diag( Σ_hr m_hr s_j / lam )
    C[j,k] = Σ_hr m_hr s_j s_k                       (every pair at the same location)
    N[j,k] = Σ_hr m_hr s_j s_k / P(g)                (only pairs in the same nest g; P(j|g) = s_j / P(g))

Check: the diagonal of C + ((1−lam)/lam) N is Σ m s_j^2 + ((1−lam)/lam) Σ m s_j P(j|g), which is exactly the
own-derivative formula. Jac is symmetric, and each row plus its abroad entry sums to zero.

Diversion:  D(j→k) = Jac[j,k] / (−Jac[j,j]).   Group price rise: add up the rows of the group's stores.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import demand as dm


def jacobian(theta: dm.Params, a: dict, alpha_lr=None):
    """Returns (Jac (J×J) sparse CSR, dq_abroad/dp_j (J,), q_store (J,)). Units: baskets per CHF."""
    s_pair, s0, _ = dm.choice_probs(theta, a, alpha_lr)
    s_pair, s0 = np.asarray(s_pair), np.asarray(s0)
    L, R = s0.shape
    J = int(a["store_banner"].shape[0])
    G = len(dm.NESTS)
    lam = 1.0 - float(theta.sigma)
    if alpha_lr is None:
        alpha_lr = np.broadcast_to(np.asarray(dm.alpha_types(theta, a["type_y"])), (L, R))
    alpha_lr = np.asarray(alpha_lr)
    n_lr = np.asarray(a["hh"])[:, None] * np.asarray(a["type_w"])
    m = n_lr * alpha_lr                                                # (L, R)
    pl, ps, seg = (np.asarray(a[k]) for k in ("pair_loc", "pair_store", "pair_seg"))
    P = len(pl)

    # P(g) per (location, nest, type): sum of s over the nest's pairs
    Pg = np.zeros((L * G, R))
    np.add.at(Pg, seg, s_pair)

    # rows = (location, type) for C, (location, nest, type) for N; columns = store
    rows_c = (pl[:, None] * R + np.arange(R)[None, :]).ravel()
    rows_n = (seg[:, None] * R + np.arange(R)[None, :]).ravel()
    cols = np.repeat(ps, R)
    vals = s_pair.ravel()
    S = sp.csr_matrix((vals, (rows_c, cols)), shape=(L * R, J))
    T = sp.csr_matrix((vals, (rows_n, cols)), shape=(L * G * R, J))
    w_c = m.ravel()
    w_n = (m[seg // G] / np.where(Pg[seg] > 0, Pg[seg], 1.0))           # (P, R) weight m / P(g) per pair-row
    wn_full = np.zeros(L * G * R)
    wn_full[rows_n] = w_n.ravel()

    C = (S.T @ sp.diags(w_c) @ S).tocsr()
    N = (T.T @ sp.diags(wn_full) @ T).tocsr()
    own = np.bincount(np.repeat(ps, R), weights=(m[pl] * s_pair).ravel(), minlength=J) / lam
    Jac = (C + ((1 - lam) / lam) * N - sp.diags(own)).tocsr()

    d_abroad = S.T @ (w_c * s0.ravel())                                # dq_0/dp_j = Σ m s_0 s_j
    q_store = np.bincount(np.repeat(ps, R), weights=(n_lr[pl] * s_pair).ravel(), minlength=J)
    assert P == s_pair.shape[0]
    return Jac, np.asarray(d_abroad).ravel(), q_store


def group_diversion(Jac, d_abroad, groups_of_store: np.ndarray, n_groups: int):
    """Diversion when every store of group A raises its price by CHF 1, to group B (and abroad).

    D(A→B) = Σ_{j∈A} Σ_{k∈B} Jac[j,k] / (− Σ_{j∈A} Σ_{k∈A} Jac[j,k]).
    Returns an (n_groups, n_groups + 1) array; the last column is abroad; the diagonal is NaN.
    """
    J = Jac.shape[0]
    Gm = sp.csr_matrix((np.ones(J), (np.arange(J), groups_of_store)), shape=(J, n_groups))
    agg = (Gm.T @ Jac @ Gm).toarray()                                  # (A, B) = Σ_{j∈A,k∈B} dq_k/dp_j
    ab = Gm.T @ d_abroad
    lost = -np.diag(agg)
    D = np.c_[agg, ab] / lost[:, None]
    D[np.arange(n_groups), np.arange(n_groups)] = np.nan
    return D, agg, lost
