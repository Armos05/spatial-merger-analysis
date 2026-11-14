"""Merger simulation: new national banner prices and consumer harm (Phase 2 step 6; DESIGN §3.6).

Banner demand and its Jacobian come straight from banner shares at each location, so no store
Jacobian is needed. With S_hrB = Σ_{j in B} s_hjr (share of banner B among type r at location h),
m_hr = n_hr × α_r, P_hrg = Σ_{B in g} S_hrB and lam = 1 − σ:

    Q_B        = Σ_hr n_hr S_hrB
    A[B,B']    = dQ_B'/dp_B = Σ_hr m_hr [ S_B S_B' + ((1−lam)/lam) 1{g(B)=g(B')} S_B S_B' / P_g − 1{B=B'} S_B / lam ]

(This is the store Jacobian of substitution.py summed over the stores of B and B'.)

Equilibrium: p = c + m(p),   m(p) = −(Ω ∘ A(p))^(−1) (1 − λμ) Q(p),   solved by damped fixed-point iteration.

Consumer surplus per household and week (one basket a week), type r at location h:
    CS_hr = logsum_hr / α_r       →   ΔCS_hr = (logsum_post − logsum_pre) / α_r
With α independent of income within a type, ΔCS is the exact compensating variation (Small & Rosen 1981).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import demand as dm
from . import supply as su

WEEKS = 52


def make_banner_demand(a: dict):
    """Returns a jitted f(theta) → (Q (B,), A (B,B), logsum (L,R), S (L,R,B), s0 (L,R))."""
    B, G = len(dm.BANNERS), len(dm.NESTS)
    nest_of_b = jnp.array([dm.NESTS.index(dm.BANNER_NEST[b]) for b in dm.BANNERS])
    same_nest = (nest_of_b[:, None] == nest_of_b[None, :]).astype(float)
    L = a["hh"].shape[0]
    key = a["pair_loc"] * B + a["store_banner"][a["pair_store"]]
    n_lr = a["hh"][:, None] * a["type_w"]

    @jax.jit
    def f(theta, a):
        s_pair, s0, logsum = dm.choice_probs(theta, a)
        S = jax.ops.segment_sum(s_pair, key, num_segments=L * B).reshape(L, B, -1).transpose(0, 2, 1)  # (L,R,B)
        alpha = dm.alpha_types(theta, a["type_y"])
        m = n_lr * alpha[None, :]                                                  # (L,R)
        lam = 1.0 - theta.sigma
        Pg = jnp.einsum("lrb,bg->lrg", S, jax.nn.one_hot(nest_of_b, G))
        Pb = jnp.take(Pg, nest_of_b, axis=2)                                        # P(g(B)) per banner
        cross = jnp.einsum("lr,lrb,lrc->bc", m, S, S)
        within = jnp.einsum("lr,lrb,lrc->bc", m / 1.0, S / jnp.where(Pb > 0, Pb, 1.0), S) * same_nest
        own = jnp.einsum("lr,lrb->b", m, S) / lam
        A = cross + ((1 - lam) / lam) * within - jnp.diag(own)
        Q = jnp.einsum("lr,lrb->b", n_lr, S)
        return Q, A, logsum, S, s0

    return lambda theta: f(theta, a)


def solve_equilibrium(theta, banner_demand, cost, lm, owner, p_start=None, damp=0.5, tol=1e-9, max_iter=500):
    """Damped fixed point p ← (1−d) p + d (c + m(p)). Returns prices and an iteration history."""
    p = np.asarray(theta.p if p_start is None else p_start, dtype=float)
    hist = []
    for it in range(max_iter):
        Q, A, *_ = banner_demand(theta._replace(p=jnp.asarray(p)))
        m = su.markups(np.asarray(A), np.asarray(Q), owner, lm)
        target = cost + m
        step = np.abs(target - p).max()
        hist.append({"iter": it, "max_abs_price_gap": step, **dict(zip(dm.BANNERS, p))})
        if step < tol:
            break
        p = (1 - damp) * p + damp * target
    return p, pd.DataFrame(hist)


def delta_cs(theta_pre, theta_post, a) -> np.ndarray:
    """(L, R) change in consumer surplus, CHF per household per week."""
    _, _, ls0 = dm.choice_probs(theta_pre, a)
    _, _, ls1 = dm.choice_probs(theta_post, a)
    alpha = dm.alpha_types(theta_pre, a["type_y"])
    return np.asarray((ls1 - ls0) / alpha[None, :])
