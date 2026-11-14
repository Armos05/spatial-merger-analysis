"""Pillar 3: data as an asset — personalised prices on the loyalty panel (DESIGN §5). [sim]

Each household i of the 2.2 panel faces the Pillar 1 nested logit with its own price sensitivity α_i, segment
tastes and travel costs. A card programme can send household i a personal coupon: a multiplier m_fi on the prices
of firm f's card banners (Cumulus: Migros, Migros Partner, Migrolino; Supercard: Coop, Coop Pronto), only for
households holding that card. Every other price stays uniform (Pillar 1 equilibrium).

Per choice occasion, firm f's objective for household i (cooperative conduct as in Pillar 1):
    W_fi = Σ_{pairs j of f} s_ij (p_ij − c_j)  +  λμ × CS_i,       CS_i = logsum_i / α_i
Firms do not know α_i, the segment tastes or the household's travel-cost multipliers. They price with a belief:
the same model with α̂_i (from their data) and average tastes. Households respond with the truth.

    competition: Nash in (m_M, m_C), each firm with its own α̂ (Cumulus or Supercard data)
    merger:      one firm picks (m_M, m_C) jointly to maximise W_M + W_C with pooled α̂

The problem is separable across households, so grad of Σ_i W_fi gives every household's derivative at once;
Newton steps use the 2×2 Hessian per household (jvp of the gradient).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..structural import demand as dm
from ..structural.demand import _segment_logsumexp
from ..twin import loyalty as ly

MIGROS_GROUP = ("Migros", "Migros Partner", "Migrolino", "Denner")
COOP_GROUP = ("Coop", "Coop Pronto")


def panel_arrays(a, hh: pd.DataFrame, xi_store, informed=True) -> dict:
    """Household-level market. informed=False is the firm's view: no segment tastes, average travel costs."""
    h = hh if informed else hh.assign(gamma_mult=1.0, kappa_mult=1.0, segment="mainstream")
    ah = dict(ly.household_arrays(a, h))
    sb = np.asarray(a["store_banner"])
    ps = np.asarray(ah["pair_store"])
    hp = np.asarray(ah["pair_loc"])
    ban = np.array(dm.BANNERS)[sb[ps]]
    ah["is_M"] = jnp.asarray(np.isin(ban, ly.CUMULUS) & hh.cumulus.to_numpy()[hp])
    ah["is_C"] = jnp.asarray(np.isin(ban, ly.SUPERCARD) & hh.supercard.to_numpy()[hp])
    ah["own_M"] = jnp.asarray(np.isin(ban, MIGROS_GROUP))
    ah["own_C"] = jnp.asarray(np.isin(ban, COOP_GROUP))
    ah["xi_pair"] = jnp.asarray(np.asarray(xi_store)[ps])
    ah["banner_pair"] = jnp.asarray(sb[ps])
    return ah


def hh_logit(theta, ah, price_pair, alpha_h):
    """Nested logit with a price per (household, store) pair. Returns s_pair (P,), s_abroad (H,), logsum (H,)."""
    H, G = alpha_h.shape[0], len(dm.NESTS)
    lam = 1.0 - theta.sigma
    loc = ah["pair_loc"]
    V = theta.delta[ah["banner_pair"]] + ah["xi_pair"] + ah["pair_xi"] - alpha_h[loc] * price_pair \
        - theta.gamma * ah["pair_time"]
    V0 = -alpha_h * theta.p_abroad - theta.kappa * ah["border_min"]
    lnD = _segment_logsumexp((V / lam)[:, None], ah["pair_seg"], H * G)[:, 0]
    I = lam * lnD.reshape(H, G)
    logsum = jax.scipy.special.logsumexp(jnp.concatenate([V0[:, None], I], 1), 1)
    s = jnp.exp(I.reshape(-1)[ah["pair_seg"]] - logsum[loc] + V / lam - lnD[ah["pair_seg"]])
    return s, jnp.exp(V0 - logsum), logsum


def pair_prices(ah, base_banner, mM, mC):
    loc = ah["pair_loc"]
    p = base_banner[ah["banner_pair"]]
    return p * jnp.where(ah["is_M"], mM[loc], 1.0) * jnp.where(ah["is_C"], mC[loc], 1.0)


def objectives(theta, ah, base, cost, lm, alpha_h, mM, mC):
    """(W_M, W_C, CS) per household for multipliers mM, mC (H,). cost, lm: (B,) per banner."""
    H = alpha_h.shape[0]
    p = pair_prices(ah, base, mM, mC)
    s, _, logsum = hh_logit(theta, ah, p, alpha_h)
    marg = s * (p - cost[ah["banner_pair"]])
    loc = ah["pair_loc"]
    piM = jax.ops.segment_sum(jnp.where(ah["own_M"], marg, 0.0), loc, num_segments=H)
    piC = jax.ops.segment_sum(jnp.where(ah["own_C"], marg, 0.0), loc, num_segments=H)
    cs = logsum / alpha_h
    lmM, lmC = lm[dm.BANNERS.index("Migros")], lm[dm.BANNERS.index("Coop")]
    return piM + lmM * cs, piC + lmC * cs, cs, piM, piC


def chain_share(theta, ah, base, alpha_h, mM, mC, side):
    p = pair_prices(ah, base, mM, mC)
    s, _, _ = hh_logit(theta, ah, p, alpha_h)
    w = ah["is_M"] if side == "M" else ah["is_C"]
    return jax.ops.segment_sum(jnp.where(w, s, 0.0), ah["pair_loc"], num_segments=alpha_h.shape[0])


# ----------------------------------------------------------------------------- equilibrium
def solve(theta, ah, base, cost, lm, alpha_M, alpha_C, merged: bool, iters=200, step_max=0.05,
          lo=0.6, hi=2.0, tol=1e-7):
    """Personalised multipliers. Competition: firm M believes alpha_M, firm C believes alpha_C (Nash).
    Merger: one firm believes alpha_M (= alpha_C = pooled) and maximises W_M + W_C.
    Returns mM, mC, max |gradient| at the end."""
    H = alpha_M.shape[0]
    hasM = np.asarray(jax.ops.segment_sum(ah["is_M"].astype(float), ah["pair_loc"], num_segments=H)) > 0
    hasC = np.asarray(jax.ops.segment_sum(ah["is_C"].astype(float), ah["pair_loc"], num_segments=H)) > 0

    def tot(mM, mC, alpha, who):
        WM, WC, cs, piM, piC = objectives(theta, ah, base, cost, lm, alpha, mM, mC)
        lmJ = lm[dm.BANNERS.index("Migros")]              # merged cooperative: member weight counted once
        return {"M": WM, "C": WC, "J": piM + piC + lmJ * cs}[who].sum()

    def grads(mM, mC, alpha, who):
        g = jax.grad(tot, argnums=(0, 1))
        gM, gC = g(mM, mC, alpha, who)
        one = jnp.ones_like(mM)
        (_, _), (hMM, hCM) = jax.jvp(lambda x: g(x, mC, alpha, who), (mM,), (one,))
        (_, _), (hMC, hCC) = jax.jvp(lambda x: g(mM, x, alpha, who), (mC,), (one,))
        return gM, gC, hMM, hMC, hCM, hCC

    gfun = jax.jit(grads, static_argnums=3)
    mM, mC = jnp.ones(H), jnp.ones(H)
    hM, hC = jnp.asarray(hasM), jnp.asarray(hasC)
    for it in range(iters):
        if merged:
            gM, gC, hMM, hMC, hCM, hCC = gfun(mM, mC, alpha_M, "J")
            hMM = jnp.where(hM, hMM, -1.0)
            hCC = jnp.where(hC, hCC, -1.0)
            hMC = jnp.where(hM & hC, hMC, 0.0)
            det = hMM * hCC - hMC * hMC
            ok = (hMM < 0) & (det > 0)
            dM = jnp.where(ok, -(hCC * gM - hMC * gC) / det, 0.01 * jnp.sign(gM))
            dC = jnp.where(ok, -(-hMC * gM + hMM * gC) / det, 0.01 * jnp.sign(gC))
            gmax = float(jnp.max(jnp.abs(jnp.where(hM, gM, 0))) + jnp.max(jnp.abs(jnp.where(hC, gC, 0))))
        else:
            gM, _, hMM, _, _, _ = gfun(mM, mC, alpha_M, "M")
            _, gC, _, _, _, hCC = gfun(mM, mC, alpha_C, "C")
            dM = jnp.where(hMM < 0, -gM / jnp.where(hMM < 0, hMM, -1.0), 0.01 * jnp.sign(gM))
            dC = jnp.where(hCC < 0, -gC / jnp.where(hCC < 0, hCC, -1.0), 0.01 * jnp.sign(gC))
            gmax = float(jnp.max(jnp.abs(jnp.where(hM, gM, 0))) + jnp.max(jnp.abs(jnp.where(hC, gC, 0))))
        dM = jnp.where(hM, jnp.clip(dM, -step_max, step_max), 0.0)
        dC = jnp.where(hC, jnp.clip(dC, -step_max, step_max), 0.0)
        mM, mC = jnp.clip(mM + dM, lo, hi), jnp.clip(mC + dC, lo, hi)
        if gmax < tol:
            break
    return np.asarray(mM), np.asarray(mC), gmax, it + 1


def evaluate(theta, ah_true, base, cost, lm, alpha_true, mM, mC):
    """Truth: CS (CHF per choice occasion), profits and chain shares per household."""
    _, _, cs, piM, piC = objectives(theta, ah_true, base, cost, lm, alpha_true, jnp.asarray(mM), jnp.asarray(mC))
    p = pair_prices(ah_true, base, jnp.asarray(mM), jnp.asarray(mC))
    s, s0, _ = hh_logit(theta, ah_true, p, alpha_true)
    H = alpha_true.shape[0]
    paid = jax.ops.segment_sum(s * p, ah_true["pair_loc"], num_segments=H) + s0 * theta.p_abroad
    return pd.DataFrame({"cs": np.asarray(cs), "profit_M": np.asarray(piM), "profit_C": np.asarray(piC),
                         "avg_price_paid": np.asarray(paid)})


# ----------------------------------------------------------------------------- experiment → α̂
def invert_alpha(theta, ah_belief, base, tau, side, discount=0.10, lo=0.002, hi=0.25, iters=40):
    """α̂ that makes the firm's model reproduce the estimated uplift τ = ln s(coupon) − ln s(no coupon)."""
    H = tau.shape[0]
    one = jnp.ones(H)
    cut = jnp.full(H, 1 - discount)

    @jax.jit
    def tau_model(alpha):
        if side == "M":
            s1 = chain_share(theta, ah_belief, base, alpha, cut, one, "M")
        else:
            s1 = chain_share(theta, ah_belief, base, alpha, one, cut, "C")
        s0 = chain_share(theta, ah_belief, base, alpha, one, one, side)
        return jnp.log(jnp.maximum(s1, 1e-300)) - jnp.log(jnp.maximum(s0, 1e-300))

    a_lo, a_hi = jnp.full(H, lo), jnp.full(H, hi)
    t_lo, t_hi = tau_model(a_lo), tau_model(a_hi)
    tau = jnp.clip(jnp.asarray(tau), t_lo + 1e-9, t_hi - 1e-9)
    for _ in range(iters):
        mid = 0.5 * (a_lo + a_hi)
        up = tau_model(mid) < tau
        a_lo, a_hi = jnp.where(up, mid, a_lo), jnp.where(up, a_hi, mid)
    return np.asarray(0.5 * (a_lo + a_hi))


def t_learner(X, y, d, folds=5, seed=0, params=None):
    """Cross-fitted T-learner (LightGBM, Poisson): τ̂ = ln μ̂1(x) − ln μ̂0(x)."""
    import lightgbm as lgb
    params = {"objective": "poisson", "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15,
              "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
              "verbose": -1, "random_state": seed, **(params or {})}
    rng = np.random.default_rng(seed)
    f = rng.integers(0, folds, len(y))
    mu = np.zeros((len(y), 2))
    for k in range(folds):
        tr, te = f != k, f == k
        for t in (0, 1):
            m = lgb.LGBMRegressor(**params).fit(X[tr & (d == t)], y[tr & (d == t)])
            mu[te, t] = m.predict(X[te])
    return np.log(np.maximum(mu[:, 1], 1e-6)) - np.log(np.maximum(mu[:, 0], 1e-6)), mu
