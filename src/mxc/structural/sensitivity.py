"""Uncertainty: exact sensitivities by implicit differentiation, and Monte Carlo (Phase 5; DESIGN §3.7).

The headline harm depends on assumed parameters ψ = (σ, α_ref, η, γ, κ, λμ, p_abroad, efficiency).
Changing ψ is not just plugging in a number: the model is re-fitted. Three nested conditions:

  1. calibration   F1(u, ψ) = ln s_G(δ0 + u_G(b), ψ) − ln target_G = 0        (6 player shifts u)
  2. costs          c(u, ψ) = p0 + (Ω_pre ∘ A(p0))^(−1) (1 − λμ) Q(p0)          (pre-merger FOCs at today's prices)
  3. equilibrium    F3(p1, u, ψ) = p1 − c × (1 − e·merged) − m_post(p1) = 0     (12 banner prices)
  harm             H(p1, u, ψ) = 52 × Σ_lr n_lr [logsum_lr(p1) − logsum_lr(p0)] / α_r / Σ_l hh_l    (CHF/household/yr)

Implicit function theorem:
    du/dψ  = −[∂F1/∂u]^(−1) ∂F1/∂ψ
    dp1/dψ = −[∂F3/∂p1]^(−1) (∂F3/∂u du/dψ + ∂F3/∂ψ)
    dH/dψ  = ∂H/∂ψ + ∂H/∂u du/dψ + ∂H/∂p1 dp1/dψ
All partial derivatives come from JAX forward-mode autodiff of the same code that produced the numbers.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import calibrate as cb
from . import demand as dm
from . import simulate as sim
from . import supply as su

PSI = ["sigma", "alpha_ref", "eta", "gamma", "kappa", "lambda_mu", "p_abroad", "efficiency"]
COOP_BANNERS = [b for b in dm.BANNERS if su.OWNER_PRE[b] in ("Migros group", "Coop group")]


class Model:
    def __init__(self, theta0, a, target, p0, lm0, eff0=0.0):
        self.theta0, self.a, self.p0 = theta0, a, jnp.asarray(p0)
        self.pidx = jnp.asarray(cb.player_index())
        self.ln_t = jnp.log(jnp.asarray(target.to_numpy()[:-1]))
        self.coop = jnp.asarray([b in COOP_BANNERS for b in dm.BANNERS], dtype=float)
        self.om_pre = jnp.asarray(su.omega(su.OWNER_PRE))
        self.om_post = jnp.asarray(su.omega(su.OWNER_POST))
        self.n_lr = a["hh"][:, None] * a["type_w"]
        self.psi0 = jnp.array([theta0.sigma, theta0.alpha_ref, theta0.eta, theta0.gamma, theta0.kappa, lm0,
                               theta0.p_abroad, eff0])
        self.bd = sim.make_banner_demand(a)

    def theta(self, psi, u, p=None):
        t = self.theta0
        return t._replace(sigma=psi[0], alpha_ref=psi[1], eta=psi[2], gamma=psi[3], kappa=psi[4], p_abroad=psi[6],
                          delta=t.delta + u[self.pidx], p=self.p0 if p is None else p)

    def F1(self, u, psi):
        s = cb.model_revenue_shares(self.theta(psi, u), self.a, self.pidx)
        return self.ln_t - jnp.log(s[:-1])

    def cost(self, u, psi):
        Q, A, *_ = self.bd(self.theta(psi, u))
        lm = psi[5] * self.coop
        return self.p0 + jnp.linalg.solve(self.om_pre * A, (1 - lm) * Q)

    def F3(self, p1, u, psi):
        c = self.cost(u, psi) * (1 - psi[7] * self.coop)
        Q, A, *_ = self.bd(self.theta(psi, u, p1))
        lm = psi[5] * self.coop
        m = -jnp.linalg.solve(self.om_post * A, (1 - lm) * Q)
        return p1 - c - m

    def harm(self, p1, u, psi):
        _, _, ls1 = dm.choice_probs(self.theta(psi, u, p1), self.a)
        _, _, ls0 = dm.choice_probs(self.theta(psi, u), self.a)
        alpha = dm.alpha_types(self.theta(psi, u), self.a["type_y"])
        return 52.0 * (self.n_lr * (ls1 - ls0) / alpha[None, :]).sum() / self.a["hh"].sum()

    def outputs(self, p1, u, psi):
        """Headline numbers: harm (CHF/hh/yr), Migros and Coop price change (%), abroad share."""
        h = self.harm(p1, u, psi)
        iM, iC = dm.BANNERS.index("Migros"), dm.BANNERS.index("Coop")
        return jnp.array([h, 100 * (p1[iM] / self.p0[iM] - 1), 100 * (p1[iC] / self.p0[iC] - 1)])

    # ---------------------------------------------------------------- solving and derivatives
    def solve_u(self, psi, u=None, tol=1e-10, it=30):
        u = jnp.zeros(6) if u is None else u
        J = jax.jacfwd(self.F1, 0)
        for _ in range(it):
            f = self.F1(u, psi)
            if float(jnp.abs(f).max()) < tol:
                break
            u = u - jnp.linalg.solve(J(u, psi), f)
        return u, float(jnp.abs(self.F1(u, psi)).max())

    def solve_p(self, psi, u, p1, tol=1e-9, it=40):
        J = jax.jacfwd(self.F3, 0)
        for _ in range(it):
            f = self.F3(p1, u, psi)
            if float(jnp.abs(f).max()) < tol:
                break
            step = jnp.linalg.solve(J(p1, u, psi), f)
            p1 = p1 - jnp.clip(step, -10, 10)
        return p1, float(jnp.abs(self.F3(p1, u, psi)).max())

    def gradient(self, psi, u, p1):
        """d outputs / d psi (3 × 8) by the implicit function theorem, plus the pieces."""
        F1u = jax.jacfwd(self.F1, 0)(u, psi)
        F1s = jax.jacfwd(self.F1, 1)(u, psi)
        du = -jnp.linalg.solve(F1u, F1s)
        F3p = jax.jacfwd(self.F3, 0)(p1, u, psi)
        F3u = jax.jacfwd(self.F3, 1)(p1, u, psi)
        F3s = jax.jacfwd(self.F3, 2)(p1, u, psi)
        dp = -jnp.linalg.solve(F3p, F3u @ du + F3s)
        Op = jax.jacfwd(self.outputs, 0)(p1, u, psi)
        Ou = jax.jacfwd(self.outputs, 1)(p1, u, psi)
        Os = jax.jacfwd(self.outputs, 2)(p1, u, psi)
        return np.asarray(Os + Ou @ du + Op @ dp), np.asarray(du), np.asarray(dp)
