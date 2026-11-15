"""Tests for the implicit-differentiation sensitivities (src/mxc/structural/sensitivity.py)."""
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from mxc.structural import demand as dm
from mxc.structural import sensitivity as se
from test_demand import make_arrays


@pytest.fixture
def full_market():
    """Every banner present (one store each, two Coop), all reachable: the banner Jacobian is invertible."""
    rng = np.random.default_rng(3)
    banners = dm.BANNERS + ["Coop"]
    L, R = 6, 3
    times = rng.uniform(1, 20, (L, len(banners)))
    a = make_arrays(times, banners, rng.uniform(5, 60, L), rng.integers(5, 50, L), rng.dirichlet(np.ones(R), L),
                    np.array([30_000, 62_500, 150_000]))
    theta = dm.prior_params()._replace(delta=jnp.array(rng.normal(0, 0.5, len(dm.BANNERS))))
    return theta, a


def model(full_market):
    theta, a = full_market
    target = pd.Series(np.full(7, 1 / 7))
    return se.Model(theta, a, target, np.asarray(theta.p), 0.3)


def test_pre_merger_prices_solve_pre_merger_focs(full_market):
    M = model(full_market)
    u = jnp.zeros(6)
    c = M.cost(u, M.psi0)
    # with recovered costs, today's prices satisfy the pre-merger FOCs: m = p0 − c
    Q, A, *_ = M.bd(M.theta(M.psi0, u))
    lm = M.psi0[5] * M.coop
    resid = (1 - lm) * Q + (M.om_pre * A) @ (M.p0 - c)
    assert float(jnp.abs(resid).max()) < 1e-8


def test_implicit_price_derivative_matches_finite_difference(full_market):
    M = model(full_market)
    u = jnp.zeros(6)
    psi = M.psi0
    p1, r = M.solve_p(psi, u, M.p0 * 1.05)
    assert r < 1e-9
    F3p = jax.jacfwd(M.F3, 0)(p1, u, psi)
    F3s = jax.jacfwd(M.F3, 2)(p1, u, psi)
    dp = -np.linalg.solve(np.asarray(F3p), np.asarray(F3s))
    for k in (0, 5, 7):                              # sigma, lambda_mu, efficiency
        h = 1e-4
        pp, _ = M.solve_p(psi.at[k].add(h), u, p1)
        pm, _ = M.solve_p(psi.at[k].add(-h), u, p1)
        fd = (np.asarray(pp) - np.asarray(pm)) / (2 * h)
        np.testing.assert_allclose(dp[:, k], fd, rtol=1e-4, atol=1e-5)


def test_harm_is_zero_without_a_price_change(full_market):
    M = model(full_market)
    assert abs(float(M.harm(M.p0, jnp.zeros(6), M.psi0))) < 1e-10


def test_parameter_gradients_finite_with_empty_nests():
    """A location with no store in some nest (lnD = -inf) must not turn the ψ-tangents into NaN."""
    rng = np.random.default_rng(5)
    banners = dm.BANNERS + ["Coop"]
    times = rng.uniform(1, 20, (4, len(banners)))
    times[0, [i for i, b in enumerate(banners) if b != "Migros"]] = np.inf      # location 0: one nest only
    a = make_arrays(times, banners, rng.uniform(5, 60, 4), rng.integers(5, 50, 4), rng.dirichlet(np.ones(3), 4),
                    np.array([30_000, 62_500, 150_000]))
    theta = dm.prior_params()

    def logsum(psi):
        t = theta._replace(sigma=psi[0], gamma=psi[1])
        return dm.choice_probs(t, a)[2].sum()
    g = jax.jacfwd(logsum)(jnp.array([float(theta.sigma), float(theta.gamma)]))
    assert np.isfinite(np.asarray(g)).all()
