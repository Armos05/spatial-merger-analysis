"""Tests for the merger simulation (Phase 2 step 6)."""
import jax.numpy as jnp
import numpy as np

from mxc.structural import demand as dm
from mxc.structural import simulate as sim
from mxc.structural import substitution as sb
from mxc.structural import supply as su
from test_demand import random_market  # noqa: F401
from test_calibrate_helpers import full_market


def test_banner_jacobian_equals_summed_store_jacobian(random_market):
    theta, a, *_ = random_market
    Q, A, *_ = sim.make_banner_demand(a)(theta)
    Jac, d0, q = sb.jacobian(theta, a)
    _, A_store, _ = sb.group_diversion(Jac, d0, np.asarray(a["store_banner"]), len(dm.BANNERS))
    np.testing.assert_allclose(np.asarray(A), A_store, atol=1e-10)
    np.testing.assert_allclose(np.asarray(Q), np.bincount(np.asarray(a["store_banner"]), weights=q,
                                                          minlength=len(dm.BANNERS)), atol=1e-10)


def test_pre_merger_equilibrium_returns_todays_prices(random_market):
    theta, a, *_ = random_market
    a = full_market(a)
    bd = sim.make_banner_demand(a)
    Q, A, *_ = bd(theta)
    lm = np.array([0.35 if su.OWNER_PRE[b] in ("Migros group", "Coop group") else 0.0 for b in dm.BANNERS])
    cost = np.asarray(theta.p) - su.markups(np.asarray(A), np.asarray(Q), su.OWNER_PRE, lm)
    p, hist = sim.solve_equilibrium(theta, bd, cost, lm, su.OWNER_PRE, p_start=np.asarray(theta.p) * 1.05)
    np.testing.assert_allclose(p, np.asarray(theta.p), atol=1e-7)
    p_post, _ = sim.solve_equilibrium(theta, bd, cost, lm, su.OWNER_POST)
    merging = np.array([su.OWNER_PRE[b] in ("Migros group", "Coop group") for b in dm.BANNERS])
    assert (p_post[merging] > np.asarray(theta.p)[merging]).all()
    dcs = sim.delta_cs(theta, theta._replace(p=jnp.asarray(p_post)), a)
    assert (dcs < 0).all()
