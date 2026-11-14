"""Pillar 1 closing checks (ROADMAP Phase 2): derivatives vs finite differences, equilibrium residual,
zero diversion, λμ = 1, and Roy's identity for the welfare measure."""
import jax.numpy as jnp
import numpy as np

from mxc.structural import demand as dm
from mxc.structural import simulate as sim
from mxc.structural import supply as su
from test_calibrate_helpers import full_market
from test_demand import random_market  # noqa: F401

MERGING = np.array([su.OWNER_PRE[b] in ("Migros group", "Coop group") for b in dm.BANNERS])
LM = np.where(MERGING, 0.35, 0.0)


def setup(random_market):
    theta, a, *_ = random_market
    a = full_market(a)
    bd = sim.make_banner_demand(a)
    Q, A, *_ = bd(theta)
    cost = np.asarray(theta.p) - su.markups(np.asarray(A), np.asarray(Q), su.OWNER_PRE, LM)
    return theta, a, bd, cost


def test_banner_jacobian_matches_finite_differences(random_market):
    """A[B,B'] = dQ_B'/dp_B, checked by central differences on every banner price."""
    theta, a, bd, _ = setup(random_market)
    A = np.asarray(bd(theta)[1])
    h = 1e-4
    p = np.asarray(theta.p)
    for b in range(len(p)):
        up = bd(theta._replace(p=jnp.asarray(p + h * np.eye(len(p))[b])))[0]
        dn = bd(theta._replace(p=jnp.asarray(p - h * np.eye(len(p))[b])))[0]
        np.testing.assert_allclose(A[b], (np.asarray(up) - np.asarray(dn)) / (2 * h), rtol=1e-6, atol=1e-9)


def test_post_merger_equilibrium_satisfies_focs(random_market):
    """At the solver's post-merger prices (with a 5% saving): (1 − λμ) Q + (Ω ∘ A)(p − c) = 0."""
    theta, a, bd, cost = setup(random_market)
    c = np.where(MERGING, cost * 0.95, cost)
    p, _ = sim.solve_equilibrium(theta, bd, c, LM, su.OWNER_POST)
    Q, A, *_ = bd(theta._replace(p=jnp.asarray(p)))
    res = su.foc_residual(np.asarray(A), np.asarray(Q), su.OWNER_POST, LM, p - c)
    np.testing.assert_allclose(res / np.asarray(Q), 0, atol=1e-8)


def test_zero_diversion_means_no_merger_effect(random_market):
    """If no location reaches both Migros-group and Coop-group stores, diversion between them is zero,
    so the merger changes no price and no welfare."""
    theta, a, _, _ = setup(random_market)
    sb = np.asarray(a["store_banner"])
    group_of_store = np.array([su.OWNER_PRE[dm.BANNERS[i]] for i in sb])
    pl, ps = np.asarray(a["pair_loc"]), np.asarray(a["pair_store"])
    banned = np.where(pl % 2 == 0, "Coop group", "Migros group")      # even locations: no Coop; odd: no Migros
    keep = group_of_store[ps] != banned
    a = {**a, **{k: a[k][keep] for k in ("pair_loc", "pair_store", "pair_time", "pair_seg")}}
    bd = sim.make_banner_demand(a)
    Q, A, *_ = bd(theta)
    A = np.asarray(A)
    mig = np.array([su.OWNER_PRE[b] == "Migros group" for b in dm.BANNERS])
    coo = np.array([su.OWNER_PRE[b] == "Coop group" for b in dm.BANNERS])
    assert np.abs(A[np.ix_(mig, coo)]).max() == 0
    cost = np.asarray(theta.p) - su.markups(A, np.asarray(Q), su.OWNER_PRE, LM)
    p, _ = sim.solve_equilibrium(theta, bd, cost, LM, su.OWNER_POST)
    np.testing.assert_allclose(p, np.asarray(theta.p), atol=1e-7)


def test_full_member_weight_prices_at_cost(random_market):
    """λμ = 1: the firm values a franc of member surplus like a franc of profit → price = marginal cost,
    before and after the merger."""
    theta, a, bd, cost = setup(random_market)
    lm = np.where(MERGING, 1.0, 0.0)
    for owner in (su.OWNER_PRE, su.OWNER_POST):
        p, _ = sim.solve_equilibrium(theta, bd, cost, lm, owner)
        np.testing.assert_allclose(p[MERGING], cost[MERGING], atol=1e-7)


def test_delta_cs_obeys_roys_identity(random_market):
    """Small price rise dp on one banner: total ΔCS ≈ −Q_B × dp (Roy's identity), to first order."""
    theta, a, bd, _ = setup(random_market)
    Q = np.asarray(bd(theta)[0])
    n_lr = np.asarray(a["hh"])[:, None] * np.asarray(a["type_w"])
    for b in (dm.BANNERS.index("Migros"), dm.BANNERS.index("Aldi")):
        dp = 1e-3
        th1 = theta._replace(p=theta.p.at[b].add(dp))
        tot = (n_lr * sim.delta_cs(theta, th1, a)).sum()
        np.testing.assert_allclose(tot, -Q[b] * dp, rtol=1e-3)
