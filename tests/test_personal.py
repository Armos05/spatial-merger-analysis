"""Tests for Pillar 3: personalised pricing engine (src/mxc/datamerger/personal.py)."""
import jax.numpy as jnp
import numpy as np
import pandas as pd

from mxc.datamerger import personal as pe
from mxc.structural import demand as dm
from test_demand import random_market  # noqa: F401


def panel(random_market, H=12, seed=1):
    theta, a, *_ = random_market
    rng = np.random.default_rng(seed)
    L = int(a["hh"].shape[0])
    hh = pd.DataFrame({"hh_id": np.arange(H), "loc": rng.integers(0, L, H), "segment": "mainstream",
                       "gamma_mult": 1.0, "kappa_mult": 1.0, "cumulus": rng.random(H) < 0.8,
                       "supercard": rng.random(H) < 0.8, "alpha_i": rng.uniform(0.015, 0.06, H)})
    hh["border_min"] = np.asarray(a["border_min"])[hh["loc"]]
    J = int(a["store_banner"].shape[0])
    ah = pe.panel_arrays(a, hh, np.zeros(J), informed=True)
    base = theta.p
    cost = theta.p * 0.75
    lm = jnp.zeros(len(dm.BANNERS))
    return theta, a, hh, ah, base, cost, lm


def test_household_logit_matches_pillar1_at_uniform_prices(random_market):
    theta, a, hh, ah, base, *_ = panel(random_market)
    H = len(hh)
    al = jnp.asarray(hh.alpha_i.to_numpy())
    p = pe.pair_prices(ah, base, jnp.ones(H), jnp.ones(H))
    s, s0, _ = pe.hh_logit(theta, ah, p, al)
    tot = np.bincount(np.asarray(ah["pair_loc"]), np.asarray(s), H) + np.asarray(s0)
    np.testing.assert_allclose(tot, 1.0, atol=1e-12)
    # same household through demand.choice_probs with its own alpha
    s_ref, s0_ref, _ = dm.choice_probs(theta, {**ah, "xi_store": jnp.zeros(a["store_banner"].shape[0])}, al[:, None])
    np.testing.assert_allclose(np.asarray(s), np.asarray(s_ref)[:, 0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(s0), np.asarray(s0_ref)[:, 0], atol=1e-12)


def test_competition_solution_satisfies_first_order_conditions(random_market):
    theta, a, hh, ah, base, cost, lm = panel(random_market)
    al = jnp.asarray(hh.alpha_i.to_numpy())
    mM, mC, g, _ = pe.solve(theta, ah, base, cost, lm, al, al, merged=False, iters=300)
    assert g < 1e-6
    # no profitable unilateral deviation for Migros (check a small perturbation)
    WM0 = np.asarray(pe.objectives(theta, ah, base, cost, lm, al, jnp.asarray(mM), jnp.asarray(mC))[0])
    for d in (-0.01, 0.01):
        WM1 = np.asarray(pe.objectives(theta, ah, base, cost, lm, al, jnp.asarray(mM + d), jnp.asarray(mC))[0])
        assert (WM1 <= WM0 + 1e-9).all()


def test_merger_raises_personal_prices(random_market):
    theta, a, hh, ah, base, cost, lm = panel(random_market)
    al = jnp.asarray(hh.alpha_i.to_numpy())
    mM0, mC0, *_ = pe.solve(theta, ah, base, cost, lm, al, al, merged=False, iters=300)
    mM1, mC1, *_ = pe.solve(theta, ah, base, cost, lm, al, al, merged=True, iters=300)
    both = (hh.cumulus & hh.supercard).to_numpy()
    assert (mM1[both] >= mM0[both] - 1e-6).all() and (mC1[both] >= mC0[both] - 1e-6).all()


def test_inversion_recovers_alpha_from_model_uplift(random_market):
    theta, a, hh, ah, base, *_ = panel(random_market)
    H = len(hh)
    al = jnp.asarray(hh.alpha_i.to_numpy())
    one, cut = jnp.ones(H), jnp.full(H, 0.9)
    s1 = pe.chain_share(theta, ah, base, al, cut, one, "M")
    s0 = pe.chain_share(theta, ah, base, al, one, one, "M")
    ok = np.asarray(s0) > 1e-9
    tau = jnp.where(ok, jnp.log(jnp.maximum(s1, 1e-300)) - jnp.log(jnp.maximum(s0, 1e-300)), 0.1)
    a_hat = pe.invert_alpha(theta, ah, base, tau, "M", 0.10)
    np.testing.assert_allclose(a_hat[ok], np.asarray(al)[ok], rtol=1e-6)
