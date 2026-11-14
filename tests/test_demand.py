"""Tests for the spatial nested-logit demand (Phase 2 step 1)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mxc.structural import demand as dm

B = len(dm.BANNERS)
IDX = {b: i for i, b in enumerate(dm.BANNERS)}


def make_arrays(times, banners, border, hh, type_w, type_y):
    """Small market from a dense (L, J) time matrix; np.inf = store out of reach."""
    L, J = times.shape
    pl, ps = np.nonzero(np.isfinite(times))
    nest = np.array([dm.NESTS.index(dm.BANNER_NEST[b]) for b in banners])
    return {"pair_loc": jnp.array(pl), "pair_store": jnp.array(ps), "pair_time": jnp.array(times[pl, ps]),
            "pair_seg": jnp.array(pl * len(dm.NESTS) + nest[ps]),
            "store_banner": jnp.array([IDX[b] for b in banners]), "hh": jnp.array(hh, float),
            "type_w": jnp.array(type_w, float), "type_y": jnp.array(type_y, float),
            "border_min": jnp.array(border, float)}


def dense_reference(theta, times, banners, border, alpha_lr):
    """Plain loops over locations: the textbook nested-logit formulas."""
    lam = 1 - theta.sigma
    L, J = times.shape
    R = alpha_lr.shape[1]
    s = np.zeros((L, J, R))
    s0 = np.zeros((L, R))
    for h in range(L):
        for r in range(R):
            V = {j: float(theta.delta[IDX[b]]) - alpha_lr[h, r] * float(theta.p[IDX[b]]) - theta.gamma * times[h, j]
                 for j, b in enumerate(banners) if np.isfinite(times[h, j])}
            V0 = -alpha_lr[h, r] * theta.p_abroad - theta.kappa * border[h]
            D = {}
            for j in V:
                g = dm.BANNER_NEST[banners[j]]
                D[g] = D.get(g, 0.0) + np.exp(V[j] / lam)
            denom = np.exp(V0) + sum(Dg ** lam for Dg in D.values())
            for j in V:
                Dg = D[dm.BANNER_NEST[banners[j]]]
                s[h, j, r] = (Dg ** lam / denom) * np.exp(V[j] / lam) / Dg
            s0[h, r] = np.exp(V0) / denom
    return s, s0


@pytest.fixture
def random_market():
    rng = np.random.default_rng(0)
    banners = ["Migros", "Coop", "Denner", "Aldi", "Volg", "Migrolino", "Coop"]
    L, J, R = 5, len(banners), 3
    times = rng.uniform(1, 25, (L, J))
    times[rng.random((L, J)) < 0.35] = np.inf          # sparse reach: some nests empty at some locations
    times[:, 0] = rng.uniform(1, 10, L)                 # every location reaches at least one store
    border = rng.uniform(5, 60, L)
    type_w = rng.dirichlet(np.ones(R), L)
    type_y = np.array([30_000, 62_500, 150_000])
    a = make_arrays(times, banners, border, rng.integers(5, 50, L), type_w, type_y)
    theta = dm.prior_params()._replace(delta=jnp.array(rng.normal(0, 1, B)))
    return theta, a, times, banners, border


def to_dense(s_pair, a, L, J):
    out = np.zeros((L, J, s_pair.shape[1]))
    out[np.asarray(a["pair_loc"]), np.asarray(a["pair_store"])] = np.asarray(s_pair)
    return out


def test_probabilities_sum_to_one(random_market):
    theta, a, *_ = random_market
    s_pair, s0, _ = dm.choice_probs(theta, a)
    tot = jax.ops.segment_sum(s_pair, a["pair_loc"], num_segments=a["hh"].shape[0]) + s0
    np.testing.assert_allclose(tot, 1.0, atol=1e-12)


def test_matches_dense_textbook_formulas(random_market):
    theta, a, times, banners, border = random_market
    alpha_lr = np.broadcast_to(np.asarray(dm.alpha_types(theta, a["type_y"])), (len(border), 3))
    s_ref, s0_ref = dense_reference(theta, times, banners, border, alpha_lr)
    s_pair, s0, _ = dm.choice_probs(theta, a)
    np.testing.assert_allclose(to_dense(s_pair, a, *times.shape), s_ref, atol=1e-12)
    np.testing.assert_allclose(s0, s0_ref, atol=1e-12)


def test_sigma_zero_is_plain_logit(random_market):
    theta, a, times, banners, border = random_market
    theta = theta._replace(sigma=0.0)
    s_pair, s0, logsum = dm.choice_probs(theta, a)
    alpha = dm.alpha_types(theta, a["type_y"])
    b = a["store_banner"][a["pair_store"]]
    V = theta.delta[b][:, None] - alpha[None, :] * theta.p[b][:, None] - theta.gamma * a["pair_time"][:, None]
    np.testing.assert_allclose(s_pair, jnp.exp(V - logsum[a["pair_loc"]]), atol=1e-12)


def test_reproduces_design_dummy_toy():
    """The toy world of design_dummy.md (Step 3): calibrated delta gives q = (3.20, 3.00, 2.20; abroad 1.60)."""
    times = np.array([[5, 15, 6], [8, 10, 12], [18, 4, 20], [14, 6, 22.0]])
    a = make_arrays(times, ["Migros", "Coop", "Aldi"], [5, 15, 35, 40], [3, 2, 3, 2],
                    np.ones((4, 1)), [62_500])
    delta = np.zeros(B)
    delta[[IDX["Migros"], IDX["Coop"], IDX["Aldi"]]] = [1.5514, 1.3317, 0.7322]
    theta = dm.prior_params()._replace(delta=jnp.array(delta), p_abroad=72.0)
    alpha_lr = jnp.array([0.0275, 0.035, 0.03125, 0.03875])[:, None]
    q, q0 = dm.quantities(theta, a, alpha_lr)
    np.testing.assert_allclose(np.r_[np.asarray(q), float(q0)], [3.20, 3.00, 2.20, 1.60], atol=2e-3)
    s_pair, _, _ = dm.choice_probs(theta, a, alpha_lr)
    assert float(s_pair[0, 0]) == pytest.approx(0.3962, abs=2e-4)     # N1, Migros (Step 5 worked cell)


def test_richer_households_less_price_sensitive():
    theta = dm.prior_params()
    al = np.asarray(dm.alpha_types(theta, jnp.array([20_000.0, 62_500.0, 300_000.0])))
    assert al[0] > al[1] > al[2] and al[1] == pytest.approx(theta.alpha_ref)


def test_gradient_matches_finite_difference(random_market):
    theta, a, *_ = random_market
    f = lambda d: dm.quantities(theta._replace(delta=d), a)[0][0]      # noqa: E731  Migros store 0
    g = jax.grad(f)(theta.delta)
    e = 1e-6
    for k in [IDX["Migros"], IDX["Coop"], IDX["Aldi"]]:
        fd = (f(theta.delta.at[k].add(e)) - f(theta.delta.at[k].add(-e))) / (2 * e)
        assert float(g[k]) == pytest.approx(float(fd), rel=1e-5, abs=1e-9)


def test_every_banner_belongs_to_one_player():
    from mxc.structural import calibrate as cb
    flat = [b for bs in cb.PLAYERS.values() for b in bs]
    assert sorted(flat) == sorted(dm.BANNERS)


def test_calibration_hits_targets(random_market):
    from mxc.structural import calibrate as cb
    theta, a, *_ = random_market
    theta = theta._replace(delta=jnp.zeros(B))
    # every player needs a store: add the missing banners as extra stores reachable from location 0
    banners = [dm.BANNERS[int(i)] for i in np.asarray(a["store_banner"])]
    extra = [b for b in dm.BANNERS if b not in banners]
    J0 = len(banners)
    nest = np.array([dm.NESTS.index(dm.BANNER_NEST[b]) for b in banners + extra])
    a = dict(a)
    a["store_banner"] = jnp.array([IDX[b] for b in banners + extra])
    new_s = np.arange(J0, J0 + len(extra))
    a["pair_loc"] = jnp.concatenate([a["pair_loc"], jnp.zeros(len(extra), int)])
    a["pair_store"] = jnp.concatenate([a["pair_store"], jnp.array(new_s)])
    a["pair_time"] = jnp.concatenate([a["pair_time"], jnp.full(len(extra), 5.0)])
    a["pair_seg"] = a["pair_loc"] * len(dm.NESTS) + jnp.array(nest)[a["pair_store"]]
    tgt = cb.targets(0.05)
    theta_c, hist = cb.calibrate(theta, a, tgt, tol=1e-11)
    model = np.asarray(cb.model_revenue_shares(theta_c, a, cb.player_index()))
    np.testing.assert_allclose(model, tgt.to_numpy(), atol=1e-9)
    assert hist.max_abs_log_gap.iloc[-1] < 1e-11
