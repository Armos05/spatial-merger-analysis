"""Tests for the store Jacobian and diversion ratios (Phase 2 step 3)."""
import jax
import jax.numpy as jnp
import numpy as np

from mxc.structural import demand as dm
from mxc.structural import substitution as sb
from test_demand import IDX, B, make_arrays, random_market  # noqa: F401  (fixture re-export)


def toy():
    times = np.array([[5, 15, 6], [8, 10, 12], [18, 4, 20], [14, 6, 22.0]])
    a = make_arrays(times, ["Migros", "Coop", "Aldi"], [5, 15, 35, 40], [3, 2, 3, 2], np.ones((4, 1)), [62_500])
    delta = np.zeros(B)
    delta[[IDX["Migros"], IDX["Coop"], IDX["Aldi"]]] = [1.5514, 1.3317, 0.7322]
    theta = dm.prior_params()._replace(delta=jnp.array(delta), p_abroad=72.0)
    return theta, a, jnp.array([0.0275, 0.035, 0.03125, 0.03875])[:, None]


def test_reproduces_design_dummy_jacobian():
    theta, a, alpha = toy()
    Jac, d0, q = sb.jacobian(theta, a, alpha)
    expected = np.array([[-0.0837, 0.0437, 0.0230], [0.0437, -0.0772, 0.0201], [0.0230, 0.0201, -0.0547]])
    np.testing.assert_allclose(Jac.toarray(), expected, atol=2e-4)
    np.testing.assert_allclose(d0, [0.0170, 0.0134, 0.0115], atol=2e-4)
    D, _, _ = sb.group_diversion(Jac, d0, np.arange(3), 3)
    assert abs(D[0, 1] - 0.522) < 3e-3                                     # D(M→C) in design_dummy.md


def test_matches_autodiff(random_market):
    theta, a, *_ = random_market
    theta = theta._replace(eta=0.0)                                         # constant alpha: dV/dp = −alpha = alpha × dV/dxi × (−1)
    J = int(a["store_banner"].shape[0])

    def q_of_xi(xi):
        aa = dict(a, xi_store=xi)
        qs, q0 = dm.quantities(theta, aa)
        return jnp.r_[qs, q0]

    jac_xi = jax.jacfwd(q_of_xi)(jnp.zeros(J))                              # (J+1, J): d q_k / d xi_j
    Jac, d0, _ = sb.jacobian(theta, a)
    np.testing.assert_allclose(Jac.toarray(), -theta.alpha_ref * np.asarray(jac_xi[:J]).T, atol=1e-10)
    np.testing.assert_allclose(d0, -theta.alpha_ref * np.asarray(jac_xi[J]), atol=1e-10)


def test_symmetric_and_rows_sum_to_zero(random_market):
    theta, a, *_ = random_market
    Jac, d0, _ = sb.jacobian(theta, a)
    Jd = Jac.toarray()
    np.testing.assert_allclose(Jd, Jd.T, atol=1e-12)
    np.testing.assert_allclose(Jd.sum(1) + d0, 0.0, atol=1e-10)
    reached = np.isin(np.arange(Jd.shape[0]), np.asarray(a["pair_store"]))
    assert (np.diag(Jd)[reached] < 0).all() and (np.diag(Jd)[~reached] == 0).all()
