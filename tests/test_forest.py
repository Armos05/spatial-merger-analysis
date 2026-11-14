"""Tests for T3 local elasticities and local GUPPI (Pillar 2 step 5)."""
import numpy as np
import scipy.sparse as sp

from mxc.causal import forest as fo


def test_store_by_store_iv_recovers_each_store():
    rng = np.random.default_rng(0)
    J, T = 20, 3000
    theta = rng.uniform(-5, -3, J)
    s = np.repeat(np.arange(J), T)
    xi, z = rng.normal(size=J * T), rng.normal(size=J * T)
    d = 0.5 * xi + z
    y = theta[s] * d + xi
    th, se = fo.store_by_store(y, d, z, s, J)
    assert np.all(np.abs(th - theta) < 4 * se) and se.max() < 0.1


def test_linear_iv_recovers_linear_heterogeneity():
    rng = np.random.default_rng(1)
    n = 50_000
    x = rng.normal(size=(n, 1))
    xi, z = rng.normal(size=n), rng.normal(size=n)
    d = 0.5 * xi + z
    y = (-4 + 0.5 * x[:, 0]) * d + xi
    b = fo.linear_iv(y, d, z, x)
    np.testing.assert_allclose(b, [-4, 0.5], atol=0.03)


def test_local_guppi_two_stores():
    """One Migros and one Coop store: GUPPI_M = D(M→C) × m_C / p_M with m_C = p_C / |θ_C|."""
    Jac = sp.csr_matrix(np.array([[-10.0, 4.0], [4.0, -8.0]]))
    price, theta = np.array([100.0, 100.0]), np.array([-4.0, -5.0])
    g = fo.local_guppi(Jac, price, theta, np.array(["migros", "coop"]))
    np.testing.assert_allclose(g, [0.4 * 20 / 100, 0.5 * 25 / 100])
