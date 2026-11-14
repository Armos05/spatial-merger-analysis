"""Tests for GUPPI / CMCR screens (Phase 2 step 5)."""
import numpy as np

from mxc.structural import screening as sc
from mxc.structural import substitution as sb
from test_substitution import toy


def test_store_guppi_reproduces_design_dummy():
    # design_dummy.md: GUPPI_M = D(M→C) × m_C / p_M = 0.522 × 24.38 / 100 = 12.7%;  GUPPI_C = 0.565 × 24.00 / 102 = 13.3%
    theta, a, alpha = toy()
    Jac, _, _ = sb.jacobian(theta, a, alpha)
    party = np.array(["Migros group", "Coop group", "Aldi"])
    g, d = sc.store_guppi(Jac, party, np.array([24.0, 24.38, 40.25]), np.array([100.0, 102.0, 85.0]))
    np.testing.assert_allclose(d[:2], [0.522, 0.565], atol=3e-3)
    np.testing.assert_allclose(g[:2], [0.127, 0.133], atol=2e-3)
    assert np.isnan(g[2])


def test_guppi_equals_revenue_diversion_times_margin():
    # Koh (2025): GUPPI_j = DR(j→k) × μ_k with DR = D × p_k / p_j, μ_k = m_k / p_k
    theta, a, alpha = toy()
    Jac, _, _ = sb.jacobian(theta, a, alpha)
    p, m = np.array([100.0, 102.0, 85.0]), np.array([24.0, 24.38, 40.25])
    g, d = sc.store_guppi(Jac, np.array(["Migros group", "Coop group", "Aldi"]), m, p)
    DR = d[0] * p[1] / p[0]
    assert abs(g[0] - DR * m[1] / p[1]) < 1e-12
