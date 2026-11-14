"""Tests for cooperative-conduct supply (Phase 2 step 4)."""
import numpy as np
import scipy.sparse as sp

from mxc.structural import demand as dm
from mxc.structural import supply as su


def banner_system(seed=0):
    rng = np.random.default_rng(seed)
    B = len(dm.BANNERS)
    X = rng.uniform(0.001, 0.01, (B, B))
    A = (X + X.T) / 2
    np.fill_diagonal(A, 0)
    A[np.diag_indices(B)] = -(A.sum(1) + rng.uniform(0.005, 0.02, B))   # own effect beats cross effects (abroad leak)
    Q = rng.uniform(1, 5, B)
    return A, Q


def test_foc_holds_and_scales_with_lambda():
    A, Q = banner_system()
    m0 = su.markups(A, Q, su.OWNER_PRE, np.zeros(len(Q)))
    lm = np.array([0.4 if su.OWNER_PRE[b] in ("Migros group", "Coop group") else 0.0 for b in dm.BANNERS])
    m = su.markups(A, Q, su.OWNER_PRE, lm)
    np.testing.assert_allclose(su.foc_residual(A, Q, su.OWNER_PRE, lm, m), 0, atol=1e-10)
    np.testing.assert_allclose(m, (1 - lm) * m0, rtol=1e-10)                   # markup = (1 − λμ) × profit-max markup
    assert (m0 > 0).all()


def test_merger_raises_merging_markups_only():
    A, Q = banner_system(1)
    pre = su.markups(A, Q, su.OWNER_PRE, np.zeros(len(Q)))
    post = su.markups(A, Q, su.OWNER_POST, np.zeros(len(Q)))
    merging = np.array([su.OWNER_PRE[b] in ("Migros group", "Coop group") for b in dm.BANNERS])
    assert (post[merging] > pre[merging]).all()
    np.testing.assert_allclose(post[~merging], pre[~merging])


def test_lambda_from_margin_toy():
    # design_dummy.md Step 5: (1 − λμ) × 38.241 = 0.24 × 100 → λμ = 0.3724
    assert abs(su.lambda_from_margin(38.241, 100.0, 0.24) - 0.3724) < 1e-4


def test_local_margin_single_owner_store_is_inverse_elasticity():
    # two stores, different owners: m_j = (1 − λμ) q_j / (−Jac_jj)
    Jac = sp.csr_matrix(np.array([[-0.08, 0.04], [0.04, -0.07]]))
    m = su.local_margin(Jac, np.array([3.2, 3.0]), np.array([0, 1]), np.array(["a", "b"]),
                        np.array([10.0, 10.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(m, [3.2 / 0.08, 3.0 / 0.07])
