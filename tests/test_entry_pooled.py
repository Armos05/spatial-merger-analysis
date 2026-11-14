"""Tests for the T1 entry estimators (Pillar 2 step 3)."""
import numpy as np
import pandas as pd

from mxc.causal import entry_pooled as ep


def test_demean_and_ols_recover_coefficient_with_two_way_fixed_effects():
    rng = np.random.default_rng(0)
    N, T = 300, 40
    i, t = np.repeat(np.arange(N), T), np.tile(np.arange(T), N)
    x = rng.normal(size=N * T) + rng.normal(size=N)[i]
    y = 0.7 * x + rng.normal(size=N)[i] * 3 + rng.normal(size=T)[t] * 2 + rng.normal(0, 0.1, N * T)
    Z = ep.demean(np.c_[y, x], [i, t])
    b, se = ep.ols_cluster(Z[:, 0], Z[:, 1:], i)
    assert abs(b[0] - 0.7) < 0.01 and 0 < se[0] < 0.01


def test_exposure_regression_recovers_additive_band_effects():
    """Synthetic panel: each open entrant within 1 km takes 3% (log), 1-2 km takes 1%, nothing beyond."""
    rng = np.random.default_rng(1)
    J, E, T = 400, 30, 60
    xy = rng.uniform(0, 12, (J, 2))
    ent = pd.DataFrame({"store": np.arange(E), "week": rng.integers(10, 50, E)})
    D = np.hypot(*(xy[:, None, :] - xy[:E][None, :, :]).transpose(2, 0, 1))
    eff = np.where(D <= 1, -0.03, np.where(D <= 2, -0.01, 0.0))
    is_open = ent.week.to_numpy()[None, :] <= np.arange(T)[:, None]
    lnq = 6 + rng.normal(0, 0.3, J)[None, :] + rng.normal(0, 0.05, T)[:, None] + is_open.astype(float) @ eff.T \
        + rng.normal(0, 0.01, (T, J))
    tt, kk = np.meshgrid(np.arange(T), np.arange(J), indexing="ij")
    panel = pd.DataFrame({"store": kk.ravel(), "week": tt.ravel(), "baskets": np.exp(lnq.ravel())})
    r = ep.exposure_regression(panel, ent, xy, np.array(["g"] * J), bands=(0, 1, 2, 4), groups=["g"])
    np.testing.assert_allclose(r.beta.to_numpy(), [-0.03, -0.01, 0.0], atol=0.003)


def test_to_loss_interval_brackets_point():
    out = ep.to_loss(pd.DataFrame({"beta": [-0.02], "se": [0.005]}))
    assert out.lo95[0] < out.loss_pct[0] < out.hi95[0]
    np.testing.assert_allclose(out.loss_pct[0], 100 * (1 - np.exp(-0.02)))
