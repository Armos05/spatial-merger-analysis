"""Tests for the T2 estimators (Pillar 2 step 4)."""
import numpy as np
import pandas as pd

from mxc.causal import dml


def synthetic(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    xi = rng.normal(0, 1, n)                     # demand shock (hidden)
    z = rng.normal(0, 1, n)                      # cost shock (instrument)
    cell = rng.integers(0, 2, n)
    theta = np.where(cell == 0, -3.0, -5.0)
    d = 0.5 * xi + z + rng.normal(0, 0.3, n)
    y = theta * d + xi + rng.normal(0, 0.5, n)
    return y, d, z, cell, rng.integers(0, 500, n)


def test_iv_recovers_cell_elasticities_and_ols_is_biased():
    y, d, z, cell, cl = synthetic()
    iv = dml.theta_by_cell(y - y.mean(), d - d.mean(), z - z.mean(), cell, cl)
    ols = dml.theta_by_cell(y - y.mean(), d - d.mean(), d - d.mean(), cell, cl)
    np.testing.assert_allclose(iv.theta, [-3.0, -5.0], atol=0.05)
    assert (ols.theta.to_numpy() > iv.theta.to_numpy() + 0.3).all()          # OLS biased towards zero
    assert ((iv.theta - 1.96 * iv.se < [-3, -5]) & ([-3, -5] < iv.theta + 1.96 * iv.se)).all()


def test_crossfit_gives_out_of_fold_predictions():
    rng = np.random.default_rng(1)
    n = 6_000
    df = pd.DataFrame({"store": rng.integers(0, 300, n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["t"] = np.sin(df.x1) + 0.5 * df.x2 ** 2 + rng.normal(0, 0.1, n)
    pred, r2 = dml.crossfit(df, ["t"], features=["x1", "x2"], params={"n_estimators": 150, "min_child_samples": 20})
    assert r2["t"] > 0.9 and len(pred["t"]) == n
