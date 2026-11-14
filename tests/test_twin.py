"""Tests for the digital-twin generator (Pillar 2 step 1)."""
import numpy as np

from mxc.structural import demand as dm
from mxc.twin import generate as tw
from test_demand import random_market  # noqa: F401

QUIET = dict(xi_perm_sd=0.0, xi_sd=0.0, cost_region_sd=0.0, cost_store_sd=0.0, cost_obs_noise_sd=0.0, flyer_share=0.0,
             promo_share=0.0, price_noise_sd=0.0, fx_sd=0.0, sales_noise_sd=0.0, cat_noise_sd=0.0, missing_rate=0.0,
             outage_store_share=0.0)


def run(random_market, open_week=None, **kw):
    theta, a, *_ = random_market
    J = int(a["store_banner"].shape[0])
    nest = np.array([dm.NESTS.index(dm.BANNER_NEST[dm.BANNERS[i]]) for i in np.asarray(a["store_banner"])])
    cfg = tw.TwinConfig(weeks=12, **kw)
    return theta, a, tw.generate(theta, a, np.array(["ZH"] * J), nest, cfg, open_week=open_week)


def test_quiet_twin_equals_pillar1_model_times_market_size(random_market):
    theta, a, out = run(random_market, **QUIET)
    q_model = np.asarray(dm.quantities(theta, a)[0])
    sw, wk = out["store_week"], out["weekly"]
    M = wk.market_size.to_numpy()[sw.week.to_numpy()]
    np.testing.assert_allclose(sw.baskets / M, q_model[sw.store.to_numpy()], rtol=1e-10)
    np.testing.assert_allclose(sw.price, np.asarray(theta.p)[np.asarray(a["store_banner"])][sw.store], rtol=1e-12)


def test_category_sales_add_up_to_revenue_without_noise(random_market):
    _, _, out = run(random_market, **QUIET)
    by = out["category"].groupby(["store", "week"], observed=True).sales_chf.sum()
    rev = out["store_week"].set_index(["store", "week"]).revenue
    np.testing.assert_allclose(by.loc[rev.index].to_numpy(), rev.to_numpy(), rtol=1e-5)


def test_closed_store_has_no_rows_and_rivals_gain_when_it_opens(random_market):
    theta, a, *_ = random_market
    J = int(a["store_banner"].shape[0])
    ow = np.zeros(J, int)
    ow[1] = 6                                                      # store 1 opens in week 6
    _, _, out = run(random_market, open_week=ow, **QUIET)
    sw = out["store_week"]
    assert set(sw[sw.store == 1].week) == set(range(6, 12))
    q = sw.assign(qn=sw.baskets / out["weekly"].market_size.to_numpy()[sw.week]).pivot(index="week", columns="store",
                                                                                      values="qn")
    others = [j for j in q.columns if j != 1]
    assert (q.loc[5, others] >= q.loc[6, others] - 1e-12).all()   # nobody gains when a rival opens
    assert (q.loc[5, others] > q.loc[6, others]).any()


def test_same_seed_same_twin(random_market):
    _, _, o1 = run(random_market)
    _, _, o2 = run(random_market)
    assert o1["store_week"].equals(o2["store_week"])
