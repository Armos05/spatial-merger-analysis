"""Tests for twin events and the loyalty panel (Pillar 2 step 2)."""
import jax.numpy as jnp
import numpy as np
import pandas as pd

from mxc.structural import demand as dm
from mxc.twin import events as evm
from mxc.twin import loyalty as ly
from test_demand import random_market  # noqa: F401


def test_draw_events_respects_window_and_banners():
    sb = np.array([dm.BANNERS.index(b) for b in ["Aldi"] * 30 + ["Lidl"] * 30 + ["Spar"] * 20 + ["Coop"] * 20])
    ow, cw, ev = evm.draw_events(1, sb, 156, n_entry=25, n_close=10)
    ent = ev[ev.event == "entry"]
    assert len(ent) == 25 and ent.banner.isin(["Aldi", "Lidl"]).all()
    assert ent.week.between(40, 130).all() and (ow[ent.store] == ent.week).all()
    assert (ev[ev.event == "closure"].banner == "Spar").all() and (cw < 10 ** 6).sum() == 10


def test_closure_diversion_adds_up_to_one(random_market):
    theta, a, *_ = random_market
    J = int(a["store_banner"].shape[0])
    ev = pd.DataFrame({"store": [2], "event": ["closure"], "week": [5]})
    rows, summ = evm.closure_truth(theta, a, np.zeros(J), ev, np.zeros((J, 2)), radius_km=1.0)
    np.testing.assert_allclose(summ.div_within_radius + summ.div_abroad, 1.0, rtol=1e-8)
    assert (rows.diversion >= 0).all()


def test_mainstream_household_matches_its_location_and_type(random_market):
    theta, a, *_ = random_market
    L = int(a["hh"].shape[0])
    alpha_r = np.asarray(dm.alpha_types(theta, a["type_y"]))
    hh = pd.DataFrame({"loc": np.arange(L), "segment": "mainstream", "alpha_i": alpha_r[1], "gamma_mult": 1.0,
                       "kappa_mult": 1.0, "border_min": np.asarray(a["border_min"])})
    ah = ly.household_arrays(a, hh)
    s_h, s0_h, _ = dm.choice_probs(theta, ah, jnp.full((L, 1), alpha_r[1]))
    s, s0, _ = dm.choice_probs(theta, a)
    np.testing.assert_allclose(np.asarray(s_h[:, 0]), np.asarray(s[:, 1]), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(s0_h[:, 0]), np.asarray(s0[:, 1]), rtol=1e-10)


def test_trip_frequencies_match_choice_probabilities(random_market):
    theta, a, *_ = random_market
    J, T = int(a["store_banner"].shape[0]), 400
    alpha_r = np.asarray(dm.alpha_types(theta, a["type_y"]))
    hh = pd.DataFrame({"hh_id": [0], "loc": [0], "segment": ["mainstream"], "alpha_i": [alpha_r[1]],
                       "gamma_mult": [1.0], "kappa_mult": [1.0], "trip_rate": [20.0], "basket_size": [1.0],
                       "border_min": [float(a["border_min"][0])], "cumulus": [True], "supercard": [True]})
    p = np.asarray(theta.p)[np.asarray(a["store_banner"])]
    arrays = {"price": np.tile(p, (T, 1)), "xi_t": np.zeros((T, J)), "active": np.ones((T, J), bool),
              "xi_perm": np.zeros(J), "M": np.ones(T), "p0": np.full(T, float(theta.p_abroad))}
    trips = ly.simulate_trips(theta, a, hh, arrays)
    s, s0, _ = dm.choice_probs(theta, a)
    pl, ps = np.asarray(a["pair_loc"]), np.asarray(a["pair_store"])
    truth = pd.Series(np.asarray(s[:, 1])[pl == 0], index=ps[pl == 0])
    freq = trips.store.value_counts(normalize=True)
    assert abs(freq.get(-1, 0) - float(s0[0, 1])) < 0.01
    for j, pj in truth.items():
        assert abs(freq.get(j, 0) - pj) < 0.015
