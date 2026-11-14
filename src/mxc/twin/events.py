"""Twin events: staggered Aldi/Lidl openings and a closure wave, with their true effects (Pillar 2 step 2). [sim]

Entry: n_entry real Aldi/Lidl stores are "not yet open" at the start and open at random weeks in a window.
Timing is random (as good as a lottery), so the true effect is clean; locations are the real ones.
Closure wave: n_close stores of one banner close in the same week (a chain restructuring).

Answer key at normal times (base prices, permanent store quality, no shocks, M = 1):
    entry of e:     Δq_k = q_k(e open) − q_k(e closed),   loss_k = −Δq_k / q_k,   take_k = −Δq_k / q_e
    closure of c:   gain_k = q_k(c closed) − q_k(c open), diversion D(c→k) = gain_k / q_c
take_k is the share of the entrant's sales that came from store k (the "diversion to the entrant", T1's target).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..structural import demand as dm
from . import generate as tw


def draw_events(seed: int, store_banner: np.ndarray, weeks: int, n_entry=120, entry_window=(40, 130),
                entry_banners=("Aldi", "Lidl"), n_close=40, close_at=104, close_banner="Spar"):
    rng = np.random.default_rng(seed + 1)
    names = np.array(dm.BANNERS)[store_banner]
    J = len(store_banner)
    open_week, close_week = np.zeros(J, int), np.full(J, 10 ** 6)
    ent = rng.choice(np.where(np.isin(names, entry_banners))[0], n_entry, replace=False)
    open_week[ent] = rng.integers(entry_window[0], entry_window[1] + 1, n_entry)
    clo = rng.choice(np.where(names == close_banner)[0], n_close, replace=False)
    close_week[clo] = close_at
    ev = pd.DataFrame({"store": np.r_[ent, clo], "event": ["entry"] * n_entry + ["closure"] * n_close,
                       "week": np.r_[open_week[ent], close_week[clo]]})
    ev["banner"] = names[ev.store]
    return open_week, close_week, ev.sort_values(["event", "week"]).reset_index(drop=True)


def _q(theta, a, wd, xi_perm, closed):
    sb = np.asarray(a["store_banner"])
    xi = np.where(closed, tw.CLOSED_XI, xi_perm)
    q, q0 = wd(theta, jnp.asarray(np.asarray(theta.p)[sb]), jnp.asarray(xi), float(theta.p_abroad))
    return np.asarray(q), float(q0)


def entry_truth(theta, a, xi_perm, events, xy_km, radius_km=20.0):
    """One row per (entrant, nearby store): true loss when the entrant opens, all other entrants closed."""
    wd = tw.make_week_demand(a)
    ent = events.loc[events.event == "entry", "store"].to_numpy()
    closed0 = np.zeros(len(xi_perm), bool)
    closed0[ent] = True
    q_base, q0_base = _q(theta, a, wd, xi_perm, closed0)
    rows, abroad = [], []
    for e in ent:
        closed = closed0.copy()
        closed[e] = False
        q1, q01 = _q(theta, a, wd, xi_perm, closed)
        dist = np.hypot(*(xy_km - xy_km[e]).T)
        near = np.where((dist <= radius_km) & (np.arange(len(dist)) != e) & ~closed0)[0]
        dq = q1[near] - q_base[near]
        rows.append(pd.DataFrame({"entrant": e, "store": near, "dist_km": dist[near], "q_base": q_base[near],
                                  "dq": dq, "loss_pct": -100 * dq / q_base[near], "take_share": -dq / q1[e]}))
        abroad.append({"entrant": e, "q_entrant": q1[e], "take_abroad": -(q01 - q0_base) / q1[e],
                       "take_within_radius": float(-dq.sum() / q1[e]),
                       "take_all_stores": float(-(q1 - q_base)[np.arange(len(q1)) != e].sum() / q1[e])})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(abroad)


def closure_truth(theta, a, xi_perm, events, xy_km, radius_km=20.0):
    """One row per (closed store, nearby store): where the closed store's shoppers go. Entrants that open
    before the closure week are open; later ones closed."""
    wd = tw.make_week_demand(a)
    ent = events[events.event == "entry"]
    clo = events[events.event == "closure"]
    closed0 = np.zeros(len(xi_perm), bool)
    closed0[ent.store[ent.week > clo.week.min()]] = True
    q_base, q0_base = _q(theta, a, wd, xi_perm, closed0)
    rows, summ = [], []
    for c in clo.store:
        closed = closed0.copy()
        closed[c] = True
        q1, q01 = _q(theta, a, wd, xi_perm, closed)
        dist = np.hypot(*(xy_km - xy_km[c]).T)
        near = np.where((dist <= radius_km) & (np.arange(len(dist)) != c) & ~closed0)[0]
        gain = q1[near] - q_base[near]
        rows.append(pd.DataFrame({"closed": c, "store": near, "dist_km": dist[near], "gain": gain,
                                  "diversion": gain / q_base[c]}))
        summ.append({"closed": c, "q_closed": q_base[c], "div_abroad": (q01 - q0_base) / q_base[c],
                     "div_within_radius": float(gain.sum() / q_base[c])})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summ)
