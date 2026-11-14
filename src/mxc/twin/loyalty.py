"""Loyalty-card panel: individual households and their weekly trips (Pillar 2 step 2; input to Pillar 3). [sim]

Households are drawn from the real shopper locations (∝ households) and their income type. Each gets a hidden
segment and an individual price sensitivity:

    α_i = α_r × seg.alpha × exp(N(0, alpha_sd²))          (α_r = Pillar 1 income-type sensitivity)
    γ_i = γ × seg.gamma,  κ_i = κ × seg.kappa,  + seg.nest_taste[nest]

Utility of store j for household i in week t (same nested logit, same σ, the twin's prices and shocks):
    V_ijt = δ_B + ξ_j + ξ_jt − α_i p_jt − γ_i t_ij + taste_i,nest(j)
    V_i0t = −α_i p0_t − κ_i border_i

Each week a household makes Poisson(trips_i × M_t) trips; each trip picks a store (or abroad) with these
probabilities. Spend = store price × basket size_i × noise; spend is split into 4 departments.

What each chain sees: Cumulus records trips at Migros, Migros Partner and Migrolino by Cumulus holders who swipe;
Supercard records trips at Coop and Coop Pronto by Supercard holders who swipe. After a merger, one firm sees both.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..structural import demand as dm

DEPARTMENTS = ["fresh", "staples", "treats", "non_food"]
DEPT_BASE = np.array([0.40, 0.30, 0.15, 0.15])
SEGMENTS = {   # [sim]
    "mainstream":      dict(weight=.35, alpha=1.0, gamma=1.0, kappa=1.0, trips=1.0, size=1.0, taste={},
                            dept=(1.0, 1.0, 1.0, 1.0)),
    "price_hunter":    dict(weight=.20, alpha=1.4, gamma=0.9, kappa=0.9, trips=1.0, size=1.1, taste={"discount": .5},
                            dept=(0.8, 1.4, 0.9, 1.1)),
    "convenience":     dict(weight=.15, alpha=0.9, gamma=1.6, kappa=1.3, trips=1.5, size=0.6,
                            taste={"convenience": .6}, dept=(0.8, 0.7, 1.6, 0.8)),
    "quality":         dict(weight=.20, alpha=0.7, gamma=1.0, kappa=1.2, trips=0.9, size=1.2,
                            taste={"full-service": .3, "discount": -.4}, dept=(1.4, 0.8, 1.0, 0.7)),
    "border_shopper":  dict(weight=.10, alpha=1.2, gamma=1.0, kappa=0.5, trips=1.0, size=1.3, taste={},
                            dept=(0.9, 1.2, 1.0, 1.2)),
}
CUMULUS = ("Migros", "Migros Partner", "Migrolino")
SUPERCARD = ("Coop", "Coop Pronto")


@dataclass
class LoyaltyConfig:
    seed: int = 2027
    n_households: int = 20_000
    alpha_sd: float = 0.15          # within-segment spread of price sensitivity
    size_sd: float = 0.30           # household basket size spread (log)
    spend_sd: float = 0.35          # trip-to-trip spend noise (log)
    dept_conc: float = 30.0         # Dirichlet concentration of department shares
    p_cumulus: float = 0.80         # share of households holding the card
    p_supercard: float = 0.70
    swipe_rate: float = 0.90        # share of trips where a holder shows the card
    border_near_min: float = 20.0   # border shoppers twice as likely within this drive time


def draw_households(theta, a, cfg: LoyaltyConfig = LoyaltyConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 2)
    hh, type_w = np.asarray(a["hh"]), np.asarray(a["type_w"])
    type_y, border = np.asarray(a["type_y"]), np.asarray(a["border_min"])
    H = cfg.n_households
    loc = rng.choice(len(hh), H, p=hh / hh.sum())
    cum = type_w[loc].cumsum(1)
    r = np.minimum((rng.random(H)[:, None] > cum).sum(1), type_w.shape[1] - 1)
    names = list(SEGMENTS)
    w = np.array([SEGMENTS[s]["weight"] for s in names])[None, :].repeat(H, 0)
    rel = (type_y[r] / dm.Y_REF) ** 0.5
    w[:, names.index("price_hunter")] /= rel                                    # poorer → more price hunters
    w[:, names.index("quality")] *= rel                                         # richer → more quality shoppers
    w[:, names.index("border_shopper")] *= np.where(border[loc] < cfg.border_near_min, 2.0, 0.5)
    w /= w.sum(1, keepdims=True)
    seg = (rng.random(H)[:, None] > w.cumsum(1)).sum(1)
    alpha_r = np.asarray(dm.alpha_types(theta, a["type_y"]))[r]
    par = {k: np.array([SEGMENTS[names[s]][k] for s in seg]) for k in ("alpha", "gamma", "kappa", "trips", "size")}
    return pd.DataFrame({
        "hh_id": np.arange(H), "loc": loc, "income_type": r, "income": type_y[r], "segment": np.array(names)[seg],
        "alpha_i": alpha_r * par["alpha"] * np.exp(rng.normal(0, cfg.alpha_sd, H)),
        "gamma_mult": par["gamma"], "kappa_mult": par["kappa"], "trip_rate": par["trips"],
        "basket_size": par["size"] * np.exp(rng.normal(0, cfg.size_sd, H)),
        "border_min": border[loc],
        "cumulus": rng.random(H) < cfg.p_cumulus, "supercard": rng.random(H) < cfg.p_supercard})


def household_arrays(a, households: pd.DataFrame) -> dict:
    """A 'market' where every household is its own location with one type."""
    pl = np.asarray(a["pair_loc"])
    start = np.searchsorted(pl, np.arange(len(a["hh"])))
    end = np.searchsorted(pl, np.arange(len(a["hh"])), side="right")
    loc = households["loc"].to_numpy()
    n = end[loc] - start[loc]
    idx = np.concatenate([np.arange(s, e) for s, e in zip(start[loc], end[loc])])
    hh_of_pair = np.repeat(np.arange(len(loc)), n)
    ps = np.asarray(a["pair_store"])[idx]
    sb = np.asarray(a["store_banner"])
    G = len(dm.NESTS)
    nest = np.array([dm.NESTS.index(dm.BANNER_NEST[b]) for b in dm.BANNERS])[sb[ps]]
    seg = households.segment.to_numpy()[hh_of_pair]
    taste = np.zeros(len(idx))
    for s, spec in SEGMENTS.items():
        for g, v in spec["taste"].items():
            taste[(seg == s) & (nest == dm.NESTS.index(g))] = v
    return {"pair_loc": jnp.asarray(hh_of_pair), "pair_store": jnp.asarray(ps),
            "pair_time": jnp.asarray(np.asarray(a["pair_time"])[idx] * households.gamma_mult.to_numpy()[hh_of_pair]),
            "pair_seg": jnp.asarray(hh_of_pair * G + nest), "pair_xi": jnp.asarray(taste),
            "store_banner": a["store_banner"], "hh": jnp.ones(len(loc)), "type_w": jnp.ones((len(loc), 1)),
            "type_y": jnp.array([dm.Y_REF]),
            "border_min": jnp.asarray(households.border_min.to_numpy() * households.kappa_mult.to_numpy())}


def simulate_trips(theta, a, households: pd.DataFrame, arrays: dict, cfg: LoyaltyConfig = LoyaltyConfig()):
    """Returns the full trip log (truth). arrays = generate(...)['arrays'] of the twin these households live in."""
    rng = np.random.default_rng(cfg.seed + 3)
    ah = household_arrays(a, households)
    H = len(households)
    alpha = jnp.asarray(households.alpha_i.to_numpy())[:, None]
    hh_of_pair = np.asarray(ah["pair_loc"])
    start = np.searchsorted(hh_of_pair, np.arange(H))

    @jax.jit
    def probs(theta, ah, p, xi, p0, alpha):
        s, s0, _ = dm.choice_probs(theta._replace(p_abroad=p0), {**ah, "p_store": p, "xi_store": xi}, alpha)
        return s[:, 0], s0[:, 0]

    price, xi_t, active, xi_perm = arrays["price"], arrays["xi_t"], arrays["active"], arrays["xi_perm"]
    M, p0 = arrays["M"], arrays["p0"]
    sb = np.asarray(a["store_banner"])
    seg_idx = pd.Categorical(households.segment, categories=list(SEGMENTS)).codes
    dept_mean = DEPT_BASE[None, :] * np.array([SEGMENTS[s]["dept"] for s in SEGMENTS])[seg_idx]
    dept_mean /= dept_mean.sum(1, keepdims=True)
    out = []
    for t in range(len(M)):
        xi = np.where(active[t], xi_perm + xi_t[t], -50.0)
        s, s0 = (np.asarray(x) for x in probs(theta, ah, jnp.asarray(price[t]), jnp.asarray(xi), p0[t], alpha))
        n = rng.poisson(households.trip_rate.to_numpy() * M[t])
        h = np.repeat(np.arange(H), n)
        u = rng.random(len(h))
        C = np.cumsum(s)
        off = np.where(start[h] > 0, C[start[h] - 1], 0.0)
        k = np.searchsorted(C, off + (u - s0[h]), side="right")
        end = np.r_[start[1:], len(s)] - 1
        k = np.minimum(k, end[h])
        abroad = u < s0[h]
        store = np.where(abroad, -1, np.asarray(ah["pair_store"])[k])
        p_trip = np.where(abroad, p0[t], price[t][np.maximum(store, 0)])
        spend = p_trip * households.basket_size.to_numpy()[h] * np.exp(rng.normal(0, cfg.spend_sd, len(h)))
        dept = rng.gamma(cfg.dept_conc * dept_mean[h])
        dept /= dept.sum(1, keepdims=True)
        out.append(pd.DataFrame({"hh_id": h.astype(np.int32), "week": np.int16(t), "store": store.astype(np.int32),
                                 "spend": spend.astype(np.float32),
                                 **{f"dept_{d}": dept[:, i].astype(np.float32) for i, d in enumerate(DEPARTMENTS)}}))
    trips = pd.concat(out, ignore_index=True)
    names = np.array(dm.BANNERS + ["abroad"])
    trips["banner"] = pd.Categorical(names[np.where(trips.store >= 0, sb[trips.store.clip(0)], len(dm.BANNERS))])
    swipe = rng.random(len(trips)) < cfg.swipe_rate
    trips["seen_cumulus"] = trips.banner.isin(CUMULUS) & households.cumulus.to_numpy()[trips.hh_id] & swipe
    trips["seen_supercard"] = trips.banner.isin(SUPERCARD) & households.supercard.to_numpy()[trips.hh_id] & swipe
    return trips


def config_table(cfg: LoyaltyConfig = LoyaltyConfig()) -> pd.DataFrame:
    rows = [{"block": "panel", "parameter": k, "value": v} for k, v in asdict(cfg).items()]
    for s, spec in SEGMENTS.items():
        rows += [{"block": f"segment {s}", "parameter": k, "value": str(v)} for k, v in spec.items()]
    return pd.DataFrame(rows)
