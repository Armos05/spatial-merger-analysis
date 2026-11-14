"""Spatial nested-logit demand for Swiss grocery stores (DESIGN §3.2).

Every shopper location h has households of R income types r. Each household picks one store j
within a 25-minute drive, or shops abroad (option 0):

    V_hjr = delta_b(j) + xi_j - alpha_r * p_b(j) - gamma * time_hj
    V_h0r = - alpha_r * p_abroad - kappa * border_min_h          (abroad: delta normalised to 0)

    alpha_r = alpha_ref * (y_r / y_ref) ** (-eta)                  (richer households: less price-sensitive)

Nests g: full-service, discount, convenience, other; abroad sits alone. lam = 1 - sigma.

    P(j | g) = exp(V_j / lam) / D_g,      D_g = sum_{k in g} exp(V_k / lam)
    I_g      = lam * ln D_g                                       (inclusive value of nest g)
    P(g)     = exp(I_g) / [ exp(V_0) + sum_g' exp(I_g') ]
    s_hjr    = P(g) * P(j | g)

The computation runs on flat (location, store) pairs, with segment sums per (location, nest).
That way 32k locations x 4.5k stores never becomes a dense matrix. Everything is written in JAX,
so later steps (calibration, Jacobian, equilibrium, implicit differentiation) can differentiate
straight through it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------------- banners and nests
NESTS = ["full-service", "discount", "convenience", "other"]
BANNER_NEST = {
    "Migros": "full-service", "Migros Partner": "full-service", "Coop": "full-service",
    "Denner": "discount", "Aldi": "discount", "Lidl": "discount",
    "Migrolino": "convenience", "Coop Pronto": "convenience",
    "Volg": "other", "Landi": "other", "Spar": "other", "Manor": "other",
}
BANNERS = list(BANNER_NEST)

# Weekly basket price index, Migros = 100. [assumed]: no public local shelf prices (Koh 2025 route
# in DESIGN §3.4). Ranges are tested in the uncertainty step.
PRICE_INDEX = {
    "Migros": 100.0, "Migros Partner": 103.0, "Coop": 102.0,
    "Denner": 88.0, "Aldi": 85.0, "Lidl": 85.0,
    "Migrolino": 115.0, "Coop Pronto": 115.0,
    "Volg": 108.0, "Landi": 100.0, "Spar": 108.0, "Manor": 110.0,
}
P_ABROAD = 70.0  # same basket across the border [assumed]; Swiss food prices are far above DE/FR/IT/AT

# ESTV net-income classes merged into R = 7 income types, with a representative income (CHF/yr).
# '0' joins the lowest class; everything above 200k forms one open top class.
INCOME_TYPES = {
    "<30k": (["0", "1-30k"], 20_000), "30-40k": (["30-40k"], 35_000), "40-50k": (["40-50k"], 45_000),
    "50-75k": (["50-75k"], 62_500), "75-100k": (["75-100k"], 87_500),
    "100-200k": (["100-200k"], 150_000), "200k+": (["200-500k", "500k-1m", "1m+"], 300_000),
}
Y_REF = 62_500.0  # reference income: the class holding the national median taxpayer


class Params(NamedTuple):
    """Demand parameters. JAX treats a NamedTuple as a pytree, so all of them are differentiable."""
    delta: jnp.ndarray      # (B,) banner quality; calibrated in Phase 2 step 2
    alpha_ref: float        # price sensitivity at Y_REF, per CHF of basket price
    eta: float              # income elasticity of price sensitivity (0 = none, 1 = alpha proportional to 1/income)
    gamma: float            # disutility per drive minute to a store
    kappa: float            # disutility per drive minute to the border
    sigma: float            # within-nest correlation; lam = 1 - sigma
    p: jnp.ndarray          # (B,) banner basket price
    p_abroad: float


def prior_params() -> Params:
    """Starting values (DESIGN §3.2), all [assumed]. Steps 2 and 9 calibrate them and vary them.

    alpha_ref = 0.03  gives banner own-price elasticities of about -2 to -4 at p ~ 100.
    gamma     = 0.06  so gamma / alpha_ref = CHF 2 per one-way minute (time + car + habit).
    kappa     = 0.04  border minutes weigh less than store minutes: abroad trips are planned and bulk.
    sigma     = 0.25  the crux parameter; tested over 0-0.6.
    eta       = 0.5   between no income effect (0) and constant budget share (1).
    """
    B = len(BANNERS)
    return Params(delta=jnp.zeros(B), alpha_ref=0.03, eta=0.5, gamma=0.06, kappa=0.04, sigma=0.25,
                  p=jnp.array([PRICE_INDEX[b] for b in BANNERS]), p_abroad=P_ABROAD)


# ----------------------------------------------------------------------------- market structure
@dataclass
class Market:
    """Flat arrays describing who can reach what. Pairs are sorted by location."""
    pair_loc: np.ndarray      # (P,) location index of each pair
    pair_store: np.ndarray    # (P,) store index
    pair_time: np.ndarray     # (P,) drive minutes
    pair_seg: np.ndarray      # (P,) segment id = location * G + nest
    store_banner: np.ndarray  # (J,) banner index
    store_nest: np.ndarray    # (J,) nest index
    store_group: np.ndarray   # (J,) owner name
    store_id: np.ndarray      # (J,) OSM-based store id
    hh: np.ndarray            # (L,) private households
    type_w: np.ndarray        # (L, R) share of households in each income type
    type_y: np.ndarray        # (R,) representative income
    border_min: np.ndarray    # (L,) drive minutes to the nearest road border crossing
    loc_id: np.ndarray        # (L,) original location id
    canton: np.ndarray        # (L,) canton code

    @property
    def L(self): return len(self.hh)

    @property
    def J(self): return len(self.store_banner)

    @property
    def G(self): return len(NESTS)


def income_type_weights(municipalities: pd.DataFrame, bfs: np.ndarray) -> np.ndarray:
    """(L, R) share of taxpayers in each income type, read from the location's municipality (ESTV 2023).

    Locations whose municipality has no ESTV row (mergers) get the national distribution [imputed].
    """
    counts = np.column_stack([municipalities[[f"n_{c}" for c in cls]].sum(1).to_numpy()
                              for cls, _ in INCOME_TYPES.values()])
    national = counts.sum(0) / counts.sum()
    tot = counts.sum(1, keepdims=True)
    w = np.where(tot > 0, counts / np.where(tot > 0, tot, 1.0), national[None, :])
    pos = pd.Index(municipalities.index).get_indexer(bfs)
    return np.where(pos[:, None] >= 0, w[np.clip(pos, 0, None)], national[None, :])


def build_market(locations: pd.DataFrame, stores: pd.DataFrame, drive_times: pd.DataFrame,
                 municipalities: pd.DataFrame, cutoff_min: float = 25.0) -> tuple[Market, dict]:
    """Assemble the choice sets. Locations with no store within the cutoff are dropped and reported."""
    stores = stores[stores.banner.isin(BANNERS)].reset_index(drop=True)
    dt = drive_times[drive_times.drive_min <= cutoff_min]
    dt = dt[dt.store_id.isin(stores.store_id)]
    reach = dt.groupby("loc_id").size()
    keep = locations[locations.loc_id.isin(reach.index) & (locations.households > 0)].reset_index(drop=True)
    dropped = locations[~locations.loc_id.isin(keep.loc_id)]

    lpos = pd.Series(np.arange(len(keep)), index=keep.loc_id)
    spos = pd.Series(np.arange(len(stores)), index=stores.store_id)
    dt = dt[dt.loc_id.isin(keep.loc_id)]
    pl = lpos.loc[dt.loc_id].to_numpy()
    ps = spos.loc[dt.store_id].to_numpy()
    order = np.lexsort((ps, pl))
    pl, ps, pt = pl[order], ps[order], dt.drive_min.to_numpy(np.float64)[order]

    banner = stores.banner.map({b: i for i, b in enumerate(BANNERS)}).to_numpy()
    nest = stores.banner.map(lambda b: NESTS.index(BANNER_NEST[b])).to_numpy()
    mkt = Market(
        pair_loc=pl, pair_store=ps, pair_time=pt, pair_seg=pl * len(NESTS) + nest[ps],
        store_banner=banner, store_nest=nest, store_group=stores.group.to_numpy(),
        store_id=stores.store_id.to_numpy(), hh=keep.households.to_numpy(np.float64),
        type_w=income_type_weights(municipalities, keep.bfs.to_numpy()),
        type_y=np.array([y for _, y in INCOME_TYPES.values()], dtype=np.float64),
        border_min=keep.border_min.to_numpy(np.float64), loc_id=keep.loc_id.to_numpy(),
        canton=keep.canton.to_numpy(),
    )
    info = {"locations_kept": len(keep), "locations_dropped": len(dropped),
            "households_dropped": int(dropped.households.sum()), "pairs": len(pl), "stores": len(stores)}
    return mkt, info


# ----------------------------------------------------------------------------- choice probabilities
def alpha_types(theta: Params, type_y) -> jnp.ndarray:
    """(R,) price sensitivity per income type: alpha_r = alpha_ref * (y_r / Y_REF) ** (-eta)."""
    return theta.alpha_ref * (jnp.asarray(type_y) / Y_REF) ** (-theta.eta)


def _segment_logsumexp(x, seg, n):
    """log sum exp of x within segments; empty segments give -inf. x: (P, R)."""
    m = jax.ops.segment_max(x, seg, num_segments=n)
    m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
    s = jax.ops.segment_sum(jnp.exp(x - m_safe[seg]), seg, num_segments=n)
    return jnp.where(s > 0, jnp.log(jnp.where(s > 0, s, 1.0)) + m_safe, -jnp.inf)


def choice_probs(theta: Params, mkt_arrays: dict, alpha_lr=None):
    """Nested-logit probabilities.

    Returns s_pair (P, R), s_abroad (L, R) and logsum (L, R) = ln[exp(V_0) + sum_g exp(I_g)],
    which gives consumer surplus CS = logsum / alpha. alpha_lr (L, R) overrides alpha_types
    (used to reproduce the toy, where alpha varies by location).
    """
    a = mkt_arrays
    L, G = a["hh"].shape[0], len(NESTS)
    lam = 1.0 - theta.sigma
    if alpha_lr is None:
        alpha_lr = jnp.broadcast_to(alpha_types(theta, a["type_y"]), (L, a["type_y"].shape[0]))
    b = a["store_banner"][a["pair_store"]]
    xi = a.get("xi_store", jnp.zeros(a["store_banner"].shape[0]))[a["pair_store"]]
    V = (theta.delta[b] + xi)[:, None] - alpha_lr[a["pair_loc"]] * theta.p[b][:, None] \
        - theta.gamma * a["pair_time"][:, None]                                   # (P, R)
    V0 = -alpha_lr * theta.p_abroad - theta.kappa * a["border_min"][:, None]      # (L, R)

    lnD = _segment_logsumexp(V / lam, a["pair_seg"], L * G)                        # (L*G, R)
    I = (lam * lnD).reshape(L, G, -1)                                              # inclusive values
    logsum = jax.scipy.special.logsumexp(jnp.concatenate([V0[:, None, :], I], axis=1), axis=1)  # (L, R)
    lnPg = I - logsum[:, None, :]                                                  # ln P(g)
    lnPjg = V / lam - lnD[a["pair_seg"]]                                           # ln P(j | g)
    s_pair = jnp.exp(lnPg.reshape(L * G, -1)[a["pair_seg"]] + lnPjg)
    s_abroad = jnp.exp(V0 - logsum)
    return s_pair, s_abroad, logsum


def to_arrays(mkt: Market, xi_store=None) -> dict:
    """Market as a dict of JAX arrays (what choice_probs consumes)."""
    d = {k: jnp.asarray(getattr(mkt, k)) for k in
         ["pair_loc", "pair_store", "pair_time", "pair_seg", "store_banner", "hh", "type_w", "type_y", "border_min"]}
    if xi_store is not None:
        d["xi_store"] = jnp.asarray(xi_store)
    return d


def quantities(theta: Params, a: dict, alpha_lr=None):
    """Weekly baskets per store (J,) and abroad (scalar): q_j = sum_h sum_r hh_h * w_hr * s_hjr."""
    s_pair, s_abroad, _ = choice_probs(theta, a, alpha_lr)
    n_lr = a["hh"][:, None] * a["type_w"]                                         # households (L, R)
    q_pair = (n_lr[a["pair_loc"]] * s_pair).sum(1)
    q_store = jax.ops.segment_sum(q_pair, a["pair_store"], num_segments=a["store_banner"].shape[0])
    return q_store, (n_lr * s_abroad).sum()


def banner_shares(theta: Params, a: dict) -> pd.Series:
    """National quantity shares by banner plus 'abroad'."""
    q_store, q_abroad = quantities(theta, a)
    qb = jax.ops.segment_sum(q_store, a["store_banner"], num_segments=len(BANNERS))
    total = float(a["hh"].sum())
    return pd.Series(np.r_[np.asarray(qb), float(q_abroad)] / total, index=BANNERS + ["abroad"])
