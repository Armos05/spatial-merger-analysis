"""Digital twin: a simulated replica of Swiss grocery on real geography (Pillar 2 step 1; DESIGN §4.2).

Everything here is [sim]: the twin is a test bed where the truth is known, never a finding about Switzerland.

Truth = the Pillar 1 demand model (calibrated δ, α, η, γ, κ, σ) on the real stores and shopper locations, plus:

    store quality      ξ_j       ~ N(0, xi_perm_sd²), demeaned within banner      (permanent, hidden)
    demand shock       ξ_jt      = rho_xi ξ_j,t−1 + N(0, xi_sd²)                  (hidden from the analyst)
    cost shock         c_jt      = c_region,t + c_store,t   (AR(1) each, region = banner × canton)
    price              ln p_jt   = ln p_B + pass × c_jt + price_on_xi × ξ_jt
                                   − flyer_depth × flyer_Bt − promo_depth × promo_jt + noise
    market size        M_t       = 1 + seasonality (Christmas, Easter, summer holidays, New Year)
    abroad price       p0_t      = 70 × exp(fx_t),  fx_t AR(1)

    true baskets       q_jt      = M_t × Σ_h Σ_r n_hr s_hjr(p_t, ξ + ξ_t, p0_t)      (Pillar 1 nested logit)
    observed baskets   q_obs_jt  = q_jt × exp(N(0, sales_noise_sd²)), some store-weeks missing

price_on_xi > 0 is the endogeneity: managers raise prices when local demand is strong, so a naive regression
of quantity on price is biased towards zero. The cost shock c_jt moves price but not demand: a valid instrument.

Category layer: each store-week basket is split into 20 categories with store-specific weights, category
seasonality and noise. It adds realism (and rows); the truth about substitution lives at the basket level.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..structural import demand as dm

CLOSED_XI = -50.0          # a store that is not open yet gets ξ = −50 → share ≈ 0

# name, basket weight, (summer, Christmas, Easter) seasonal amplitude
CATEGORIES = [
    ("fruit_veg", .14, (.10, .00, .00)), ("dairy", .11, (.00, .03, .02)), ("meat", .10, (.15, .08, .03)),
    ("bread_bakery", .07, (.00, .04, .05)), ("cheese", .06, (-.03, .15, .00)), ("soft_drinks", .06, (.25, .05, .00)),
    ("alcohol", .05, (.10, .30, .05)), ("coffee_tea", .03, (-.05, .05, .00)), ("frozen", .04, (.20, .00, .00)),
    ("sweets_chocolate", .05, (-.10, .45, .60)), ("snacks", .03, (.10, .15, .00)), ("pasta_rice", .03, (-.05, .00, .00)),
    ("canned_dry", .03, (-.05, .05, .00)), ("baking", .02, (-.10, .60, .15)), ("ready_meals", .04, (.00, -.05, .00)),
    ("fish", .02, (.00, .25, .20)), ("household", .05, (.00, .00, .00)), ("personal_care", .04, (.05, .05, .00)),
    ("baby", .01, (.00, .00, .00)), ("pet", .02, (.00, .00, .00)),
]
# how each nest tilts the basket (multiplies the weight before normalising)
NEST_TILT = {
    "discount": {"household": 1.4, "canned_dry": 1.4, "pasta_rice": 1.3, "fish": 0.7, "cheese": 0.8},
    "convenience": {"soft_drinks": 2.0, "snacks": 2.0, "ready_meals": 2.5, "alcohol": 1.5, "fruit_veg": 0.5,
                    "meat": 0.4, "household": 0.5, "baby": 0.3, "pet": 0.3},
    "other": {"pet": 2.0, "household": 1.3},
}
EASTER = {2022: "2022-04-17", 2023: "2023-04-09", 2024: "2024-03-31"}


@dataclass
class TwinConfig:
    seed: int = 2027
    weeks: int = 156
    start: str = "2022-01-03"            # Monday; 3 years of weeks [sim]
    xi_perm_sd: float = 0.15             # permanent store quality, utility units
    xi_rho: float = 0.7                  # demand-shock persistence
    xi_sd: float = 0.05                  # demand-shock innovation
    cost_region_sd: float = 0.010        # banner × canton cost shock (log)
    cost_store_sd: float = 0.015         # store cost shock (log): rent, staff, logistics
    cost_rho: float = 0.8
    cost_obs_noise_sd: float = 0.003     # the analyst sees costs with a little noise
    pass_through: float = 0.8            # log price per log cost
    price_on_xi: float = 0.5             # log price per unit of demand shock: the endogeneity
    flyer_share: float = 0.15            # share of banner-weeks with a national flyer
    flyer_depth: float = 0.03            # basket price cut in a flyer week
    promo_share: float = 0.05            # share of store-weeks with a local promotion
    promo_depth: float = 0.04
    price_noise_sd: float = 0.004
    fx_rho: float = 0.98
    fx_sd: float = 0.006                 # weekly EUR/CHF wobble → abroad price
    sales_noise_sd: float = 0.03         # scanner / measurement noise
    cat_noise_sd: float = 0.06
    missing_rate: float = 0.015          # random missing store-weeks
    outage_store_share: float = 0.02     # stores with a 4-12 week data outage


def season(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Weekly seasonal bumps in [0, 1]: summer holidays, Christmas run-up, Easter week, New Year."""
    wk = dates.isocalendar().week.to_numpy().astype(int)
    summer = ((wk >= 28) & (wk <= 32)).astype(float)
    xmas = np.select([wk == 51, wk == 50, wk == 52, wk == 49], [1.0, 0.6, 0.5, 0.3], 0.0)
    easter = np.zeros(len(dates))
    for y, d in EASTER.items():
        e = pd.Timestamp(d)
        easter[(dates <= e) & (dates > e - pd.Timedelta(days=7))] = 1.0
    newyear = (wk == 1).astype(float)
    return pd.DataFrame({"summer": summer, "xmas": xmas, "easter": easter, "newyear": newyear}, index=dates)


def market_size(s: pd.DataFrame) -> np.ndarray:
    """M_t: baskets per household per week relative to normal (Christmas +18%, Easter +6%, summer −6%, New Year −5%)."""
    t = np.arange(len(s))
    return (1 + 0.18 * s.xmas + 0.06 * s.easter - 0.06 * s.summer - 0.05 * s.newyear
            + 0.015 * np.cos(2 * np.pi * (t - 2) / 52)).to_numpy()


def ar1(rng, shape, rho, sd, T):
    """(T, *shape) stationary AR(1) paths."""
    x = np.empty((T, *shape))
    x[0] = rng.normal(0, sd / np.sqrt(1 - rho ** 2), shape)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + rng.normal(0, sd, shape)
    return x


def category_weights(rng, store_nest: np.ndarray) -> np.ndarray:
    """(J, C) basket weights: national weights × nest tilt × store noise (Dirichlet), rows sum to one."""
    names = [c[0] for c in CATEGORIES]
    base = np.array([c[1] for c in CATEGORIES])
    W = np.empty((len(store_nest), len(names)))
    for g, nest in enumerate(dm.NESTS):
        tilt = np.array([NEST_TILT.get(nest, {}).get(n, 1.0) for n in names])
        w = base * tilt
        idx = np.where(store_nest == g)[0]
        W[idx] = rng.dirichlet(200 * w / w.sum(), len(idx))
    return W


def category_season(s: pd.DataFrame) -> np.ndarray:
    """(T, C) category demand multipliers."""
    amp = np.array([c[2] for c in CATEGORIES])                           # (C, 3)
    return 1 + s[["summer", "xmas", "easter"]].to_numpy() @ amp.T


def make_week_demand(a: dict):
    """Jitted (theta, p_store, xi_store, p_abroad) → (q_store (J,), q_abroad)."""
    @jax.jit
    def f(theta, a, p_store, xi, p0):
        return dm.quantities(theta._replace(p_abroad=p0), {**a, "p_store": p_store, "xi_store": xi})
    return lambda theta, p, xi, p0: f(theta, a, p, xi, p0)


def generate(theta: dm.Params, a: dict, store_canton: np.ndarray, store_nest: np.ndarray,
             cfg: TwinConfig = TwinConfig(), open_week: np.ndarray | None = None, store_id=None):
    """Simulate the twin. Returns dict of DataFrames:

    store_week   what the analyst sees: price, observed baskets and revenue, promotions, observed cost index
    truth_week   hidden: true baskets, ξ_jt, cost shock, market size
    category     store × category × week observed sales (CHF), missing store-weeks dropped
    weekly       national: market size, abroad price, true abroad baskets
    store        store attributes and permanent quality ξ_j (hidden)

    open_week (J,): first week each store trades (default 0 = open throughout); used for entry events.
    """
    rng = np.random.default_rng(cfg.seed)
    T = cfg.weeks
    sb = np.asarray(a["store_banner"])
    J, B = len(sb), len(dm.BANNERS)
    dates = pd.date_range(cfg.start, periods=T, freq="7D")
    s = season(dates)
    M = market_size(s)
    open_week = np.zeros(J, int) if open_week is None else np.asarray(open_week)

    # permanent quality, demeaned within banner (keeps the calibrated banner shares roughly intact)
    xi_perm = rng.normal(0, cfg.xi_perm_sd, J)
    xi_perm -= np.bincount(sb, xi_perm, B)[sb] / np.maximum(np.bincount(sb, minlength=B), 1)[sb]
    xi_t = ar1(rng, (J,), cfg.xi_rho, cfg.xi_sd, T)

    # costs: region = banner × canton
    region = pd.factorize(pd.Series(sb).astype(str) + "|" + pd.Series(store_canton).astype(str))[0]
    c_reg = ar1(rng, (region.max() + 1,), cfg.cost_rho, cfg.cost_region_sd, T)[:, region]
    c_sto = ar1(rng, (J,), cfg.cost_rho, cfg.cost_store_sd, T)
    cost = c_reg + c_sto
    cost_obs = cost + rng.normal(0, cfg.cost_obs_noise_sd, cost.shape)

    flyer = (rng.random((T, B)) < cfg.flyer_share)[:, sb]
    promo = rng.random((T, J)) < cfg.promo_share
    p_base = np.asarray(theta.p)[sb]
    lnp = (np.log(p_base)[None] + cfg.pass_through * cost + cfg.price_on_xi * xi_t
           - cfg.flyer_depth * flyer - cfg.promo_depth * promo + rng.normal(0, cfg.price_noise_sd, (T, J)))
    price = np.exp(lnp)
    fx = ar1(rng, (), cfg.fx_rho, cfg.fx_sd, T)
    p0 = float(theta.p_abroad) * np.exp(fx)

    wd = make_week_demand(a)
    q = np.empty((T, J))
    q0 = np.empty(T)
    for t in range(T):
        xi = np.where(open_week <= t, xi_perm + xi_t[t], CLOSED_XI)
        qs, qa = wd(theta, jnp.asarray(price[t]), jnp.asarray(xi), p0[t])
        q[t], q0[t] = M[t] * np.asarray(qs), M[t] * float(qa)
    is_open = open_week[None, :] <= np.arange(T)[:, None]

    q_obs = q * np.exp(rng.normal(0, cfg.sales_noise_sd, (T, J)))
    missing = rng.random((T, J)) < cfg.missing_rate
    for j in rng.choice(J, int(cfg.outage_store_share * J), replace=False):
        start, length = rng.integers(0, T - 12), rng.integers(4, 13)
        missing[start:start + length, j] = True
    keep = is_open & ~missing

    tt, jj = np.nonzero(keep)
    store_id = np.arange(J) if store_id is None else np.asarray(store_id)
    store_week = pd.DataFrame({
        "store_id": store_id[jj], "store": jj.astype(np.int32), "week": tt.astype(np.int16), "date": dates[tt],
        "banner": np.array(dm.BANNERS)[sb[jj]], "price": price[tt, jj], "baskets": q_obs[tt, jj],
        "revenue": q_obs[tt, jj] * price[tt, jj], "flyer": flyer[tt, jj], "local_promo": promo[tt, jj],
        "cost_index": cost_obs[tt, jj]})
    truth_week = pd.DataFrame({
        "store": jj.astype(np.int32), "week": tt.astype(np.int16), "q_true": q[tt, jj], "xi_t": xi_t[tt, jj],
        "cost_shock": cost[tt, jj], "market_size": M[tt]})

    # category layer (observed, CHF)
    W = category_weights(rng, store_nest)                                     # (J, C)
    S = category_season(s)                                                   # (T, C)
    C = len(CATEGORIES)
    share = W[jj] * S[tt]
    share /= share.sum(1, keepdims=True)
    cat_sales = (q_obs[tt, jj] * price[tt, jj])[:, None] * share * np.exp(rng.normal(0, cfg.cat_noise_sd, share.shape))
    category = pd.DataFrame({
        "store": np.repeat(jj, C).astype(np.int32), "week": np.repeat(tt, C).astype(np.int16),
        "category": pd.Categorical.from_codes(np.tile(np.arange(C), len(jj)), [c[0] for c in CATEGORIES]),
        "sales_chf": cat_sales.ravel().astype(np.float32)})

    weekly = pd.DataFrame({"week": np.arange(T), "date": dates, "market_size": M, "p_abroad": p0,
                           "q_abroad_true": q0, **{k: s[k].to_numpy() for k in s}})
    store = pd.DataFrame({"store": np.arange(J), "store_id": store_id, "banner": np.array(dm.BANNERS)[sb],
                          "canton": store_canton, "xi_perm": xi_perm, "open_week": open_week,
                          "region": region})
    return {"store_week": store_week, "truth_week": truth_week, "category": category, "weekly": weekly,
            "store": store, "config": asdict(cfg)}
