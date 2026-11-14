"""Calibrate banner qualities delta to observed sales shares (Phase 2 step 2).

Targets are domestic food-retail sales shares by player (R10 in DATA_REGISTER) plus the share of
grocery spending made abroad (from MCS card data). Banners inside a player move together. Each
banner keeps its own price and geography, so the split within a player comes out of the model.

Contraction (Berry 1994, damped for nested logit; same rule as design_dummy.md Step 3):

    delta_b  <-  delta_b + lam * [ ln(target_G) - ln(model_G) ]     for every banner b of player G

    model_G = R_G / (sum_G' R_G' + R_abroad),   R_G = sum_{j in G} p_b(j) * q_j,   R_abroad = p_abroad * q_0

Abroad is the anchor (delta = 0). Once every domestic share matches, the abroad share matches too,
because shares add up to 1.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import demand as dm

# Player -> banners. "Migros" here means Migros excluding Denner, as in the published 33.5%.
PLAYERS = {
    "Migros": ["Migros", "Migros Partner", "Migrolino"],
    "Denner": ["Denner"],
    "Coop": ["Coop", "Coop Pronto"],
    "Lidl": ["Lidl"],
    "Aldi": ["Aldi"],
    "Others": ["Volg", "Landi", "Spar", "Manor"],
}
# Domestic food-retail sales shares 2025, % (watson.ch 29 Oct 2026; Denner = 42.6 - 33.5; Others = residual).
DOMESTIC_SHARE = {"Migros": 33.5, "Denner": 9.1, "Coop": 33.7, "Lidl": 8.4, "Aldi": 7.3, "Others": 8.0}


def player_index() -> np.ndarray:
    """(B,) index of the player each banner belongs to."""
    lookup = {b: g for g, (_, bs) in enumerate(PLAYERS.items()) for b in bs}
    return np.array([lookup[b] for b in dm.BANNERS])


def targets(abroad_share: float) -> pd.Series:
    """Shares of total grocery spending (domestic + abroad), by player and abroad."""
    dom = pd.Series(DOMESTIC_SHARE, dtype=float)
    dom = dom / dom.sum()
    return pd.concat([dom * (1 - abroad_share), pd.Series({"abroad": abroad_share})])


def model_revenue_shares(theta: dm.Params, a: dict, pidx) -> jnp.ndarray:
    """(G + 1,) revenue shares: players in PLAYERS order, then abroad."""
    q_store, q_abroad = dm.quantities(theta, a)
    rev_store = q_store * theta.p[a["store_banner"]]
    rev_player = jax.ops.segment_sum(rev_store, jnp.asarray(pidx)[a["store_banner"]], num_segments=len(PLAYERS))
    rev = jnp.concatenate([rev_player, jnp.array([q_abroad * theta.p_abroad])])
    return rev / rev.sum()


def calibrate(theta: dm.Params, a: dict, target: pd.Series, tol: float = 1e-10, max_iter: int = 2000):
    """Run the contraction. Returns calibrated theta and the iteration history."""
    pidx = jnp.asarray(player_index())
    ln_t = jnp.log(jnp.asarray(target.to_numpy()[:-1]))
    lam = 1.0 - theta.sigma

    @jax.jit
    def step(delta, a):                                  # market passed as an argument, not baked in as constants
        s = model_revenue_shares(theta._replace(delta=delta), a, pidx)
        gap = ln_t - jnp.log(s[:-1])                     # (G,) log gap per player
        return delta + lam * gap[pidx], gap, s

    delta, hist = theta.delta, []
    for it in range(max_iter):
        new, gap, s = step(delta, a)
        hist.append({"iter": it, "max_abs_log_gap": float(jnp.abs(gap).max()),
                     **{g: float(s[i]) for i, g in enumerate(list(PLAYERS) + ["abroad"])}})
        if hist[-1]["max_abs_log_gap"] < tol:
            break
        delta = new
    return theta._replace(delta=delta), pd.DataFrame(hist)


def abroad_share_from_cards(debit_csv) -> pd.DataFrame:
    """Share of Swiss debit-card grocery spending made in DE, FR, IT, AT, by year (MCS / SIX).

    Category 'Retail: Food, beverage, tobacco', point-of-sale payments. Liechtenstein (same customs
    area) and the rest of the world (holidays) are excluded. All values share one scaling divisor
    (MCS README), so the ratios are valid.
    """
    d = pd.read_csv(debit_csv, usecols=["Date", "Type", "Merchant category", "Country", "Scaled Value"])
    d = d[(d.Type == "POS") & (d["Merchant category"] == "Retail: Food, beverage, tobacco")]
    d["year"] = d.Date.str[:4].astype(int)
    d = d[d.year <= 2022]                                # the file ends 31 Dec 2022 (2023 = 1 stray day)
    v = d.pivot_table(index="year", columns="Country", values="Scaled Value", aggfunc="sum")
    v["neighbours"] = v[["DEU", "FRA", "ITA", "AUT"]].sum(1)
    v["abroad_share"] = v.neighbours / (v.neighbours + v.CHE)
    return v[["CHE", "DEU", "FRA", "ITA", "AUT", "neighbours", "abroad_share"]]
