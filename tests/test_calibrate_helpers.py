"""Helper: give every banner at least one reachable store, so all FOCs are defined."""
import jax.numpy as jnp
import numpy as np

from mxc.structural import demand as dm


def full_market(a):
    banners = [dm.BANNERS[int(i)] for i in np.asarray(a["store_banner"])]
    extra = [b for b in dm.BANNERS if b not in banners]
    if not extra:
        return a
    a = dict(a)
    J0 = len(banners)
    L = int(a["hh"].shape[0])
    nest = np.array([dm.NESTS.index(dm.BANNER_NEST[b]) for b in banners + extra])
    a["store_banner"] = jnp.array([dm.BANNERS.index(b) for b in banners + extra])
    new_l = np.repeat(np.arange(L), len(extra))
    new_s = np.tile(np.arange(J0, J0 + len(extra)), L)
    a["pair_loc"] = jnp.concatenate([a["pair_loc"], jnp.array(new_l)])
    a["pair_store"] = jnp.concatenate([a["pair_store"], jnp.array(new_s)])
    a["pair_time"] = jnp.concatenate([a["pair_time"], jnp.full(len(new_l), 8.0)])
    a["pair_seg"] = a["pair_loc"] * len(dm.NESTS) + jnp.array(nest)[a["pair_store"]]
    return a
