"""Phase 2 step 8: the outcome table (DESIGN §7), rows A-D and G, plus a divestiture remedy curve.

Rows: A status quo · B merger, no remedy · C = B + 5% cost saving · D = C + divestiture packages · G prohibit.
E/F (data ring-fence) need Pillar 3 and are left as placeholders.

Divested stores keep their location and nest but get a new owner and banner:
  "Coop (divested)":   Coop supermarkets ranked by store GUPPI (step 1.5), δ = δ_Coop − 0.2 (weaker brand) [assumed]
  "Denner": the whole banner (all stores and the brand) changes owner; δ unchanged
The buyer is either profit-maximising (λμ = 0) or a cooperative (λμ = merged firm's 0.357, e.g. fenaco).

Run:  conda activate keystone && python scripts/17_outcomes.py      (needs 10-14; ~6 min)
Outputs (assets/p1/m8/): 1_8_outcome_table.csv, 1_8_remedy_curve.csv, 1_8_divestiture_list.csv,
  1_8_divestiture_map.png, 1_8_remedy_gain_map.png, 1_8_remedy_curve.png
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyproj  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from mxc import viz  # noqa: E402
from mxc.structural import demand as dm  # noqa: E402
from mxc.structural import screening as sc  # noqa: E402
from mxc.structural import simulate as sim  # noqa: E402
from mxc.structural import supply as su  # noqa: E402

PROCESSED, INTERIM = ROOT / "data/processed", ROOT / "data/interim"
OUT = ROOT / "assets/p1/m8"
P1 = ROOT / "assets/p1"
EFF = 0.05
BRAND_LOSS = 0.2
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


def load():
    z = np.load(PROCESSED / "market.npz", allow_pickle=True)
    mkt = dm.Market(**{k: z[k] for k in z.files})
    par = json.load(open(P1 / "m2/1_2_params.json"))
    sup = json.load(open(P1 / "m4/1_4_supply_params.json"))
    theta = dm.prior_params()._replace(
        delta=jnp.array([par["delta"][b] for b in dm.BANNERS]), alpha_ref=par["alpha_ref"], eta=par["eta"],
        gamma=par["gamma"], kappa=par["kappa"], sigma=par["sigma"],
        p=jnp.array([par["price_index"][b] for b in dm.BANNERS]), p_abroad=par["p_abroad"])
    lm = np.array([sup["lambda_mu"][b] for b in dm.BANNERS])
    cost = np.array([sup["marginal_cost"][b] for b in dm.BANNERS])
    return mkt, theta, lm, cost


class Engine:
    def __init__(self):
        self.mkt, self.theta, self.lm, self.cost = load()
        self.a = dm.to_arrays(self.mkt)
        self.p0 = np.asarray(self.theta.p)
        self.n_lr = np.asarray(self.a["hh"])[:, None] * np.asarray(self.a["type_w"])
        _, _, self.ls0 = dm.choice_probs(self.theta, self.a)
        self.alpha = np.asarray(dm.alpha_types(self.theta, self.a["type_y"]))
        ja = np.load(PROCESSED / "jacobian_abroad.npz", allow_pickle=True)
        self.q_store = ja["q_store"]
        self.Jac = sp.load_npz(PROCESSED / "jacobian.npz").tocsr()
        self.m_banner = self.p0 - self.cost
        self.lam0 = float(self.lm[dm.BANNERS.index("Migros")])
        self.Q0 = np.bincount(self.mkt.store_banner, weights=self.q_store, minlength=len(dm.BANNERS))
        scr = pd.read_csv(P1 / "m5/1_5_store_screen.csv")
        coop = scr[scr.banner == "Coop"].sort_values("guppi", ascending=False)
        self.coop_rank = coop.store_id.to_numpy()                                  # highest GUPPI first
        self.sid_pos = pd.Series(np.arange(self.mkt.J), index=self.mkt.store_id)
        self.border = self.mkt.border_min

    def run(self, merged=True, eff=0.0, coop_share=0.0, denner=False, buyer_lm=0.0):
        banners, nest = list(dm.BANNERS), dict(dm.BANNER_NEST)
        owner = dict(su.OWNER_POST if merged else su.OWNER_PRE)
        sb = self.mkt.store_banner.copy()
        delta, price = list(np.asarray(self.theta.delta)), list(self.p0)
        divested = np.zeros(self.mkt.J, bool)
        if denner:                                       # the whole banner changes owner; brand sold with the stores
            owner["Denner"] = "Buyer B"
            divested[self.mkt.store_banner == dm.BANNERS.index("Denner")] = True
        own_arr = np.array([owner[b] for b in banners])
        mk = own_arr == "Migros-Coop"
        cost = list(np.where(mk, self.cost * (1 - eff), self.cost))
        coop_firm = np.isin(own_arr, ["Migros-Coop", "Migros group", "Coop group"])
        lm = list(np.where(coop_firm, self.lam0, np.where(own_arr == "Buyer B", buyer_lm, 0.0)))
        n_coop = int(round(coop_share * len(self.coop_rank)))
        if n_coop:
            idx = self.sid_pos.loc[self.coop_rank[:n_coop]].to_numpy()
            parent = dm.BANNERS.index("Coop")
            banners.append("Coop (divested)"), nest.update({"Coop (divested)": "full-service"})
            owner["Coop (divested)"] = "Buyer A"
            sb[idx] = len(banners) - 1
            divested[idx] = True
            delta.append(delta[parent] - BRAND_LOSS), price.append(self.p0[parent]), cost.append(self.cost[parent])
            lm.append(buyer_lm)
        a2 = dict(self.a, store_banner=jnp.asarray(sb))
        th = self.theta._replace(delta=jnp.asarray(delta), p=jnp.asarray(price))
        bd = sim.make_banner_demand(a2, banners, nest)
        cost, lm = np.asarray(cost), np.asarray(lm)
        p1, hist = sim.solve_equilibrium(th, bd, cost, lm, owner, banners=banners)
        th1 = th._replace(p=jnp.asarray(p1))
        Q1, *_ = bd(th1)
        Q1 = np.asarray(Q1)
        _, _, ls1 = dm.choice_probs(th1, a2)
        dcs = np.asarray(ls1 - self.ls0) / self.alpha[None, :] * sim.WEEKS          # (L,R) CHF / hh / yr
        loc_cs = (self.n_lr * dcs).sum(1) / self.n_lr.sum(1)
        hh = self.n_lr.sum(1)

        # profits by owner (CHF bn / yr), pre-merger reference uses today's prices and owners
        prof1 = pd.Series((p1 - cost) * Q1 * sim.WEEKS, index=[owner[b] for b in banners]).groupby(level=0).sum()
        prof0 = pd.Series((self.p0 - self.cost) * self.Q0 * sim.WEEKS, index=[su.OWNER_POST[b] for b in dm.BANNERS]).groupby(level=0).sum()
        prof1_merged = prof1.get("Migros-Coop", 0.0)
        # average price paid, Laspeyres with today's store quantities
        p_store0 = self.p0[self.mkt.store_banner]
        p_store1 = p1[sb]
        lasp = (p_store1 * self.q_store).sum() / (p_store0 * self.q_store).sum() - 1
        # residual overlap: store GUPPI between retained Migros-group and retained Coop-group stores
        party = np.array([su.OWNER_PRE[dm.BANNERS[b]] for b in self.mkt.store_banner], dtype=object)
        party[divested] = "Buyer"
        g, _ = sc.store_guppi(self.Jac, party, self.m_banner[self.mkt.store_banner], p_store0)
        fmt = np.array([dm.BANNER_NEST[dm.BANNERS[b]] for b in self.mkt.store_banner])
        thr = np.where(fmt == "convenience", 0.0325, 0.0275)
        retained = np.isfinite(g)
        tot_cs = (self.n_lr * dcs).sum()
        return {
            "p": dict(zip(banners, p1)), "iterations": len(hist) - 1,
            "dp_Migros_pct": (p1[0] / self.p0[0] - 1) * 100,
            "dp_Coop_pct": (p1[dm.BANNERS.index("Coop")] / self.p0[dm.BANNERS.index("Coop")] - 1) * 100,
            "dp_Aldi_pct": (p1[dm.BANNERS.index("Aldi")] / self.p0[dm.BANNERS.index("Aldi")] - 1) * 100,
            "avg_price_paid_pct": lasp * 100,
            "dcs_chf_hh_year": tot_cs / hh.sum(),
            "dcs_near_border_lt15": np.average(loc_cs[self.border < 15], weights=hh[self.border < 15]),
            "dcs_far_from_border_gt30": np.average(loc_cs[self.border > 30], weights=hh[self.border > 30]),
            "dcs_chf_bn_year": tot_cs / 1e9,
            "dprofit_merged_chf_bn_year": (prof1_merged - prof0.get("Migros-Coop", 0.0)) / 1e9,
            "dprofit_all_chf_bn_year": (prof1.sum() - prof0.sum()) / 1e9,
            "net_welfare_chf_bn_year": (tot_cs + prof1.sum() - prof0.sum()) / 1e9,
            "stores_divested": int(divested.sum()),
            "sales_divested_share": float((self.q_store * p_store0)[divested].sum() / (self.q_store * p_store0)[
                np.isin(self.mkt.store_group, ["Migros", "Coop"])].sum()),
            "retained_stores_above_threshold": int((g[retained] > thr[retained]).sum()),
            "retained_stores": int(retained.sum()),
            "median_retained_guppi": float(np.median(g[retained])) if retained.any() else np.nan,
            "_loc_cs": loc_cs, "_divested": divested,
        }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    E = Engine()
    rows, keep = [], {}
    spec = [
        ("B", "Merger, no remedy", dict(eff=0.0)),
        ("C", "B + 5% cost saving", dict(eff=EFF)),
        ("D1", "C + sell Denner (profit-max buyer)", dict(eff=EFF, denner=True)),
        ("D2", "C + sell top 25% Coop supermarkets by GUPPI (profit-max buyer)", dict(eff=EFF, coop_share=0.25)),
        ("D3", "C + sell top 25% Coop supermarkets (cooperative buyer)", dict(eff=EFF, coop_share=0.25, buyer_lm=E.lam0)),
        ("D4", "C + sell top 50% Coop supermarkets (profit-max buyer)", dict(eff=EFF, coop_share=0.50)),
        ("D5", "C + sell top 50% Coop supermarkets + Denner (profit-max buyers)", dict(eff=EFF, coop_share=0.50, denner=True)),
        ("D6", "C + sell top 50% Coop supermarkets + Denner (cooperative buyers)",
         dict(eff=EFF, coop_share=0.50, denner=True, buyer_lm=E.lam0)),
    ]
    for code, label, kw in spec:
        r = E.run(**kw)
        keep[code] = r
        rows.append({"row": code, "scenario": label, **{k: v for k, v in r.items() if not k.startswith("_") and k != "p"}})
        print(f"{code}: Migros {r['dp_Migros_pct']:+.1f}%, Coop {r['dp_Coop_pct']:+.1f}%, ΔCS {r['dcs_chf_hh_year']:,.0f}, "
              f"divested {r['stores_divested']}, retained flagged {r['retained_stores_above_threshold']}/{r['retained_stores']}")
    tab = pd.DataFrame(rows)
    zero = {c: 0.0 for c in tab.columns if c not in ("row", "scenario")}
    tab = pd.concat([pd.DataFrame([{"row": "A", "scenario": "Status quo", **zero}]), tab,
                     pd.DataFrame([{"row": "E", "scenario": "C + data ring-fence (Pillar 3, pending)"},
                                   {"row": "F", "scenario": "D + E (Pillar 3, pending)"},
                                   {"row": "G", "scenario": "Prohibit", **zero}])], ignore_index=True)
    tab.round(4).to_csv(OUT / "1_8_outcome_table.csv", index=False)

    # remedy curve: share of Coop supermarkets sold × buyer type (with 5% saving)
    curve = []
    for share in [0.0, 0.1, 0.25, 0.5, 0.75]:
        for buyer, blm in [("profit-max", 0.0), ("cooperative", E.lam0)]:
            if share == 0.0 and buyer == "cooperative":
                continue
            r = keep["D2"] if (share, buyer) == (0.25, "profit-max") else keep["D3"] if (share, buyer) == (0.25, "cooperative") \
                else keep["D4"] if (share, buyer) == (0.5, "profit-max") else keep["C"] if share == 0.0 \
                else E.run(eff=EFF, coop_share=share, buyer_lm=blm)
            curve.append({"coop_share_sold": share, "buyer": buyer, "stores_sold": r["stores_divested"],
                          "dcs_chf_hh_year": r["dcs_chf_hh_year"], "dp_Migros_pct": r["dp_Migros_pct"],
                          "dp_Coop_pct": r["dp_Coop_pct"], "retained_flagged": r["retained_stores_above_threshold"]})
            print(f"curve {share:.2f} {buyer}: ΔCS {r['dcs_chf_hh_year']:,.0f}")
    cv = pd.DataFrame(curve)
    cv.round(3).to_csv(OUT / "1_8_remedy_curve.csv", index=False)

    st = pd.read_parquet(PROCESSED / "stores.parquet", columns=["store_id", "name", "lon", "lat", "canton", "municipality"])
    scr = pd.read_csv(P1 / "m5/1_5_store_screen.csv")[["store_id", "guppi"]]
    rank = pd.Series(np.arange(len(E.coop_rank)), index=E.coop_rank)
    dl = st[st.store_id.isin(E.coop_rank)].merge(scr, on="store_id")
    dl["rank"] = dl.store_id.map(rank)
    dl["package"] = np.where(dl["rank"] < 0.25 * len(rank), "top 25%", np.where(dl["rank"] < 0.5 * len(rank), "25-50%", "retained"))
    dl.sort_values("rank").round(4).to_csv(OUT / "1_8_divestiture_list.csv", index=False)

    figures(E, tab, cv, dl, keep)


def figures(E, tab, cv, dl, keep):
    viz.setup()
    adm = gpd.read_file(INTERIM / "admin_boundaries.gpkg")
    cantons = adm[adm.admin_level.astype(str) == "4"].to_crs(2056)
    tr = pyproj.Transformer.from_crs(4326, 2056, always_xy=True)

    # 1. divestiture map
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    viz.map_axes(ax, cantons)
    for pkg, colr, s in [("retained", viz.MUTED, 3), ("25-50%", "#f3a27e", 7), ("top 25%", viz.ORANGE, 9)]:
        d = dl[dl.package == pkg]
        x, y = tr.transform(d.lon.to_numpy(), d.lat.to_numpy())
        ax.scatter(x, y, s=s, color=colr, linewidths=0, zorder=3 if pkg != "retained" else 2,
                   label={"retained": "Coop supermarket kept", "25-50%": "sold in D4 (GUPPI rank 25-50%)",
                          "top 25%": "sold in D2/D3 and D4 (top 25% GUPPI)"}[pkg] + f"  ({len(d)})")
    ax.legend(loc="lower left", markerscale=2, fontsize=8.5)
    ax.set_title("The divestiture list: Coop supermarkets ranked by how much they compete with Migros")
    viz.source(ax, "Ranking by store GUPPI (step 1.5). Denner (sold whole in D1/D5) not shown. Model [calibrated].")
    viz.save(fig, 1, "1_8", "divestiture_map")

    # 2. gain map: CHF per household saved by D5 vs C
    loc = pd.read_parquet(PROCESSED / "locations.parquet", columns=["loc_id", "x", "y"]).set_index("loc_id").loc[E.mkt.loc_id]
    gain = keep["D5"]["_loc_cs"] - keep["C"]["_loc_cs"]
    bins = [0] + list(np.round(np.quantile(gain, [0.2, 0.4, 0.6, 0.8]), -1)) + [gain.max() + 1]
    bins = sorted(set(bins))
    cmap, norm = ListedColormap(BLUE_RAMP[: len(bins) - 1]), BoundaryNorm(bins, len(bins) - 1)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    viz.map_axes(ax, cantons)
    o = np.argsort(gain)
    ax.scatter(loc.x.to_numpy()[o], loc.y.to_numpy()[o], c=gain[o], cmap=cmap, norm=norm, s=1.2, linewidths=0, zorder=2)
    left = keep["D5"]["dcs_chf_hh_year"]
    ax.set_title(f"Selling half of Coop's supermarkets + Denner saves CHF {np.average(gain, weights=E.mkt.hh):,.0f}"
                 f" per household; CHF {-left:,.0f} of harm remains")
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal",
                       fraction=0.04, pad=0.02, aspect=40, ticks=bins[1:-1])
    cb_.ax.set_xticklabels([f"{b:,.0f}" for b in bins[1:-1]], fontsize=8)
    cb_.set_label("CHF per household per year saved by remedy D5 relative to row C (no divestiture)", fontsize=8.5,
                  color=viz.INK_2)
    cb_.outline.set_visible(False)
    viz.source(ax, "D5 = merger + 5% cost saving + sale of the top-50% GUPPI Coop supermarkets and all Denner stores to "
                   "profit-maximising buyers. Model [calibrated].")
    viz.save(fig, 1, "1_8", "remedy_gain_map")

    # 3. remedy curve
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for buyer, colr in [("profit-max", viz.BLUE), ("cooperative", viz.AQUA)]:
        g = cv[(cv.buyer == buyer) | (cv.coop_share_sold == 0)].sort_values("coop_share_sold")
        ax.plot(g.coop_share_sold * 100, -g.dcs_chf_hh_year, marker="o", color=colr, label=f"{buyer} buyer")
    b = tab.set_index("row").loc["B", "dcs_chf_hh_year"]
    ax.axhline(-b, color=viz.INK_2, linestyle=":", linewidth=1)
    ax.text(1, -b + 8, "B: merger, no remedy, no saving", fontsize=8.5, color=viz.INK_2)
    ax.set_ylim(0, -b * 1.12)
    ax.set_xlabel("share of Coop supermarkets sold (highest GUPPI first), %")
    ax.set_ylabel("consumer loss, CHF per household per year")
    ax.legend(loc="lower left")
    ax.set_title("Even selling three quarters of Coop's supermarkets leaves a loss")
    viz.source(ax, "All points include the 5% cost saving (row C). Buyer brand 0.2 utility weaker than Coop [assumed].")
    viz.save(fig, 1, "1_8", "remedy_curve")


if __name__ == "__main__":
    main()
