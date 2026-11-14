"""Population (FSO STATPOP hectare grid) and income (ESTV direct federal tax, by municipality).

Data labels: population, households, taxpayer income distribution [real].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ESTV net-income classes (CHF), as in sheet 511. Upper bound of the top class is open.
INCOME_CLASSES = ["0", "1-30k", "30-40k", "40-50k", "50-75k", "75-100k",
                  "100-200k", "200-500k", "500k-1m", "1m+"]
CLASS_BOUNDS = [(0, 0), (1, 30_000), (30_001, 40_000), (40_001, 50_000), (50_001, 75_000),
                (75_001, 100_000), (100_001, 200_000), (200_001, 500_000),
                (500_001, 1_000_000), (1_000_001, np.inf)]


def read_statpop_hectares(csv: Path) -> pd.DataFrame:
    """Populated hectares: centre coordinates (LV95), population, private households."""
    df = pd.read_csv(csv, sep=";", usecols=["E_KOORD", "N_KOORD", "BBTOT", "HPTOT"])
    # Coordinates are the south-west corner of the 100 m cell.
    return pd.DataFrame({"x": df.E_KOORD + 50.0, "y": df.N_KOORD + 50.0,
                         "pop": df.BBTOT, "households": df.HPTOT})


def _estv_sheet(path: Path, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=3)
    raw = raw.iloc[:, :15]
    raw.columns = ["canton_id", "canton_name", "bfs", "municipality"] + INCOME_CLASSES + ["total"]
    raw = raw[pd.to_numeric(raw["bfs"], errors="coerce").notna()].copy()
    raw["bfs"] = raw["bfs"].astype(int)
    raw = raw[raw["bfs"] < 10000]          # 10000 = canton / national "Total" rows
    for c in INCOME_CLASSES + ["total"]:
        raw[c] = pd.to_numeric(raw[c].replace("-", 0), errors="coerce").fillna(0)  # '-' = suppressed / zero
    return raw.set_index("bfs")


def read_estv_income(xlsx: Path) -> pd.DataFrame:
    """Per municipality: taxpayers by net-income class, mean net income per taxpayer (CHF).

    mean_net_income = total net income (sheet 522, total) / taxpayers (sheet 521, total)
    A taxpayer is a tax unit (a married couple files jointly), close to a household.
    """
    counts = _estv_sheet(xlsx, "511")
    net = _estv_sheet(xlsx, "522")
    n_tax = _estv_sheet(xlsx, "521")["total"]
    out = counts[["municipality"] + INCOME_CLASSES].rename(columns={c: f"n_{c}" for c in INCOME_CLASSES})
    out["taxpayers"] = n_tax
    out["mean_net_income"] = net["total"] / n_tax
    # median from the grouped distribution (linear interpolation within the class)
    out["median_net_income"] = [_grouped_median(r) for r in counts[INCOME_CLASSES].to_numpy()]
    return out


def _grouped_median(counts: np.ndarray) -> float:
    cum = np.cumsum(counts)
    if cum[-1] == 0:
        return np.nan
    half = cum[-1] / 2
    k = int(np.searchsorted(cum, half))
    lo, hi = CLASS_BOUNDS[k]
    if not np.isfinite(hi):
        return float(lo)
    prev = cum[k - 1] if k > 0 else 0
    return lo + (half - prev) / max(counts[k], 1) * (hi - lo)
