"""
Strokes Gained (SG) engine.

1. Baseline   : E[strokes to hole out | lie, distance], fitted from the data.
2. SG/stroke  : SG = E(before) - E(after) - 1, with penalties folded into the
                stroke that caused them. Per hole, SG sums exactly to
                E(tee) - score (the "telescoping" property).
3. Round table: one row per player-round with field-relative SG, total and by
                game area (OTT / APP / ARG / PUTT).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .data_prep import HOLE_KEY, ROUND_KEY

CATEGORIES = ["OTT", "APP", "ARG", "PUTT"]

# Codes with enough strokes get their own curve; the rest are pooled.
OWN_CURVE = ["LIE_16", "LIE_4", "LIE_15", "LIE_10", "LIE_8", "LIE_11", "LIE_5", "LIE_7", "LIE_6"]
POOLED = "recovery"  # trees, native areas, relief areas, and rare plays from hazards


def baseline_lie(code: pd.Series) -> pd.Series:
    """Map a raw lie code to the curve it uses."""
    return code.where(code.isin(OWN_CURVE), POOLED)


# --------------------------------------------------------------------------
# 1. Baseline
# --------------------------------------------------------------------------
@dataclass
class Baseline:
    """Monotone expected-strokes curves, one per baseline lie, on log distance."""

    grid: dict = field(default_factory=dict)  # lie -> (log_yd array, expected array)

    @staticmethod
    def fit(shots: pd.DataFrame, n_bins: int = 60, min_per_bin: int = 25) -> "Baseline":
        """
        shots must contain lie_before, yd_before, strokes_remaining.
        For each lie: quantile-bin the distance, average strokes remaining,
        then enforce 'further away is never easier' with isotonic regression,
        and smooth lightly (5-bin moving average, which keeps monotonicity).
        """
        b = Baseline()
        d = shots.assign(blie=baseline_lie(shots["lie_before"]))
        d = d[d.yd_before.notna()]
        for lie, g in d.groupby("blie"):
            x = np.log(g.yd_before.clip(lower=0.05).to_numpy())
            y = g.strokes_remaining.to_numpy(dtype=float)
            q = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
            idx = np.clip(np.searchsorted(q, x, side="right") - 1, 0, len(q) - 2)
            cnt = np.bincount(idx, minlength=len(q) - 1)
            keep = cnt >= min_per_bin
            xm = np.bincount(idx, weights=x, minlength=len(q) - 1)[keep] / cnt[keep]
            ym = np.bincount(idx, weights=y, minlength=len(q) - 1)[keep] / cnt[keep]
            iso = IsotonicRegression(increasing=True, y_min=1.0).fit(xm, ym, sample_weight=cnt[keep])
            ys = iso.predict(xm)
            # light moving-average smoothing; averaging a monotone sequence keeps it monotone
            k = 5
            ys = np.convolve(np.pad(ys, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
            b.grid[lie] = (xm, ys)
        return b

    def predict(self, lie_code: pd.Series, yd: pd.Series) -> np.ndarray:
        """Expected strokes; holed balls (LIE_9) are 0."""
        lie = baseline_lie(lie_code.fillna(POOLED))
        out = np.full(len(yd), np.nan)
        logd = np.log(np.clip(yd.to_numpy(dtype=float), 0.05, None))
        for l in np.unique(lie):
            m = (lie == l).to_numpy()
            xs, ys = self.grid[l]
            out[m] = np.interp(logd[m], xs, ys)  # flat beyond the observed range
        out[(lie_code == "LIE_9").to_numpy()] = 0.0
        return out

    def table(self, yards=(1, 3, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500)) -> pd.DataFrame:
        """Readable lookup table: expected strokes by lie (rows) and distance (columns)."""
        rows = {}
        for lie in self.grid:
            code = pd.Series([lie] * len(yards))
            rows[lie] = self.predict(code.replace(POOLED, "LIE_1"), pd.Series(yards, dtype=float))
        return pd.DataFrame(rows, index=list(yards)).T


# --------------------------------------------------------------------------
# 2. Strokes Gained per stroke
# --------------------------------------------------------------------------
def add_strokes_gained(df: pd.DataFrame, base: Baseline) -> pd.DataFrame:
    """
    Adds E_before, E_after and sg to every non-penalty stroke of detail holes.

    E_after of a stroke is taken from the start of the NEXT real stroke, plus
    one for every penalty row in between. This means:
      * a ball into water is charged the penalty and the drop position;
      * free relief (e.g. LIE_12) moves the ball without distorting SG;
      * per hole, sum(sg) == E(tee) - score exactly.
    Penalty rows themselves get sg = 0 (their cost sits on the causing stroke).
    """
    df = df.copy()
    real = df["lie_before"].notna() & ~df["is_penalty"]  # a few penalty rows also carry a lie
    df["E_before"] = np.where(real, base.predict(df["lie_before"], df["yd_before"]), np.nan)

    keys = [df[k] for k in HOLE_KEY]
    cum_pen = df["is_penalty"].astype(int).groupby(keys, sort=False).cumsum()
    e_real = df["E_before"].where(real)
    c_real = cum_pen.where(real).astype(float)
    # start of the next real stroke, and penalties counted up to it
    nxt_E = e_real.groupby(keys, sort=False).shift(-1).groupby(keys, sort=False).bfill()
    nxt_c = c_real.groupby(keys, sort=False).shift(-1).groupby(keys, sort=False).bfill()
    between = nxt_c - cum_pen
    holed = df["lie_after"].eq("LIE_9")
    df["E_after"] = np.where(holed, 0.0, nxt_E + between)
    # last real stroke of a hole that ends with penalties only cannot happen (holes end LIE_9)
    df["sg"] = np.where(real, df["E_before"] - df["E_after"] - 1.0, 0.0)
    df.loc[df["score_only_hole"], ["E_before", "E_after", "sg"]] = np.nan
    return df


# --------------------------------------------------------------------------
# 3. Round table
# --------------------------------------------------------------------------
def round_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per player-round-course.

    sg_total : field-relative strokes gained per 18 holes, from hole SCORES
               (uses every hole, including score-only ones):
               sum over holes of (field mean score on that hole & round - score).
    sg_<cat> : field-relative SG by game area, detail holes only, scaled to 18 holes.
    """
    hole = df.drop_duplicates(HOLE_KEY)[HOLE_KEY + ["event_start_date", "draft_seed", "hole_score",
                                                    "score_only_hole", "low_quality_event"]].copy()
    fld = ["event_id", "round_no", "course_no", "hole_no"]
    hole["hole_sg"] = hole.groupby(fld)["hole_score"].transform("mean") - hole["hole_score"]

    r = hole.groupby(ROUND_KEY).agg(
        date=("event_start_date", "first"),
        seed=("draft_seed", "first"),
        holes=("hole_no", "size"),
        detail_holes=("score_only_hole", lambda s: (~s).sum()),
        score=("hole_score", "sum"),
        sg_total_raw=("hole_sg", "sum"),
        low_quality=("low_quality_event", "first"),
    ).reset_index()
    r["sg_total"] = r.sg_total_raw * 18 / r.holes

    # game areas: sum SG per category over detail holes, then subtract the
    # field average for the same event-round-course (per detail hole played)
    det = df[~df["score_only_hole"] & df["category"].notna()]
    cat = det.pivot_table(index=ROUND_KEY, columns="category", values="sg", aggfunc="sum").reindex(columns=CATEGORIES)
    cat = cat.div(r.set_index(ROUND_KEY)["detail_holes"].reindex(cat.index), axis=0) * 18
    field_mean = cat.groupby(level=["event_id", "round_no", "course_no"]).transform("mean")
    cat = (cat - field_mean).add_prefix("sg_").reset_index()
    r = r.merge(cat, on=ROUND_KEY, how="left")
    full = r["detail_holes"] == r["holes"]
    for c in CATEGORIES:
        r.loc[~full, f"sg_{c}"] = np.nan  # only rounds with complete shot detail
    return r
