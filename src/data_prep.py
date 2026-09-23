"""
Data loading, cleaning and feature derivation for shot_data.csv.

Every function here is pure (DataFrame in, DataFrame out) so the notebooks can
show each step and the later models can call `load_clean()` in one line.

Key conventions
---------------
* One row = one stroke. Penalty strokes are their own row (lie_before is NaN).
* Hole score = max(shot_no) for the player-hole (robust to the few missing rows).
* Distances are stored in inches; we add yard columns (1 yd = 36 in).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INCH_PER_YD = 36.0
HOLE_KEY = ["event_id", "player_id", "round_no", "course_no", "hole_no"]
ROUND_KEY = ["event_id", "player_id", "round_no", "course_no"]
COURSE_HOLE_KEY = ["event_id", "course_no", "hole_no"]

# --------------------------------------------------------------------------
# Lie decoding. Codes are anonymised; the mapping below is inferred from
# behaviour (shot order, distance, next-shot penalties, strokes-to-hole-out).
# Notebook 01 shows the evidence for every line.
# --------------------------------------------------------------------------
LIE_MAP = {
    "LIE_16": ("Tee", "tee"),
    "LIE_4": ("Fairway", "fairway"),
    "LIE_15": ("Rough", "rough"),
    "LIE_10": ("Rough (light / first cut)", "rough"),
    "LIE_8": ("Rough (around green)", "rough"),
    "LIE_5": ("Fairway bunker", "sand"),
    "LIE_11": ("Bunker", "sand"),
    "LIE_7": ("Fringe", "fringe"),
    "LIE_6": ("Green", "green"),
    "LIE_9": ("Holed", "holed"),
    "LIE_1": ("Trees / bushes (unplayable often declared)", "recovery"),
    "LIE_2": ("Recovery", "recovery"),
    "LIE_3": ("Recovery", "recovery"),
    "LIE_12": ("Free-relief area (e.g. cart path)", "recovery"),
    "LIE_13": ("Recovery", "recovery"),
    "LIE_17": ("Recovery", "recovery"),
    "LIE_14": ("Out of bounds / lost ball", "hazard"),
    "LIE_18": ("Out of bounds / lost ball", "hazard"),
    "LIE_19": ("Water (penalty area)", "hazard"),
}
LIE_GROUP = {k: v[1] for k, v in LIE_MAP.items()}
LIE_LABEL = {k: v[0] for k, v in LIE_MAP.items()}

# Short game boundary, as defined in the brief (~100 yards).
ARG_MAX_YD = 100.0


def load_raw(path: str | Path = "../data/shot_data.csv") -> pd.DataFrame:
    """Read the CSV and parse dates. No rows are dropped here."""
    df = pd.read_csv(path, parse_dates=["event_start_date", "event_end_date"])
    return df.sort_values(HOLE_KEY + ["shot_no"]).reset_index(drop=True)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Yards, lie groups, penalty flag, hole score and strokes remaining."""
    df = df.copy()
    df["yd_before"] = df["distance_before_inch"] / INCH_PER_YD
    df["yd_after"] = df["distance_after_inch"] / INCH_PER_YD
    df["is_penalty"] = df["penalty_type"].notna()
    df["lie_group_before"] = df["lie_before"].map(LIE_GROUP)
    df["lie_group_after"] = df["lie_after"].map(LIE_GROUP)

    g = df.groupby(HOLE_KEY, sort=False)
    df["hole_rows"] = g["shot_no"].transform("size")
    df["hole_score"] = g["shot_no"].transform("max")
    # strokes still to play from this row, this row included (penalties count)
    df["strokes_remaining"] = df["hole_score"] - df["shot_no"] + 1
    return df


def flag_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row and hole level quality flags.

    score_only_hole : the hole has fewer rows than strokes, so shot detail is
                      missing (the single stub row carries the final score).
                      A genuine hole in one (1 row, score 1) is NOT flagged.
    low_quality_event : event where >5% of holes are score-only OR >50% of the
                        `distance` column is missing (rounded / legacy feed).
    """
    df = df.copy()
    df["score_only_hole"] = df["hole_rows"] < df["hole_score"]

    hole = df.drop_duplicates(HOLE_KEY)
    ev = pd.DataFrame(
        {
            "score_only_share": hole.groupby("event_id")["score_only_hole"].mean(),
            "distance_missing_share": df.groupby("event_id")["distance"].apply(
                lambda s: s.isna().mean()
            ),
        }
    )
    ev["low_quality_event"] = (ev.score_only_share > 0.05) | (
        ev.distance_missing_share > 0.5
    )
    df = df.merge(ev[["low_quality_event"]], left_on="event_id", right_index=True)
    return df


def infer_par(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer par for every (event, course, hole). The data has no par column.

    Primary signal: median distance left after the tee shot.
        < 60 yd  -> Par 3 (tee shot is aimed at the green)
        < 205 yd -> Par 4
        else     -> Par 5
    This is robust to the mis-recorded tee yardages found in some events.
    Two overrides reconcile it with the field's scoring average (see code).

    Fallback when no tee-shot detail exists (score-only courses): the field's
    mean score on the hole (<3.5 -> 3, >=4.45 -> 5, otherwise 4).
    """
    tee = df[(df["shot_no"] == 1) & (df["lie_before"] == "LIE_16") & ~df["score_only_hole"]]
    t = tee.groupby(COURSE_HOLE_KEY).agg(
        tee_after_med_yd=("yd_after", "median"),
        hole_yd=("yd_before", "median"),
        n_tee=("yd_after", "size"),
    )
    s = (
        df.drop_duplicates(HOLE_KEY)
        .groupby(COURSE_HOLE_KEY)
        .agg(mean_score=("hole_score", "mean"), n_players=("hole_score", "size"))
    )
    ch = s.join(t, how="left").reset_index()

    by_tee = np.select(
        [ch.tee_after_med_yd < 60, ch.tee_after_med_yd < 205], [3, 4], 5
    )
    by_score = np.select([ch.mean_score < 3.5, ch.mean_score >= 4.45], [3, 5], 4)
    use_tee = ch.n_tee.fillna(0) >= 5
    par = np.where(use_tee, by_tee, by_score).astype(int)
    # Consistency overrides where the two signals disagree:
    #  * tee shots finish near the green but the field averages >= 3.5
    #    -> a short / driveable par 4, not a par 3
    #  * tee shots leave 205-230 yd but the field averages < 4.45
    #    -> a long par 4, not a par 5
    short4 = use_tee & (par == 3) & (ch.mean_score >= 3.5)
    #  * a par 4 by the tee rule that measures >= 520 yd with a field average
    #    >= 4.2 -> a par 5 (tee yardage and tee-shot finish disagree here)
    long4 = (use_tee & (par == 5) & (ch.tee_after_med_yd < 230)
             & (ch.mean_score < 4.45) & (ch.hole_yd < 500))
    long5 = use_tee & (par == 4) & (ch.hole_yd >= 520) & (ch.mean_score >= 4.2)
    par = np.where(short4 | long4, 4, par)
    par = np.where(long5, 5, par)
    ch["par"] = par
    ch["par_source"] = np.where(use_tee, "tee_shot_finish", "field_mean_score")
    return ch


def add_par_and_category(df: pd.DataFrame, course_holes: pd.DataFrame) -> pd.DataFrame:
    """
    Attach par and assign every stroke to a game area:
        OTT  : tee shot on a par 4 / 5
        APP  : any other shot from > 100 yd (incl. par-3 tee shots)
        ARG  : shots from <= 100 yd that are not on the green
        PUTT : shots from the green
    Penalty rows inherit the category of the stroke that caused them.
    """
    df = df.merge(course_holes[COURSE_HOLE_KEY + ["par"]], on=COURSE_HOLE_KEY, how="left")
    cat = np.select(
        [
            (df["lie_before"] == "LIE_16") & (df["par"] >= 4),
            df["lie_before"] == "LIE_6",
            df["yd_before"] > ARG_MAX_YD,
            df["yd_before"] <= ARG_MAX_YD,
        ],
        ["OTT", "PUTT", "APP", "ARG"],
        default="",
    )
    df["category"] = pd.Series(cat, index=df.index).replace("", np.nan)
    df["category"] = df.groupby(HOLE_KEY, sort=False)["category"].ffill()
    df["hole_score_to_par"] = df["hole_score"] - df["par"]
    return df


def load_clean(path: str | Path = "../data/shot_data.csv"):
    """
    Full pipeline. Returns (shots, course_holes).

    shots        : every row of the raw file plus derived columns and flags.
    course_holes : one row per (event, course, hole) with inferred par.
    Nothing is dropped; downstream models filter on the flags.
    """
    df = load_raw(path)
    df = add_basic_features(df)
    df = flag_quality(df)
    ch = infer_par(df)
    df = add_par_and_category(df, ch)
    return df, ch
