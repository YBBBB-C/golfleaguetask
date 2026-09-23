"""
One-call access to everything notebooks 03-05 need, with a disk cache.

    from src.pipeline import build
    P = build()                     # dict of DataFrames / objects
    P["ratings"]                    # one row per player
"""
from __future__ import annotations

import pickle
from pathlib import Path

from .data_prep import load_clean
from .profiles import style_profile
from .ratings import rate_all
from .strokes_gained import Baseline, add_strokes_gained, round_table

HALF_LIFE_DAYS = 365


def build(data_path="../data/shot_data.csv", cache_dir="../data/processed", refresh=False) -> dict:
    cache = Path(cache_dir) / "pipeline.pkl"
    if cache.exists() and not refresh:
        with open(cache, "rb") as f:
            return pickle.load(f)

    shots, course_holes = load_clean(data_path)
    fit_rows = shots[~shots.low_quality_event & ~shots.score_only_hole
                     & shots.lie_before.notna() & ~shots.is_penalty]
    baseline = Baseline.fit(fit_rows)
    shots = add_strokes_gained(shots, baseline)
    rounds = round_table(shots)
    ratings, fits = rate_all(rounds, detail_filter=~rounds.low_quality, half_life_days=HALF_LIFE_DAYS)
    ratings = ratings.join(rounds.groupby("player_id").seed.first()).join(style_profile(shots))

    out = dict(shots=shots, course_holes=course_holes, baseline=baseline,
               rounds=rounds, ratings=ratings, fits=fits)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    ratings.to_csv(cache.parent / "player_ratings.csv")
    return out
