"""Rolling time-split validation for rating models."""
import numpy as np
import pandas as pd

from .ratings import fit_ratings

CUTOFFS = ["2022-07-01", "2022-10-01", "2023-05-01", "2023-08-01"]


def rolling_splits(rounds: pd.DataFrame, cutoffs=CUTOFFS, horizon_days: int = 120):
    """Train on everything before the cutoff, test on the next `horizon_days`."""
    for c in cutoffs:
        c = pd.Timestamp(c)
        tr = rounds[rounds.date < c]
        te = rounds[(rounds.date >= c) & (rounds.date < c + pd.Timedelta(days=horizon_days))]
        yield c, tr, te


def score_prediction(pred: pd.Series, test: pd.DataFrame, metric: str) -> dict:
    """Per-round squared error of predicting each test round by the player's rating."""
    tt = test.dropna(subset=[metric]).join(pred.rename("p"), on="player_id")
    tt["p"] = tt["p"].fillna(0.0)  # unseen players are predicted as average
    mse = np.mean((tt[metric] - tt.p) ** 2)
    mse0 = np.mean(tt[metric] ** 2)
    return {"mse": mse, "gain": mse0 - mse, "n": len(tt)}


def evaluate_grid(rounds, metric, grid: list[dict], cutoffs=CUTOFFS, extra: dict | None = None):
    """
    grid  : list of kwargs for fit_ratings (each gets a 'name').
    extra : {name: function(train)->pd.Series} for non-model baselines.
    Returns mean over folds of the variance explained ('gain', strokes^2 per round).
    """
    rows = []
    for c, tr, te in rolling_splits(rounds, cutoffs):
        for kw in grid:
            kw = dict(kw); name = kw.pop("name")
            f = fit_ratings(tr, metric, ref_date=c, **kw)
            rows.append({"cutoff": c, "model": name, **score_prediction(f.skills.skill, te, metric)})
        for name, fn in (extra or {}).items():
            rows.append({"cutoff": c, "model": name, **score_prediction(fn(tr), te, metric)})
    res = pd.DataFrame(rows)
    out = res.groupby("model").apply(lambda g: pd.Series({
        "gain_per_round": np.average(g.gain, weights=g.n),
        "rmse": np.sqrt(np.average(g.mse, weights=g.n)),
    }), include_groups=False)
    return out.sort_values("gain_per_round", ascending=False), res
