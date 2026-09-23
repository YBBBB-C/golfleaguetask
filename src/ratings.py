"""
Player ratings from round-level Strokes Gained.

Model (one per metric: total, OTT, APP, ARG, PUTT)
--------------------------------------------------
    y[p, r] = skill[p] + field_adj[r] + noise,     noise ~ N(0, sigma_e^2 / w)

* y         : a player's field-relative SG in round r (per 18 holes).
* skill[p]  : what we want. Strokes per round better than an average player.
* field_adj : one term per (event, round, course). The data is already
              relative to the field average, so this term absorbs FIELD
              STRENGTH: in a strong field, beating the average is worth more.
* w         : time-decay weight, 0.5 ** (age_in_days / half_life_days).

Skills get a Normal(0, sigma_s^2) prior, which is ridge shrinkage with
lambda = sigma_e^2 / sigma_s^2. Both variances are estimated from the data
(empirical Bayes), so the shrinkage strength is not hand-tuned.

Readable consequence: a player with effective sample n_eff keeps
    n_eff / (n_eff + lambda)
of her observed edge. Few rounds -> pulled harder toward average.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ROUND_ID = ["event_id", "round_no", "course_no"]


def time_weights(dates: pd.Series, ref_date, half_life_days: float | None) -> np.ndarray:
    if half_life_days is None:
        return np.ones(len(dates))
    age = (pd.Timestamp(ref_date) - pd.to_datetime(dates)).dt.days.to_numpy()
    return 0.5 ** (age / half_life_days)


@dataclass
class RatingFit:
    metric: str
    skills: pd.DataFrame     # per player: skill, sd, n_rounds, n_eff, shrink, raw_mean
    field_adj: pd.Series     # per round id
    sigma_e: float           # round-to-round noise (strokes per round)
    sigma_s: float           # spread of true skill across players
    lam: float               # sigma_e^2 / sigma_s^2, in "effective rounds"


def fit_ratings(rounds: pd.DataFrame, metric: str, ref_date=None, half_life_days: float | None = 365,
                n_iter: int = 500, field_ridge: float = 1e-3, use_field: bool = True,
                shrink: bool = True, lam_fixed: float | None = None) -> RatingFit:
    """
    lam_fixed : use this shrinkage strength (in effective rounds) instead of
                the empirical-Bayes estimate. Chosen by time-split validation.
    use_field=False drops the field-strength term (ablation).
    shrink=False sets lambda ~ 0, i.e. plain (weighted) averages (ablation).
    """
    d = rounds.dropna(subset=[metric]).copy()
    ref_date = ref_date or d["date"].max()
    w = time_weights(d["date"], ref_date, half_life_days)
    y = d[metric].to_numpy(float)

    p_codes, players = pd.factorize(d["player_id"])
    r_codes, rids = pd.factorize(pd.MultiIndex.from_frame(d[ROUND_ID]))
    P, R = len(players), len(rids)

    # weighted normal equations for [skill | field_adj]
    A = np.zeros((P + R, P + R))
    np.add.at(A, (p_codes, p_codes), w)
    np.add.at(A, (P + r_codes, P + r_codes), w)
    np.add.at(A, (p_codes, P + r_codes), w)
    np.add.at(A, (P + r_codes, p_codes), w)
    b = np.zeros(P + R)
    np.add.at(b, p_codes, w * y)
    np.add.at(b, P + r_codes, w * y)

    n_eff = np.bincount(p_codes, weights=w, minlength=P)
    # method-of-moments start: spread of player means minus their sampling noise
    pm = np.bincount(p_codes, weights=w * y, minlength=P) / n_eff
    sigma_e2 = np.average((y - pm[p_codes]) ** 2, weights=w)
    sigma_s2 = max(np.var(pm) - np.mean(sigma_e2 / n_eff), 1e-3)
    for _ in range(n_iter):
        prev = sigma_s2
        lam = lam_fixed if lam_fixed is not None else (sigma_e2 / sigma_s2 if shrink else 1e-6)
        reg = np.r_[np.full(P, lam), np.full(R, field_ridge if use_field else 1e9)]
        Ainv = np.linalg.inv(A + np.diag(reg))
        beta = Ainv @ b
        skill, fadj = beta[:P], beta[P:]
        resid = y - skill[p_codes] - fadj[r_codes]
        post_var = sigma_e2 * np.diag(Ainv)[:P]
        # EM updates for the two variance components
        sigma_s2 = np.mean(skill ** 2 + post_var)
        dof = len(y) - np.trace(A @ Ainv)  # effective residual degrees of freedom
        sigma_e2 = np.sum(w * resid ** 2) / max(dof, 1) * len(y) / w.sum()
        if not shrink or lam_fixed is not None or abs(sigma_s2 - prev) < 1e-7:
            break

    raw = pd.Series(np.bincount(p_codes, weights=w * y, minlength=P) / n_eff, index=players)
    skills = pd.DataFrame({
        "skill": skill,
        "sd": np.sqrt(post_var),
        "n_rounds": np.bincount(p_codes, minlength=P),
        "n_eff": n_eff,
        "shrink": n_eff / (n_eff + lam),
        "raw_mean": raw.to_numpy(),
    }, index=pd.Index(players, name="player_id"))
    return RatingFit(metric, skills, pd.Series(fadj, index=rids), float(np.sqrt(sigma_e2)),
                     float(np.sqrt(sigma_s2)), float(lam))


def rate_all(rounds: pd.DataFrame, metrics=("sg_total", "sg_OTT", "sg_APP", "sg_ARG", "sg_PUTT"),
             detail_filter=None, **kw) -> tuple[pd.DataFrame, dict]:
    """
    Fit every metric and return one wide table plus the fit objects.
    detail_filter : boolean mask applied to rounds for the game-area metrics
                    (e.g. exclude low-quality events).
    """
    fits, cols = {}, []
    for m in metrics:
        data = rounds if (m == "sg_total" or detail_filter is None) else rounds[detail_filter]
        f = fit_ratings(data, m, **kw)
        fits[m] = f
        s = f.skills[["skill", "sd"]].rename(columns={"skill": m.replace("sg_", ""), "sd": m.replace("sg_", "") + "_sd"})
        cols.append(s)
    table = pd.concat(cols, axis=1)
    table["n_rounds"] = fits["sg_total"].skills["n_rounds"]
    table["n_eff"] = fits["sg_total"].skills["n_eff"]
    return table, fits
