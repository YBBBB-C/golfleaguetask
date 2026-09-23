"""
Three-layer simulator.

Layer 1  Shot model (Monte Carlo)
    A "state" is (baseline lie, distance bin). For every state we keep a pool
    of real strokes the field hit from that state, with their outcomes
    (next lie, next distance, penalty strokes, holed?) and their SG.
    To simulate a stroke we replay one of those real outcomes.
    Player skill enters by EXPONENTIAL TILTING: outcomes are re-weighted by
    exp(kappa * sg), with kappa chosen so the weighted mean SG shifts by
    exactly the player's per-stroke skill in that game area. The shape of the
    distribution (water, recoveries, long putts holed) stays the real one.

Layer 2  Hole score distributions
    Simulate many copies of one hole for one pair (vectorized). Alternate
    shot: the tee shot is fixed by odd/even hole, then partners alternate
    every real stroke. Penalty strokes do not change the order of play.

Layer 3  Exact match play and fixtures
    Holes are independent given the pairs, so match play is a Markov chain on
    the running score. Dynamic programming gives exact win / halve / loss
    probabilities including early finishes (e.g. 3 & 2). A fixture is 5
    independent sub-matches, combined exactly by convolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_prep import HOLE_KEY
from .strokes_gained import CATEGORIES, OWN_CURVE, POOLED, baseline_lie

LIES = OWN_CURVE + [POOLED]
LIE_IDX = {l: i for i, l in enumerate(LIES)}
TEE, GREEN = LIE_IDX["LIE_16"], LIE_IDX["LIE_6"]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
MAX_STROKES = 15  # pick up after this many strokes (never binding in practice)


# ==========================================================================
# Layer 1: shot model
# ==========================================================================
@dataclass
class ShotModel:
    edges: np.ndarray                      # log-yard bin edges (shared by all lies)
    pool_of_state: np.ndarray              # (n_lies, n_bins) -> pool id
    pools: list                            # per pool: dict of arrays
    per_stroke_skill: pd.DataFrame         # player x category, strokes per stroke
    strokes_per_round: pd.Series           # field average strokes per category per round
    _cdf_cache: dict = field(default_factory=dict)

    # ---------------- construction ----------------
    @staticmethod
    def build(shots: pd.DataFrame, ratings: pd.DataFrame, n_bins: int = 140,
              min_pool: int = 40, max_pool: int = 4000, seed: int = 0) -> "ShotModel":
        rng = np.random.default_rng(seed)
        d = shots[~shots.low_quality_event & ~shots.score_only_hole].copy()
        real = d.lie_before.notna() & ~d.is_penalty
        keys = [d[k] for k in HOLE_KEY]
        # next real stroke's starting state (after any penalty / relief)
        nl = d.lie_before.where(real).groupby(keys, sort=False).shift(-1).groupby(keys, sort=False).bfill()
        ny = d.yd_before.where(real).groupby(keys, sort=False).shift(-1).groupby(keys, sort=False).bfill()
        cum = d.is_penalty.astype(int).groupby(keys, sort=False).cumsum()
        nc = cum.where(real).astype(float).groupby(keys, sort=False).shift(-1).groupby(keys, sort=False).bfill()
        d = d.assign(next_lie=nl, next_yd=ny, pen=(nc - cum).fillna(0), holed=d.lie_after.eq("LIE_9"))
        d = d[real & (d.holed | d.next_lie.notna())]

        edges = np.linspace(np.log(0.05), np.log(700), n_bins + 1)
        lie_i = baseline_lie(d.lie_before).map(LIE_IDX).to_numpy()
        bin_i = np.clip(np.digitize(np.log(d.yd_before.clip(lower=0.05)), edges) - 1, 0, n_bins - 1)
        nxt_lie_i = baseline_lie(d.next_lie.fillna("LIE_6")).map(LIE_IDX).to_numpy()

        pools, pool_of_state = [], -np.ones((len(LIES), n_bins), dtype=int)
        state = lie_i * n_bins + bin_i
        order = np.argsort(state, kind="stable")
        st_sorted = state[order]
        starts = np.searchsorted(st_sorted, np.arange(len(LIES) * n_bins))
        ends = np.searchsorted(st_sorted, np.arange(len(LIES) * n_bins), side="right")
        cols = dict(sg=d.sg.to_numpy(), next_lie=nxt_lie_i, next_yd=d.next_yd.fillna(0).to_numpy(),
                    start_yd=d.yd_before.clip(lower=0.05).to_numpy(),
                    pen=d.pen.to_numpy().astype(int), holed=d.holed.to_numpy())
        for li in range(len(LIES)):
            for bi in range(n_bins):
                s = li * n_bins + bi
                idx = order[starts[s]:ends[s]]
                if len(idx) >= min_pool:
                    if len(idx) > max_pool:
                        idx = rng.choice(idx, max_pool, replace=False)
                    pool_of_state[li, bi] = len(pools)
                    pools.append({k: v[idx] for k, v in cols.items()})
            # states with too little data borrow the nearest well-populated bin of the same lie
            have = np.where(pool_of_state[li] >= 0)[0]
            for bi in np.where(pool_of_state[li] < 0)[0]:
                pool_of_state[li, bi] = pool_of_state[li, have[np.argmin(np.abs(have - bi))]]

        # field-average strokes per category per 18-hole round, to turn
        # "strokes per round" ratings into "strokes per stroke"
        det = shots[~shots.low_quality_event & ~shots.score_only_hole & shots.lie_before.notna() & ~shots.is_penalty]
        n_rounds = det.groupby(["event_id", "player_id", "round_no", "course_no"]).ngroups
        spr = det.category.value_counts().reindex(CATEGORIES) / n_rounds
        per_stroke = ratings[CATEGORIES].div(spr, axis=1).astype(float)
        return ShotModel(edges, pool_of_state, pools, per_stroke, spr)

    # ---------------- sampling ----------------
    def state_pool(self, lie: np.ndarray, yd: np.ndarray) -> np.ndarray:
        b = np.clip(np.digitize(np.log(np.clip(yd, 0.05, None)), self.edges) - 1, 0, len(self.edges) - 2)
        return self.pool_of_state[lie, b]

    def _cdf(self, pool_id: int, delta: float) -> np.ndarray:
        """CDF over a pool, tilted so the mean SG rises by `delta` strokes."""
        key = (pool_id, round(float(delta), 5))
        c = self._cdf_cache.get(key)
        if c is not None:
            return c
        sg = self.pools[pool_id]["sg"]
        z0 = sg - sg.mean()
        spread = np.ptp(sg)
        if sg.var() < 1e-6 or delta == 0:
            w = np.full(len(sg), 1 / len(sg))   # nothing to tilt (e.g. tap-ins)
        else:
            k_max = 30.0 / spread                # keeps weights finite and sane
            target = sg.mean() + delta
            kappa = np.clip(delta / sg.var(), -k_max, k_max)   # first-order start
            for _ in range(12):                                # Newton refinement
                z = kappa * z0
                w = np.exp(z - z.max())
                w /= w.sum()
                m = w @ sg
                v = w @ (sg - m) ** 2
                if v <= 1e-12 or abs(m - target) < 1e-7:
                    break
                kappa = np.clip(kappa + (target - m) / v, -k_max, k_max)
        c = np.cumsum(w)
        c /= c[-1]
        self._cdf_cache[key] = c
        return c

    def add_player(self, name: str, per_round: dict) -> None:
        """Register a hypothetical player from game-area skills in strokes per round."""
        row = pd.Series(per_round).reindex(CATEGORIES).fillna(0) / self.strokes_per_round
        self.per_stroke_skill.loc[name] = row

    def skill(self, player, cat_i: int) -> float:
        if player is None:
            return 0.0
        return float(self.per_stroke_skill.at[player, CATEGORIES[cat_i]])


def _category(lie: np.ndarray, yd: np.ndarray, par: int) -> np.ndarray:
    return np.select(
        [(lie == TEE) & (par >= 4), lie == GREEN, yd > 100],
        [CAT_IDX["OTT"], CAT_IDX["PUTT"], CAT_IDX["APP"]],
        CAT_IDX["ARG"],
    )


# ==========================================================================
# Layer 2: holes
# ==========================================================================
def simulate_hole(model: ShotModel, hitters: tuple, tee_yd: float, par: int, n: int,
                  rng: np.random.Generator, tally: np.ndarray | None = None,
                  roles: tuple = (0, 1)) -> np.ndarray:
    """
    Scores for n independent plays of one hole.
    hitters : (teer, partner). Real strokes alternate teer, partner, teer, ...
              Use (p, p) for an individual, (None, None) for a field-average player.
    tally   : optional (2, 4) array; adds the number of real strokes each role
              hits in each game area (summed over the n plays).
    roles   : role index of (teer, partner) in `tally`.
    """
    lie = np.full(n, TEE)
    yd = np.full(n, float(tee_yd))
    score = np.zeros(n, dtype=int)
    live = np.ones(n, dtype=bool)
    for k in range(MAX_STROKES):
        if not live.any():
            break
        who = hitters[k % 2]
        ia = np.where(live)[0]
        pid = model.state_pool(lie[ia], yd[ia])
        cat = _category(lie[ia], yd[ia], par)
        grp = pid * 4 + cat
        for g in np.unique(grp):
            m = ia[grp == g]
            p_id, c_i = divmod(int(g), 4)
            if tally is not None:
                tally[roles[k % 2], c_i] += len(m)
            pool = model.pools[p_id]
            cdf = model._cdf(p_id, model.skill(who, c_i))
            j = np.minimum(np.searchsorted(cdf, rng.random(len(m))), len(cdf) - 1)
            score[m] += 1 + pool["pen"][j]
            h = pool["holed"][j]
            live[m[h]] = False
            lie[m] = pool["next_lie"][j]
            # replayed outcome, rescaled to this ball's exact starting distance
            # (bins are ~7% wide). A re-tee after out-of-bounds therefore
            # returns to exactly the original hole length.
            yd[m] = pool["next_yd"][j] * (yd[m] / pool["start_yd"][j])
    score[live] += 2  # picked up: charge two more strokes (practically never happens)
    return score


def hole_pmfs(model: ShotModel, course: pd.DataFrame, pair: tuple, n: int = 4000,
              seed: int = 0, max_score: int = 12) -> np.ndarray:
    """
    Score distribution on every hole of `course` for one pair.
    pair = (odd_teer, even_teer). Returns array (18, max_score + 1).
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((len(course), max_score + 1))
    a, b = pair
    for i, h in enumerate(course.itertuples()):
        hitters = (a, b) if h.hole_no % 2 == 1 else (b, a)
        s = np.minimum(simulate_hole(model, hitters, h.yd, h.par, n, rng), max_score)
        out[i] = np.bincount(s, minlength=max_score + 1) / n
    return out


# ==========================================================================
# Layer 3: match play and fixtures (exact)
# ==========================================================================
def hole_outcomes(pmf_a: np.ndarray, pmf_b: np.ndarray) -> np.ndarray:
    """Per hole: P(A wins hole), P(halved), P(B wins hole). Shape (18, 3)."""
    ca = np.cumsum(pmf_a, axis=1)
    cb = np.cumsum(pmf_b, axis=1)
    p_a = np.sum(pmf_a[:, :-1] * (1 - cb[:, :-1]), axis=1)  # A scores s, B scores more than s
    p_b = np.sum(pmf_b[:, :-1] * (1 - ca[:, :-1]), axis=1)
    p_h = 1 - p_a - p_b
    return np.c_[p_a, p_h, p_b]


def match_play(outc: np.ndarray) -> dict:
    """
    Exact match-play result from per-hole outcome probabilities.
    Running score = A's lead. The match stops once |lead| > holes remaining.
    Returns P(A wins), P(halved), P(B wins), expected holes played and the
    distribution of final results ('3&2', '1 up', 'halved', ...).
    """
    H = len(outc)
    off = H
    dist = np.zeros(2 * H + 1)
    dist[off] = 1.0
    finals, holes_played = {}, 0.0
    for i, (pa, ph, pb) in enumerate(outc):
        new = np.zeros_like(dist)
        new[1:] += dist[:-1] * pa
        new += dist * ph
        new[:-1] += dist[1:] * pb
        dist = new
        remaining = H - (i + 1)
        for lead in range(-H, H + 1):
            p = dist[lead + off]
            if p > 0 and abs(lead) > remaining:
                tag = (f"A {abs(lead)}&{remaining}" if lead > 0 else f"B {abs(lead)}&{remaining}") if remaining \
                    else (f"A {lead} up" if lead > 0 else f"B {-lead} up")
                finals[tag] = finals.get(tag, 0) + p
                holes_played += p * (i + 1)
                dist[lead + off] = 0.0
    finals["halved"] = dist[off]
    holes_played += dist[off] * H
    p_a = sum(v for k, v in finals.items() if k.startswith("A"))
    p_b = sum(v for k, v in finals.items() if k.startswith("B"))
    return {"A": p_a, "halved": finals["halved"], "B": p_b, "holes_played": holes_played, "finals": finals}


def fixture(match_results: list[dict]) -> dict:
    """
    Combine 5 independent sub-matches. Team A points in half-point units.
    Early closure at 3.0 points changes when play stops, not who wins,
    so the exact outcome is read off the full points distribution.
    """
    dist = np.array([1.0])
    for m in match_results:
        step = np.array([m["B"], m["halved"], m["A"]])  # 0, 0.5, 1 point for A
        dist = np.convolve(dist, step)
    half_points = np.arange(len(dist))
    n = len(match_results)
    win = dist[half_points > n].sum()
    lose = dist[half_points < n].sum()
    return {"A_win": win, "draw": dist[half_points == n].sum(), "B_win": lose,
            "exp_points_A": (dist * half_points / 2).sum(),
            "points_dist": pd.Series(dist, index=half_points / 2)}


# ==========================================================================
# Courses
# ==========================================================================
def league_course(course_holes: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """
    A representative par-72 course from a high-quality event: 4 par 3s,
    10 par 4s, 4 par 5s, total length closest to the median of such courses.
    """
    hq = shots.loc[~shots.low_quality_event, "event_id"].unique()
    ch = course_holes[course_holes.event_id.isin(hq)]
    g = ch.groupby(["event_id", "course_no"])
    summ = g.agg(par=("par", "sum"), p3=("par", lambda s: (s == 3).sum()), p5=("par", lambda s: (s == 5).sum()),
                 yd=("hole_yd", "sum"))
    cand = summ[(summ.par == 72) & (summ.p3 == 4) & (summ.p5 == 4)]
    pick = (cand.yd - cand.yd.median()).abs().idxmin()
    c = ch[(ch.event_id == pick[0]) & (ch.course_no == pick[1])].sort_values("hole_no")
    return c[["hole_no", "par"]].assign(yd=c.hole_yd.round(0)).reset_index(drop=True)


def pair_vs_pair(model: ShotModel, course: pd.DataFrame, pair_a: tuple, pair_b: tuple,
                 n: int = 4000, seed: int = 0) -> dict:
    """Exact match-play probabilities for two pairs, via simulated hole distributions."""
    pa = hole_pmfs(model, course, pair_a, n=n, seed=seed)
    pb = hole_pmfs(model, course, pair_b, n=n, seed=seed + 7919)
    return match_play(hole_outcomes(pa, pb))


def monte_carlo_match(outc: np.ndarray, n: int = 200_000, seed: int = 0) -> np.ndarray:
    """Brute-force check of match_play(): returns share of A wins, halves, B wins."""
    rng = np.random.default_rng(seed)
    H = len(outc)
    u = rng.random((n, H))
    step = np.where(u < outc[:, 0], 1, np.where(u < outc[:, 0] + outc[:, 1], 0, -1))
    lead = np.cumsum(step, axis=1)
    remaining = H - np.arange(1, H + 1)
    decided = np.abs(lead) > remaining
    first = np.where(decided.any(axis=1), decided.argmax(axis=1), H - 1)
    final = lead[np.arange(n), first]
    return np.array([(final > 0).mean(), (final == 0).mean(), (final < 0).mean()])
