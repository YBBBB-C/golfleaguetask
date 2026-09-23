"""
Snake-draft engine with cumulative seed caps.

Value of a roster
-----------------
From notebook 04, a lineup's strength (strokes per round, summed over its
5 pairs) depends only on which 5 players take the odd-hole tee role:

    V(roster) = sum of the best split into 5 "odd" and 5 "even" role values.

Expected fixture points rise almost linearly with this number, so the draft
maximises the expected final V of our roster.

Opponent models
---------------
A  "seed"        : every opponent ranks players by last season's seed.
B  "noisy seed"  : (MAIN ASSUMPTION) each opponent ranks by seed x exp(eps),
                   eps ~ N(0, sigma) drawn per opponent and per player. They
                   broadly follow the ranking but disagree with it and each other.
C  "sharp"       : opponents rank by our own model rating (plus small noise).
All opponents obey the caps: they take their best-ranked FEASIBLE player.

Our policies
------------
seed      : behave like model A (a baseline).
greedy    : best available player by our rating, ignoring when others will take whom.
lookahead : for each of the top-K candidates, simulate the rest of the draft
            R times under the opponent model and pick the candidate with the
            highest expected final V. It waits on players others undervalue
            and grabs scarce ones before they go.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CAPS = [(10, 2), (20, 4), (30, 6), (40, 8), (50, 10)]   # (seed <= limit) at most cap players


def snake_order(n_teams: int, rounds: int) -> list[int]:
    seq = []
    for r in range(rounds):
        seq += list(range(n_teams)) if r % 2 == 0 else list(range(n_teams - 1, -1, -1))
    return seq


@dataclass
class Pool:
    ids: np.ndarray        # player ids
    seed: np.ndarray       # draft seed
    v_odd: np.ndarray      # role values (strokes per round)
    v_even: np.ndarray
    rating: np.ndarray     # individual rating used by greedy / sharp opponents
    tier_mask: np.ndarray  # (n_players, 5) bool: player counts toward cap tier k

    @staticmethod
    def from_values(values: pd.DataFrame, ratings: pd.DataFrame) -> "Pool":
        seed = ratings.loc[values.index, "seed"].to_numpy()
        rating = (values.odd + values.even).to_numpy() / 2
        tiers = np.array([[s <= lim for lim, _ in CAPS] for s in seed])
        return Pool(values.index.to_numpy(), seed, values.odd.to_numpy(), values.even.to_numpy(), rating, tiers)

    def roster_value(self, idx) -> float:
        idx = np.asarray(idx)
        if len(idx) == 0:
            return 0.0
        adv = self.v_odd[idx] - self.v_even[idx]
        order = np.argsort(-adv)
        h = len(idx) // 2
        return float(self.v_odd[idx[order[:h]]].sum() + self.v_even[idx[order[h:]]].sum())


CAP_LIMITS = np.array([c for _, c in CAPS])


class Draft:
    """Mutable draft state."""

    def __init__(self, pool: Pool, n_teams: int = 10, rounds: int = 10):
        self.pool, self.n_teams, self.rounds = pool, n_teams, rounds
        self.order = snake_order(n_teams, rounds)
        self.available = np.ones(len(pool.ids), dtype=bool)
        self.tier_counts = np.zeros((n_teams, len(CAPS)), dtype=int)
        self.rosters = [[] for _ in range(n_teams)]
        self.pick_no = 0

    def copy(self) -> "Draft":
        d = Draft.__new__(Draft)
        d.pool, d.n_teams, d.rounds, d.order = self.pool, self.n_teams, self.rounds, self.order
        d.available = self.available.copy()
        d.tier_counts = self.tier_counts.copy()
        d.rosters = [r.copy() for r in self.rosters]
        d.pick_no = self.pick_no
        return d

    def feasible(self, team: int, p: int) -> bool:
        return bool(np.all(self.tier_counts[team] + self.pool.tier_mask[p] <= CAP_LIMITS))

    def feasible_mask(self, team: int) -> np.ndarray:
        room = CAP_LIMITS - self.tier_counts[team]
        return self.available & np.all(self.pool.tier_mask <= room, axis=1)

    def take(self, team: int, p: int):
        assert self.available[p] and self.feasible(team, p)
        self.available[p] = False
        self.tier_counts[team] += self.pool.tier_mask[p]
        self.rosters[team].append(p)
        self.pick_no += 1

    def first_feasible(self, team: int, pref: np.ndarray) -> int:
        """First player in preference order `pref` (array of indices) this team may take."""
        room = CAP_LIMITS - self.tier_counts[team]
        for p in pref:
            if self.available[p] and np.all(self.pool.tier_mask[p] <= room):
                return int(p)
        raise RuntimeError("no feasible player")

    @property
    def done(self) -> bool:
        return self.pick_no >= len(self.order)

    @property
    def on_clock(self) -> int:
        return self.order[self.pick_no]


# --------------------------------------------------------------------------
# Opponent preference lists
# --------------------------------------------------------------------------
def opponent_prefs(pool: Pool, model: str, n_teams: int, rng: np.random.Generator,
                   sigma: float = 0.35, sharp_noise: float = 0.05) -> list[np.ndarray]:
    """One preference order (best first) per team."""
    prefs = []
    for _ in range(n_teams):
        if model == "A":
            score = pool.seed.astype(float)
        elif model == "B":
            score = pool.seed * np.exp(rng.normal(0, sigma, len(pool.seed)))
        elif model == "C":
            score = -(pool.rating + rng.normal(0, sharp_noise, len(pool.seed)))
        else:
            raise ValueError(model)
        prefs.append(np.argsort(score, kind="stable"))
    return prefs


# --------------------------------------------------------------------------
# Our policies
# --------------------------------------------------------------------------
def pick_seed(d: Draft, team: int, **_) -> int:
    return d.first_feasible(team, np.argsort(d.pool.seed, kind="stable"))


def pick_greedy(d: Draft, team: int, **_) -> int:
    return d.first_feasible(team, np.argsort(-d.pool.rating, kind="stable"))


def rollout(d: Draft, us: int, prefs: list, greedy_pref: np.ndarray) -> float:
    """Play the draft to the end: opponents follow `prefs`, we follow greedy."""
    while not d.done:
        t = d.on_clock
        d.take(t, d.first_feasible(t, greedy_pref if t == us else prefs[t]))
    return d.pool.roster_value(d.rosters[us])


def pick_lookahead(d: Draft, team: int, rng: np.random.Generator, opp_model: str = "B",
                   k: int = 6, r: int = 24, return_scores: bool = False, **_):
    """
    Evaluate the top-k feasible candidates (by rating, plus the best feasible
    by seed so we can 'take the scarce one') with r rollouts each.
    Common random numbers: the same r opponent scenarios for every candidate.
    """
    feas = np.where(d.feasible_mask(team))[0]
    by_rating = feas[np.argsort(-d.pool.rating[feas])][:k]
    by_seed = feas[np.argsort(d.pool.seed[feas])][:2]
    cands = list(dict.fromkeys(list(by_rating) + list(by_seed)))
    greedy_pref = np.argsort(-d.pool.rating, kind="stable")
    scenarios = [opponent_prefs(d.pool, opp_model, d.n_teams, rng) for _ in range(r)]
    scores = {}
    for c in cands:
        vals = []
        for prefs in scenarios:
            dd = d.copy()
            dd.take(team, c)
            vals.append(rollout(dd, team, prefs, greedy_pref))
        scores[c] = float(np.mean(vals))
    best = max(scores, key=scores.get)
    return (best, scores) if return_scores else best


POLICIES = {"seed": pick_seed, "greedy": pick_greedy, "lookahead": pick_lookahead}


def run_draft(pool: Pool, our_slot: int, policy: str, opp_model: str = "B", seed: int = 0,
              n_teams: int = 10, rounds: int = 10, log: bool = False, **kw):
    """
    Full draft. our_slot is 0-based. Opponents' true preferences are drawn once
    per draft from the opponent model; our lookahead does NOT see them and
    samples its own scenarios.
    """
    rng = np.random.default_rng(seed)
    d = Draft(pool, n_teams, rounds)
    prefs = opponent_prefs(pool, opp_model, n_teams, rng)
    history = []
    while not d.done:
        t = d.on_clock
        if t == our_slot:
            if policy == "lookahead" and log:
                p, scores = pick_lookahead(d, t, rng, opp_model=opp_model, return_scores=True, **kw)
            else:
                p = POLICIES[policy](d, t, rng=rng, opp_model=opp_model, **kw)
                scores = None
        else:
            p = d.first_feasible(t, prefs[t])
            scores = None
        if log:
            history.append({"pick": d.pick_no + 1, "round": d.pick_no // n_teams + 1, "team": t + 1,
                            "player": pool.ids[p], "seed": int(pool.seed[p]), "rating": pool.rating[p],
                            "ours": t == our_slot, "scores": scores})
        d.take(t, p)
    return d, (pd.DataFrame(history) if log else None)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def lineup_pair_edges(pool: Pool, roster) -> list[float]:
    """Pair edges of the optimal-role lineup (strongest odd with strongest even; pairing does not change the total)."""
    idx = np.asarray(roster)
    adv = pool.v_odd[idx] - pool.v_even[idx]
    o = idx[np.argsort(-adv)[:5]]
    e = idx[np.argsort(-adv)[5:]]
    return sorted((np.sort(pool.v_odd[o])[::-1] + np.sort(pool.v_even[e])[::-1]).tolist(), reverse=True)


def league_table(pool: Pool, rosters: list, curve: pd.DataFrame) -> pd.DataFrame:
    """Round robin: every team plays every other once. Expected points and win % per fixture."""
    from .lineup import fixture_from_edges
    edges = [lineup_pair_edges(pool, r) for r in rosters]
    rows = []
    for i, ei in enumerate(edges):
        pts, win, draw = [], [], []
        for j, ej in enumerate(edges):
            if i == j:
                continue
            f = fixture_from_edges(ei, ej, curve)
            pts.append(f["exp_points_A"]); win.append(f["A_win"]); draw.append(f["draw"])
        rows.append({"team": i + 1, "lineup_value": sum(ei), "exp_points": np.mean(pts),
                     "win_pct": np.mean(win) * 100, "draw_pct": np.mean(draw) * 100})
    return pd.DataFrame(rows).set_index("team")


def draft_positions(pool: Pool, opp_model: str = "B", n: int = 300, seed: int = 0, n_teams: int = 10) -> pd.DataFrame:
    """
    Where does each player go when all teams draft by the opponent model?
    Returns each player's average pick number and the share of drafts in which
    she is still available at each pick (for availability forecasts).
    """
    rng = np.random.default_rng(seed)
    picks = np.zeros((n, len(pool.ids)))
    for k in range(n):
        d = Draft(pool, n_teams)
        prefs = opponent_prefs(pool, opp_model, n_teams, rng)
        while not d.done:
            t = d.on_clock
            p = d.first_feasible(t, prefs[t])
            picks[k, p] = d.pick_no + 1
            d.take(t, p)
    picks[picks == 0] = np.inf   # undrafted
    return pd.DataFrame(picks, columns=pool.ids)
