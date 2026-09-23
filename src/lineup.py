"""
Pairing and lineup tools built on the simulator.

Roles
-----
In each pair one player tees off on odd holes ("odd" role), the other on
even holes ("even" role). On a given course each role hits a predictable
number of shots in each game area. Because skill enters the simulator per
stroke, a pair's edge over an average pair is, to first order,

    edge(A odd, B even) = sum_c  n_odd[c] * s_A[c]  +  n_even[c] * s_B[c]

where n_role[c] = shots that role hits in area c per round and s_X[c] is
player X's per-stroke skill. This is the "analytic pair rating".
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from .simulator import CATEGORIES, ShotModel, simulate_hole, match_play, hole_outcomes, hole_pmfs, fixture

ROLES = ["odd", "even"]


def role_shot_counts(model: ShotModel, course: pd.DataFrame, n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Expected real strokes per 18 holes, by role (odd/even tee) and game area, for an average pair."""
    rng = np.random.default_rng(seed)
    tally = np.zeros((2, 4))
    for h in course.itertuples():
        roles = (0, 1) if h.hole_no % 2 == 1 else (1, 0)   # who tees on this hole
        simulate_hole(model, (None, None), h.yd, h.par, n, rng, tally=tally, roles=roles)
    return pd.DataFrame(tally / n, index=ROLES, columns=CATEGORIES)


def role_values(model: ShotModel, counts: pd.DataFrame, players=None) -> pd.DataFrame:
    """Each player's contribution (strokes per round) when playing the odd or the even role."""
    s = model.per_stroke_skill if players is None else model.per_stroke_skill.loc[players]
    s = s[CATEGORIES]
    return pd.DataFrame({"odd": s @ counts.loc["odd"], "even": s @ counts.loc["even"]})


def pair_edge(values: pd.DataFrame, odd_player, even_player) -> float:
    return values.at[odd_player, "odd"] + values.at[even_player, "even"]


def best_orientation(values: pd.DataFrame, a, b) -> tuple[tuple, float]:
    """Return ((odd, even), edge) for the better of the two tee assignments."""
    e1, e2 = pair_edge(values, a, b), pair_edge(values, b, a)
    return ((a, b), e1) if e1 >= e2 else ((b, a), e2)


# --------------------------------------------------------------------------
# Edge -> match outcome calibration
# --------------------------------------------------------------------------
def outcome_curve(model: ShotModel, course: pd.DataFrame, edges=np.arange(-3.0, 3.01, 0.25),
                  n: int = 12000, seed: int = 21, split: pd.Series | None = None) -> pd.DataFrame:
    """
    P(win / halve / lose) of a pair with the given edge (strokes per round)
    against an average pair, from full simulation with common random numbers.
    """
    if split is None:
        split = pd.Series(1 / 4, index=CATEGORIES)
    base = hole_pmfs(model, course, (None, None), n=n, seed=seed)
    rows = []
    for x in edges:
        name = f"_edge{x:+.2f}"
        model.add_player(name, (split * x).to_dict())
        pm = hole_pmfs(model, course, (name, name), n=n, seed=seed)
        r = match_play(hole_outcomes(pm, base))
        rows.append({"edge": x, "win": r["A"], "halve": r["halved"], "lose": r["B"]})
    return pd.DataFrame(rows).set_index("edge")


def match_from_edges(edge_a: float, edge_b: float, curve: pd.DataFrame) -> dict:
    """Approximate a pair-vs-pair match by the edge difference, read off the calibration curve."""
    d = edge_a - edge_b
    x = curve.index.to_numpy()
    w = np.interp(d, x, curve.win); h = np.interp(d, x, curve.halve); l = np.interp(d, x, curve.lose)
    t = w + h + l
    return {"A": w / t, "halved": h / t, "B": l / t}


def fixture_from_edges(ours: list[float], theirs: list[float], curve: pd.DataFrame) -> dict:
    return fixture([match_from_edges(a, b, curve) for a, b in zip(ours, theirs)])


# --------------------------------------------------------------------------
# Lineup search
# --------------------------------------------------------------------------
def all_matchings(players: list):
    """All ways to split an even-sized list into unordered pairs (945 for 10 players)."""
    if not players:
        yield []
        return
    a = players[0]
    for i in range(1, len(players)):
        b = players[i]
        rest = players[1:i] + players[i + 1:]
        for m in all_matchings(rest):
            yield [(a, b)] + m


def best_role_split(values: pd.DataFrame, roster: list) -> tuple[list, list, float]:
    """
    Maximise total edge of the lineup. Because the total is a sum of each
    player's role value, only WHO takes the odd role matters, not who
    partners whom: give the odd role to the 5 players with the largest
    (odd - even) advantage.
    """
    v = values.loc[roster]
    adv = (v.odd - v.even).sort_values(ascending=False)
    odd = adv.index[:len(roster) // 2].tolist()
    even = adv.index[len(roster) // 2:].tolist()
    return odd, even, v.loc[odd, "odd"].sum() + v.loc[even, "even"].sum()


def recommend_lineup(values: pd.DataFrame, roster: list, curve: pd.DataFrame,
                     opponent_edges: list | None = None) -> pd.DataFrame:
    """
    Full lineup recommendation for a 10-player roster.

    1. Tee roles: best_role_split (maximises expected points).
    2. Pairing: among the 120 ways to match odd-role and even-role players,
       pick the one with the highest fixture win probability.
    3. Order: if the opponent's pair edges are known in order, choose our
       order as a best response; otherwise any order is equivalent and we
       list pairs strongest first.
    """
    odd, even, _ = best_role_split(values, roster)
    odd = sorted(odd, key=lambda p: -values.at[p, "odd"])
    even = sorted(even, key=lambda p: -values.at[p, "even"])
    opp = [0.0] * 5 if opponent_edges is None else list(opponent_edges)
    best = None
    for m in permutations(range(5)):
        pairs = [(odd[i], even[m[i]]) for i in range(5)]
        edges = [values.at[a, "odd"] + values.at[b, "even"] for a, b in pairs]
        if opponent_edges is None:
            orders = [tuple(range(5))]
        else:
            orders = permutations(range(5))
        for o in orders:
            f = fixture_from_edges([edges[i] for i in o], opp, curve)
            key = (f["A_win"], f["exp_points_A"])
            if best is None or key > best[0]:
                best = (key, [pairs[i] for i in o], [edges[i] for i in o], f)
    (_, pairs, edges, f) = best
    out = pd.DataFrame({"odd_tee": [p[0] for p in pairs], "even_tee": [p[1] for p in pairs], "pair_edge": edges},
                       index=pd.Index(range(1, 6), name="slot"))
    if opponent_edges is None:
        out = out.sort_values("pair_edge", ascending=False)
        out.index = pd.Index(range(1, 6), name="slot")
    out.attrs.update(win=f["A_win"], draw=f["draw"], exp_points=f["exp_points_A"])
    return out
