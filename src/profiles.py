"""Style profile per player: how she plays, not just how well."""
import numpy as np
import pandas as pd


def style_profile(shots: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive tee-shot and scoring stats from high-quality detail holes.

    drive_yd      : median yards gained toward the hole on par-4/5 tee shots
    fairway_pct   : share of those tee shots finishing on the fairway (LIE_4)
    ott_penalty   : share of those tee shots followed by a penalty stroke
    green_hit_pct : share of approach shots (APP) finishing on the green
    """
    s = shots[~shots.low_quality_event & ~shots.score_only_hole & shots.lie_before.notna() & ~shots.is_penalty]
    ott = s[s.category == "OTT"]
    ott = ott.assign(gain=ott.yd_before - ott.yd_after)
    nxt_pen = shots.groupby(["event_id", "player_id", "round_no", "course_no", "hole_no"], sort=False
                            ).is_penalty.shift(-1, fill_value=False).reindex(ott.index)
    app = s[s.category == "APP"]
    return pd.DataFrame({
        "drive_yd": ott.groupby("player_id").gain.median(),
        "fairway_pct": ott.assign(f=ott.lie_after.eq("LIE_4")).groupby("player_id").f.mean(),
        "ott_penalty": nxt_pen.groupby(ott.player_id).mean(),
        "green_hit_pct": app.assign(g=app.lie_after.isin(["LIE_6", "LIE_9"])).groupby("player_id").g.mean(),
        "n_drives": ott.groupby("player_id").size(),
    })
