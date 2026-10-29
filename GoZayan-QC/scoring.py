"""
Pure scoring helpers (standard library only).

Used by the backend when a call is evaluated AND by the portal when a QA reviewer
overrides a rating, so both always compute the score the same way. The scheme is
passed in (from config.py on the backend, from GET /qc/config in the portal).
"""
import copy
from typing import Optional

SCORE_OUT_OF = 10


def compute_score(
    ratings: list[Optional[str]],
    rating_points: dict[str, float],
    na_policy: str = "full_credit",
    critical_error: bool = False,
    critical_error_policy: str = "flag",
) -> dict:
    """
    ratings: one of Good / Average / Bad / N/A (or None if missing) per parameter.
    Returns {"score", "out_of", "percentage", "points_earned", "applicable"}.
    """
    earned = 0.0
    applicable = 0
    for rating in ratings:
        if rating == "N/A":
            if na_policy == "full_credit":
                earned += max(rating_points.values())
                applicable += 1
            continue
        earned += rating_points.get(rating or "", 0.0)
        applicable += 1

    if na_policy == "exclude" and applicable:
        score = earned / applicable * SCORE_OUT_OF
    else:
        score = earned * SCORE_OUT_OF / max(len(ratings), 1)

    if critical_error and critical_error_policy == "zero":
        score = 0.0

    score = round(score, 2)
    return {
        "score": score,
        "out_of": SCORE_OUT_OF,
        "percentage": round(score / SCORE_OUT_OF * 100, 1),
        "points_earned": earned,
        "applicable": applicable,
    }


def band_for(score: float, bands: list[dict]) -> str:
    """bands: [{"min": 9.0, "label": "Excellent"}, ...] ordered high to low."""
    for band in bands:
        if score >= band["min"]:
            return band["label"]
    return bands[-1]["label"] if bands else ""


def rating_breakdown(ratings: list[Optional[str]], rating_points: dict[str, float]) -> dict:
    """Average points (0-1) over rated calls, ignoring N/A, plus the count of each rating."""
    rated = [rating_points[r] for r in ratings if r in rating_points]
    return {
        "average_points": round(sum(rated) / len(rated), 2) if rated else None,
        "good": ratings.count("Good"),
        "average": ratings.count("Average"),
        "bad": ratings.count("Bad"),
        "na": ratings.count("N/A"),
    }


def apply_overrides(report: dict, reviews: dict) -> dict:
    """
    Returns a copy of a BatchReport dict with QA reviewer decisions applied.

    reviews: {call_id: {"ratings": {param_key: rating}, "dismiss_critical": bool, "comment": str}}

    Every parameter gets "ai_rating" and "final_rating"; every call gets "ai_score", a recomputed
    "score" / "band" / "passed", "overrides" (count) and "reviewer_comment"; agent and team
    aggregates are recomputed from the final ratings.
    """
    cfg = report.get("config") or {}
    points = cfg.get("rating_points") or {"Good": 1.0, "Average": 0.5, "Bad": 0.0}
    na_policy = cfg.get("na_policy", "full_credit")
    critical_policy = cfg.get("critical_error_policy", "flag")
    bands = cfg.get("score_bands") or []
    pass_mark = cfg.get("pass_mark", 8)

    out = copy.deepcopy(report)
    for call in out.get("calls", []):
        review = reviews.get(call["call_id"]) or {}
        changed = review.get("ratings") or {}
        call["ai_score"] = call["score"]
        call["overrides"] = 0
        for p in call["parameters"]:
            p["ai_rating"] = p["rating"]
            final = changed.get(p["key"], p["rating"])
            if final != p["rating"]:
                call["overrides"] += 1
            p["final_rating"] = final
        critical = bool(call.get("critical_errors")) and not review.get("dismiss_critical")
        call["critical_dismissed"] = bool(call.get("critical_errors")) and bool(review.get("dismiss_critical"))
        result = compute_score([p["final_rating"] for p in call["parameters"]], points, na_policy, critical, critical_policy)
        call["score"] = result["score"]
        call["percentage"] = result["percentage"]
        call["band"] = band_for(result["score"], bands)
        call["passed"] = result["score"] >= pass_mark and not critical
        call["reviewer_comment"] = review.get("comment", "")
        call["reviewed"] = call["call_id"] in reviews

    calls = out.get("calls", [])
    for agent in out.get("agents", []):
        agent_calls = [c for c in calls if c["agent_name"] == agent["agent_name"]]
        if not agent_calls:
            continue
        agent["average_score"] = round(sum(c["score"] for c in agent_calls) / len(agent_calls), 2)
        agent["pass_rate"] = round(sum(c["passed"] for c in agent_calls) / len(agent_calls) * 100, 1)
        for i, pa in enumerate(agent["parameter_averages"]):
            pa.update(rating_breakdown([c["parameters"][i]["final_rating"] for c in agent_calls], points))
    if calls:
        out["team_average_score"] = round(sum(c["score"] for c in calls) / len(calls), 2)
        out["team_pass_rate"] = round(sum(c["passed"] for c in calls) / len(calls) * 100, 1)
    return out
