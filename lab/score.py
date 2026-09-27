"""Score a one-target pool: z(mean ipSAE) + 0.5 z(optimization_rounds).

Shipped ranker (LAB.md §6b). Learned models lost under leave-one-target-out.
z-score inside this pool — at competition time we only ever rank one target.
"""

from __future__ import annotations

import math


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 1.0
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return math.sqrt(var) if var > 1e-18 else 1.0


def _z(values: list[float | None]) -> list[float]:
    present = [v for v in values if v is not None and not math.isnan(v)]
    if not present:
        return [0.0 for _ in values]
    mu, sd = _mean(present), _std(present)
    out = []
    for v in values:
        if v is None or math.isnan(v):
            out.append(0.0)
        else:
            out.append((v - mu) / sd)
    return out


def mean_ipsae(ipsae_by_predictor: dict[str, float | None]) -> float | None:
    vals = [v for v in ipsae_by_predictor.values() if v is not None]
    return _mean(vals) if vals else None


def score_pool(designs: list[dict]) -> list[dict]:
    """Mutate each design with mean_ipsae, z_ipsae, z_rounds, score. Return same list."""
    means = [d.get("mean_ipsae") for d in designs]
    if all(m is None for m in means):
        for d in designs:
            d["mean_ipsae"] = mean_ipsae(d.get("ipsae") or {})
        means = [d.get("mean_ipsae") for d in designs]
    rounds = []
    for d in designs:
        r = d.get("optimization_rounds")
        try:
            rounds.append(float(r) if r is not None else 0.0)
        except (TypeError, ValueError):
            rounds.append(0.0)
    z_i = _z(means)
    z_r = _z(rounds)
    for d, zi, zr, r in zip(designs, z_i, z_r, rounds):
        d["z_ipsae"] = round(zi, 4)
        d["z_rounds"] = round(zr, 4)
        d["score"] = round(zi + 0.5 * zr, 4)
        d["optimization_rounds"] = int(r)
    designs.sort(key=lambda d: (-(d.get("score") or -1e9), d.get("id") or ""))
    for i, d in enumerate(designs, 1):
        d["rank"] = i
    return designs
