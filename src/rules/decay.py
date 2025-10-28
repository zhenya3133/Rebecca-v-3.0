from math import exp


def decay_weight(w: float, dt_days: float, lam: float, boosts: float = 0.0, immune: bool = False) -> float:
    if immune:
        return w
    return w * exp(-lam * dt_days) + boosts
