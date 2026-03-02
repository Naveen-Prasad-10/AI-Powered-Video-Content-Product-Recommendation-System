"""
optimizer.py — Non-Linear Dynamic Utility (NDU) Framework
ShopVision Pro v4.0

Implements the NDU objective function for real-time vendor ranking:

    U_i = (w_p · e^(-γ·P̂_i)) + (w_t · e^(-δ·T̂_i)) + (w_r · ln(1 + R̂_i))

Where:
    P̂_i, T̂_i, R̂_i  — Min-Max normalised price, delivery-time, and rating
    w_p, w_t, w_r    — Importance weights (default: 0.40, 0.35, 0.25)
    γ (gamma)        — Exponential decay constant for price  (default: 2.0)
    δ (delta)        — Exponential decay constant for time   (default: 1.5)

This module is framework-agnostic (no Streamlit dependency) and can be
imported standalone or cited as an independent algorithm component.
"""

import math
from typing import Any, Dict, List

# ── Default Hyperparameters ───────────────────────────────────────────────────
DEFAULT_WP    = 0.40   # Weight: price       (primary consumer driver)
DEFAULT_WT    = 0.35   # Weight: delivery time (secondary preference)
DEFAULT_WR    = 0.25   # Weight: rating       (trust / quality signal)
DEFAULT_GAMMA = 2.0    # Exponential decay constant for normalised price
DEFAULT_DELTA = 1.5    # Exponential decay constant for normalised delivery time


# ── Internal helpers ──────────────────────────────────────────────────────────

def _min_max_norm(value: float, v_min: float, v_max: float) -> float:
    """
    Return Min-Max normalised value in [0, 1].
    Returns 0.0 when range is zero (degenerate / all-same-value case)
    so that the attribute contributes no discriminating penalty/reward.
    """
    if v_max == v_min:
        return 0.0
    return (value - v_min) / (v_max - v_min)


def _generate_why(winner: Dict[str, Any], all_vendors: List[Dict[str, Any]]) -> str:
    """
    Build a concise human-readable explanation of why *winner* ranked first.
    Compares winner's raw attributes against the full competitor pool.
    """
    prices  = [v["price"]         for v in all_vendors]
    times   = [v["delivery_time"] for v in all_vendors]
    ratings = [v["rating"]        for v in all_vendors]

    reasons = []

    # Price — winner is cheapest or significantly below average
    if winner["price"] <= min(prices) * 1.05:
        reasons.append(f"Cheapest price (₹{winner['price']:.0f})")
    elif winner["price"] < (sum(prices) / len(prices)):
        reasons.append(f"Below-avg price (₹{winner['price']:.0f})")

    # Delivery — winner is fastest or significantly quicker than average
    if winner["delivery_time"] <= min(times) * 1.10:
        reasons.append(f"Fastest delivery ({winner['delivery_time']} min)")
    elif winner["delivery_time"] < (sum(times) / len(times)):
        reasons.append(f"Quick delivery ({winner['delivery_time']} min)")

    # Rating — winner is top-rated or very close to the top
    if winner["rating"] >= max(ratings) * 0.97:
        reasons.append(f"Top-rated ({winner['rating']}⭐)")

    return " + ".join(reasons) if reasons else "Best weighted balance of price, speed & rating"


# ── Public API ────────────────────────────────────────────────────────────────

def rank_vendors(
    vendors:      List[Dict[str, Any]],
    wp:           float = DEFAULT_WP,
    wt:           float = DEFAULT_WT,
    wr:           float = DEFAULT_WR,
    gamma:        float = DEFAULT_GAMMA,
    delta:        float = DEFAULT_DELTA,
    delivery_cap: int   = 480,
) -> List[Dict[str, Any]]:
    """
    Rank a list of vendor dicts using the NDU objective function.

    Parameters
    ----------
    vendors : list of dicts, each containing:
        vendor_name    (str)   — display name
        price          (float) — price in local currency  (lower → better)
        delivery_time  (int)   — estimated delivery in minutes (lower → better)
        rating         (float) — platform rating 1–5   (higher → better)
        url            (str)   — purchase link
    wp, wt, wr : float
        Importance weights for price, time, and rating respectively.
    gamma, delta : float
        Exponential decay constants controlling sensitivity to price and time.
    delivery_cap : int
        Delivery times above this value (minutes) are clamped before
        normalisation. Prevents a single very-slow vendor (e.g. Amazon at
        1440 min) from stretching the scale so far that a 240-min delivery
        looks almost as fast as a 12-min one. Default: 480 min (8 hours).
        Vendors above the cap are treated equally as ‘slow’ — the algorithm
        distinguishes quick-commerce tiers, not slow-commerce granularity.

    Returns
    -------
    list of dicts
        Same vendor list, sorted descending by `utility_score`.
        - `utility_score` (float) is added to every dict.
        - `why`           (str)   is added only to the rank-1 vendor.
    """
    if not vendors:
        return []

    # ── Extract raw attribute vectors ─────────────────────────────────────────
    # ── Clamp delivery times at cap before computing statistics ──────────────
    # Platform SLA constants span [12, 1440] min — a 120× range.
    # Without capping, Amazon's 1440-min value dominates the normalisation
    # denominator, making JioMart's 240 min look nearly as fast as Blinkit's
    # 12 min. Capping at 480 min means anything slower than 8 hours is treated
    # equally as 'slow'; quick-commerce tiers are then separated by price/rating.
    capped_times = [min(v["delivery_time"], delivery_cap) for v in vendors]

    # ── Extract raw attribute vectors ─────────────────────────────────────────
    prices  = [v["price"]  for v in vendors]
    ratings = [v["rating"] for v in vendors]

    p_min, p_max = min(prices),       max(prices)
    t_min, t_max = min(capped_times), max(capped_times)
    r_min, r_max = min(ratings),      max(ratings)

    # ── Compute U_i for each vendor ───────────────────────────────────────────
    scored: List[Dict[str, Any]] = []
    for v, t_capped in zip(vendors, capped_times):
        p_hat = _min_max_norm(v["price"],  p_min, p_max)
        t_hat = _min_max_norm(t_capped,   t_min, t_max)  # uses capped delivery time
        r_hat = _min_max_norm(v["rating"], r_min, r_max)

        u_i = (
            wp * math.exp(-gamma * p_hat) +   # price component
            wt * math.exp(-delta * t_hat) +   # speed component
            wr * math.log(1.0 + r_hat)        # rating component
        )

        entry = dict(v)                       # shallow copy — never mutate input
        entry["utility_score"] = round(u_i, 6)
        scored.append(entry)

    # ── Sort descending by utility score ─────────────────────────────────────
    scored.sort(key=lambda x: x["utility_score"], reverse=True)

    # ── Annotate winner with human-readable explanation ───────────────────────
    scored[0]["why"] = _generate_why(scored[0], vendors)

    return scored


# ── Self-test — run: python optimizer.py ─────────────────────────────────────
if __name__ == "__main__":
    import json

    # ── Test 1: Coke Can (300ml) — typical quick-commerce scenario ────────────
    coke_vendors = [
        {"vendor_name": "Amazon",      "price": 55.0, "delivery_time": 90,  "rating": 4.2, "url": "#"},
        {"vendor_name": "Blinkit",     "price": 40.0, "delivery_time": 12,  "rating": 4.5, "url": "#"},
        {"vendor_name": "Zepto",       "price": 33.0, "delivery_time": 15,  "rating": 4.3, "url": "#"},
        {"vendor_name": "Local Store", "price": 30.0, "delivery_time": 4,   "rating": 3.8, "url": "#"},
    ]

    # ── Test 2: Dove Shampoo — 1-day vs 15-min delivery contrast ─────────────
    shampoo_vendors = [
        {"vendor_name": "Amazon",      "price": 178.0, "delivery_time": 1440, "rating": 4.6, "url": "#"},
        {"vendor_name": "Blinkit",     "price": 169.0, "delivery_time": 14,   "rating": 4.4, "url": "#"},
        {"vendor_name": "Zepto",       "price": 159.0, "delivery_time": 17,   "rating": 4.3, "url": "#"},
        {"vendor_name": "Local Store", "price": 155.0, "delivery_time": 7,    "rating": 3.7, "url": "#"},
    ]

    for label, vendors in [("Coke Can (300ml)", coke_vendors), ("Dove Shampoo (180ml)", shampoo_vendors)]:
        print("\n" + "=" * 64)
        print(f"  NDU FRAMEWORK SELF-TEST — {label}")
        print(f"  Weights: wp={DEFAULT_WP}  wt={DEFAULT_WT}  wr={DEFAULT_WR}")
        print(f"  γ={DEFAULT_GAMMA}  δ={DEFAULT_DELTA}")
        print("=" * 64)

        ranked = rank_vendors(vendors)

        for i, v in enumerate(ranked, 1):
            marker = "✅ WINNER" if i == 1 else f"   #{i}    "
            price_str = f"Rs.{v['price']:.0f}"
            print(
                f"  {marker}  {v['vendor_name']:<14}"
                f"  Price={price_str:<9}"
                f"  Time={v['delivery_time']:>4} min"
                f"  Rating={v['rating']}"
                f"  U={v['utility_score']:.6f}"
            )
            if i == 1:
                print(f"           Why: {v['why']}")

    print("\n" + "=" * 64)
    print("  Edge-case: single vendor (no competitors)")
    single = [{"vendor_name": "Only Shop", "price": 50.0, "delivery_time": 10, "rating": 4.0, "url": "#"}]
    r = rank_vendors(single)
    print(f"  Result: {r[0]['vendor_name']}  U={r[0]['utility_score']:.6f}")
    print("=" * 64)
