#levenshtein.py

from __future__ import annotations
from typing import Tuple, Optional


def levenshtein_distance(a: Optional[str], b: Optional[str]) -> int:
    """
    Classic Levenshtein edit distance (insert/delete/substitute).
    Memory: O(min(len(a), len(b))).
    """
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)

    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Ensure b is the shorter string to minimize memory
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))  # prev[j] = distance(a[:i-1], b[:j])

    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur

    return prev[-1]


def levenshtein_distance_bounded(a: str, b: str, max_distance: int) -> int:
    """
    Levenshtein distance with an upper bound.
    If the true distance exceeds max_distance, returns max_distance + 1.

    This is a performance optimization only; used to early-exit when a field
    cannot possibly meet its similarity threshold.
    """
    if max_distance < 0:
        return 0

    if a == b:
        return 0

    la = len(a)
    lb = len(b)

    if la == 0:
        return 0 if lb == 0 else (lb if lb <= max_distance else max_distance + 1)
    if lb == 0:
        return la if la <= max_distance else max_distance + 1

    if abs(la - lb) > max_distance:
        return max_distance + 1

    # Ensure b is the shorter string to minimize memory
    if la < lb:
        a, b = b, a
        la, lb = lb, la

    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [max_distance + 1] * (lb + 1)

        start_j = max(1, i - max_distance)
        end_j = min(lb, i + max_distance)

        if start_j == 1:
            cur[0] = i

        if start_j > end_j:
            return max_distance + 1

        for j in range(start_j, end_j + 1):
            cb = b[j - 1]
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur[j] = min(ins, dele, sub)

        prev = cur

    d = prev[lb]
    return d if d <= max_distance else max_distance + 1


def conservative_similarity(a: Optional[str], b: Optional[str]) -> Tuple[float, int]:

    """
    Your model:
      sim = 1 - (lev(a,b) / min(len(a),len(b)))

    Returns: (similarity in [0..1], distance)
    Notes:
      - If either value is missing/empty -> sim=0 (including both empty)
      - Clamps sim to [0,1] (so you never get negative %)
    """
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)

    # Missing value rule: missing/blank contributes 0 similarity for this field.
    if len(a) == 0 or len(b) == 0:
        if len(a) == 0 and len(b) == 0:
            return 0.0, 0
        return 0.0, max(len(a), len(b))

    if a == b:
        return 1.0, 0

    min_len = min(len(a), len(b))
    if min_len == 0:
        # One is empty, the other isn't
        return 0.0, max(len(a), len(b))

    d = levenshtein_distance(a, b)
    sim = 1.0 - (d / float(min_len))

    # clamp
    if sim < 0.0:
        sim = 0.0
    elif sim > 1.0:
        sim = 1.0

    return sim, d


def conservative_match(a: Optional[str], b: Optional[str], threshold: float) -> Tuple[bool, float, int]:
    """
    Returns: (is_match, similarity, distance)

    threshold can be:
      - 0..1 (e.g., 0.85)
      - 0..100 (e.g., 85)

    Performance shortcuts (no quality change):
      - If threshold is 100%, match is strict equality (non-blank) and we skip Levenshtein.
      - Otherwise, compute the maximum allowed edit distance implied by the threshold and
        stop Levenshtein early once it cannot pass.
    """
    t = float(threshold)
    if t > 1.0:
        t = t / 100.0
    if not (0.0 <= t <= 1.0):
        raise ValueError("threshold must be in [0..1] or [0..100]")

    a0 = "" if a is None else str(a)
    b0 = "" if b is None else str(b)

    # Missing value rule: missing/blank is automatic 0 match for this field.
    if len(a0) == 0 or len(b0) == 0:
        if len(a0) == 0 and len(b0) == 0:
            return False, 0.0, 0
        return False, 0.0, max(len(a0), len(b0))

    # Threshold=100%: only exact equality can pass. No Levenshtein needed.
    if t >= 1.0:
        if a0 == b0:
            return True, 1.0, 0
        return False, 0.0, 1

    # Exact match shortcut
    if a0 == b0:
        return True, 1.0, 0

    min_len = min(len(a0), len(b0))
    if min_len == 0:
        return False, 0.0, max(len(a0), len(b0))

    # sim = 1 - d/min_len >= t  =>  d <= (1 - t) * min_len
    max_d = int((1.0 - t) * float(min_len) + 1e-9)

    if max_d <= 0:
        return False, 0.0, 1

    d = levenshtein_distance_bounded(a0, b0, max_d)
    sim = 1.0 - (d / float(min_len))

    if sim < 0.0:
        sim = 0.0
    elif sim > 1.0:
        sim = 1.0

    return (sim >= t), sim, d


def weighted_gated_score(
    a_values,
    b_values,
    weights,
    thresholds,
) -> float:
    """
    Weighted + gated scoring used by your match model.

    For each field i:
      - compute sim_i using conservative_similarity()
      - if sim_i < threshold_i -> contribution_i = 0
      - else contribution_i = sim_i * weight_i

    total_score = sum(contribution_i)

    Notes:
      - weights are expected to be normalized (sum ~= 1.0) by the caller
      - thresholds can be 0..1 or 0..100 (same as conservative_match)
    """
    score = 0.0

    for a, b, w, t in zip(a_values, b_values, weights, thresholds):
        try:
            wv = float(w)
        except Exception:
            continue

        if wv <= 0.0:
            continue

        ok, sim, _ = conservative_match(a, b, t)
        if ok:
            score += float(sim) * wv

    return float(score)



if __name__ == "__main__":
    tests = [("cat", "bat"), ("grab", "crab"), ("cat", "cater")]
    for x, y in tests:
        ok, sim, d = conservative_match(x, y, 0.5)
        print(f"{x!r} vs {y!r} -> d={d}, sim={sim:.4f}, match={ok}")
