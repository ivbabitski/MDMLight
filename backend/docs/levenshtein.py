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


def conservative_similarity(a: Optional[str], b: Optional[str]) -> Tuple[float, int]:
    """
    Your model:
      sim = 1 - (lev(a,b) / min(len(a),len(b)))

    Returns: (similarity in [0..1], distance)
    Notes:
      - If both empty -> sim=1
      - If one empty -> sim=0
      - Clamps sim to [0,1] (so you never get negative %)
    """
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)

    if a == b:
        return 1.0, 0

    min_len = min(len(a), len(b))
    if min_len == 0:
        # One is empty, the other isn't (since a != b here)
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
    """
    t = float(threshold)
    if t > 1.0:
        t = t / 100.0
    if not (0.0 <= t <= 1.0):
        raise ValueError("threshold must be in [0..1] or [0..100]")

    sim, d = conservative_similarity(a, b)
    return (sim >= t), sim, d


if __name__ == "__main__":
    tests = [("cat", "bat"), ("grab", "crab"), ("cat", "cater")]
    for x, y in tests:
        ok, sim, d = conservative_match(x, y, 0.5)
        print(f"{x!r} vs {y!r} -> d={d}, sim={sim:.4f}, match={ok}")
