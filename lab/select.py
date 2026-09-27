"""Diversity sheet: high score first, then Anthropic's caps as a tie-break.

Levenshtein ≥ 5, ≤50% per generator, prefer ≥3 generators when the pool has them.
We do not pad. A short sheet is a short sheet.
"""

from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def select_sheet(ranked: list[dict], n: int = 20, min_lev: int = 5) -> list[dict]:
    picked: list[dict] = []
    gen_counts: dict[str, int] = {}
    max_per_gen = max(1, n // 2)

    def ok(d: dict) -> bool:
        seq = d.get("sequence") or ""
        if any(levenshtein(seq, p.get("sequence") or "") < min_lev for p in picked):
            return False
        g = d.get("generator") or "unknown"
        if gen_counts.get(g, 0) >= max_per_gen:
            return False
        return True

    for d in ranked:
        if len(picked) >= n:
            break
        if not d.get("gates_ok", True):
            continue
        if ok(d):
            picked.append(d)
            g = d.get("generator") or "unknown"
            gen_counts[g] = gen_counts.get(g, 0) + 1

    gens = {p.get("generator") for p in picked}
    if len(gens) < 3:
        for d in ranked:
            if len(picked) >= n:
                break
            if d in picked or not d.get("gates_ok", True):
                continue
            g = d.get("generator") or "unknown"
            if g in gens:
                continue
            if any(levenshtein(d.get("sequence") or "", p.get("sequence") or "") < min_lev for p in picked):
                continue
            picked.append(d)
            gens.add(g)
            gen_counts[g] = gen_counts.get(g, 0) + 1

    for i, d in enumerate(picked, 1):
        d["sheet_rank"] = i
    return picked
