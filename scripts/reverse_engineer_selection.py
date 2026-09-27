#!/usr/bin/env python3
"""Reverse-engineer the Claude-based design selection workflow.

The competition FAQ says designs are chosen for wet-lab testing by a
"Claude-based workflow" that is secret until after the competition and that
"will not rely on a single in silico metric".

We cannot read that workflow. But the released dataset contains Claude doing
the *same task*: in each campaign an agent generated a large candidate pool
and selected/ordered 30 designs as rank01..rank30. The `rank` column is
therefore an observed sample of Claude's selection policy, and we know the
wet-lab outcome of every one of those picks.

This script asks four questions:
  1. Does Claude's rank have skill? (does rank01-10 bind more than rank21-30)
  2. Is rank just a sort on consensus ipSAE?
  3. Does rank add anything BEYOND consensus ipSAE?
  4. Which developability heuristics actually survive a within-target test?

Question 4 matters because between-target confounding is severe here: a
feature can look strongly predictive pooled and be worth exactly nothing
within a target, which is the only setting we ever act in.

Stdlib only. Run scripts/baseline_ranker.py first to fetch the CSV.
"""

import collections
import csv
import math
import os
import statistics

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "design_summary.csv")


def load():
    with open(CSV_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows, [r for r in rows if r["binder_final"] in ("True", "False")]


def num(row, col):
    try:
        return float(row[col])
    except (ValueError, KeyError):
        return None


def auc(pairs):
    pairs = [(s, y) for s, y in pairs if s is not None]
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    pairs.sort(key=lambda t: t[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(ranks[k] for k in range(len(pairs)) if pairs[k][1])
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def within_target_auc(rows, score_fn):
    """Pair-count-weighted mean of per-target AUC. The metric that matters."""
    num_, den = 0.0, 0.0
    for target in {r["target"] for r in rows}:
        sub = [(score_fn(r), r["binder_final"] == "True") for r in rows if r["target"] == target]
        sub = [(s, y) for s, y in sub if s is not None]
        a = auc(sub)
        if a is None:
            continue
        n_pos = sum(1 for _, y in sub if y)
        w = n_pos * (len(sub) - n_pos)
        num_ += a * w
        den += w
    return num_ / den if den else None


def spearman(xs, ys):
    n = len(xs)
    def ranked(v):
        order = sorted(range(n), key=lambda i: v[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out
    a, b = ranked(xs), ranked(ys)
    ma, mb = statistics.mean(a), statistics.mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return cov / den if den else 0.0


def main():
    rows, lab = load()
    predictors = [c.replace("ipsae_min_", "") for c in rows[0] if c.startswith("ipsae_min_")]

    def mean_ipsae(r):
        v = [x for x in (num(r, "ipsae_min_" + p) for p in predictors) if x is not None]
        return sum(v) / len(v) if v else None

    print("Q1. Does Claude's rank have skill?")
    buckets = collections.defaultdict(lambda: [0, 0])
    for r in lab:
        k = int(r["rank"])
        b = "rank01-10" if k <= 10 else ("rank11-20" if k <= 20 else "rank21-30")
        buckets[b][0] += r["binder_final"] == "True"
        buckets[b][1] += 1
    for b in sorted(buckets):
        hits, n = buckets[b]
        print(f"   {b}  {hits:3d}/{n}  {hits / n:.1%} bind")

    print("\nQ2. Is rank just a sort on consensus ipSAE?")
    groups = collections.defaultdict(list)
    for r in rows:
        if mean_ipsae(r) is not None:
            groups[(r["design_model"], r["campaign"], r["target"])].append(r)
    cors = [spearman([int(x["rank"]) for x in g], [mean_ipsae(x) for x in g])
            for g in groups.values() if len(g) >= 10]
    print(f"   spearman(rank, mean ipSAE) over {len(cors)} campaign x target groups")
    print(f"   median {statistics.median(cors):+.3f}   (-1.0 would be a pure sort)")

    print("\nQ3. Does rank beat / add to consensus ipSAE?")
    print(f"   Claude rank        within-target AUC {within_target_auc(lab, lambda r: -int(r['rank'])):.3f}")
    print(f"   mean ipSAE         within-target AUC {within_target_auc(lab, mean_ipsae):.3f}")
    strata = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for target in {r["target"] for r in lab}:
        g = sorted([r for r in lab if r["target"] == target and mean_ipsae(r) is not None],
                   key=mean_ipsae)
        for i, r in enumerate(g):
            q = min(4, i * 5 // len(g))
            k = int(r["rank"])
            b = "top10" if k <= 10 else ("mid" if k <= 20 else "bot10")
            strata[q][b][0] += r["binder_final"] == "True"
            strata[q][b][1] += 1
    print(f"   {'ipSAE quintile':16}{'rank01-10':>12}{'rank11-20':>12}{'rank21-30':>12}")
    for q in range(5):
        cells = []
        for b in ("top10", "mid", "bot10"):
            hits, n = strata[q][b]
            cells.append(f"{hits}/{n}={hits / n:.0%}" if n else "-")
        tag = "(low)" if q == 0 else ("(high)" if q == 4 else "")
        print(f"   Q{q + 1} {tag:12}" + "".join(f"{c:>12}" for c in cells))

    print("\nQ4. Developability heuristics: pooled intuition vs within-target truth")
    heuristics = [
        ("smaller epitope", lambda r: -int(r["epitope_n_residues"]) if r["epitope_n_residues"] else None),
        ("fewer cysteines", lambda r: -r["sequence"].count("C")),
        ("shorter binder", lambda r: -int(r["binder_length"])),
    ]
    for name, fn in heuristics:
        print(f"   {name:18} within-target AUC {within_target_auc(lab, fn):.3f}")
    print("   cysteine count, pooled binding rate:")
    for n_cys in range(3):
        sub = [r for r in lab if r["sequence"].count("C") == n_cys]
        if len(sub) < 20:
            continue
        hits = sum(1 for r in sub if r["binder_final"] == "True")
        print(f"     {n_cys} Cys  n={len(sub):4d}  {hits / len(sub):.1%} bind")


if __name__ == "__main__":
    main()
