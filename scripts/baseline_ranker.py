#!/usr/bin/env python3
"""Baseline scorer comparison on the Anthropic binder-design release.

Reproduces Section 1 of PLAN.md using only the standard library, so it runs
on a bare Python install with no venv.

The question: how well does each in-silico score rank true wet-lab binders
*within a single target*? Within-target is the metric that matters, because
in the competition we choose among our own candidates for one revealed
target. Pooled-across-target AUC is inflated by between-target differences
in base binding rate, which tell us nothing actionable.

Data (1.2 MB):
  https://huggingface.co/datasets/Anthropic/claude-protein-binder-design
  -> data/tables/design_summary.csv
"""

import csv
import os
import statistics
import sys
import urllib.request

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "design_summary.csv")
CSV_URL = (
    "https://huggingface.co/datasets/Anthropic/claude-protein-binder-design/"
    "resolve/main/data/tables/design_summary.csv"
)
IPSAE_THRESHOLD = 0.3  # informal "confident interface" cutoff, for the frac-above feature


def load_rows():
    if not os.path.exists(CSV_PATH):
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        print(f"downloading {CSV_URL}", file=sys.stderr)
        urllib.request.urlretrieve(CSV_URL, CSV_PATH)
    with open(CSV_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))
    # Drop the 120 GDF-8 designs whose assay was inconclusive: no ground truth.
    return [r for r in rows if r["binder_final"] in ("True", "False")]


def num(row, col):
    try:
        return float(row[col])
    except (ValueError, KeyError):
        return None


def auc(pairs):
    """Mann-Whitney AUC with average ranks for ties. pairs: [(score, is_binder)]."""
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
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum = sum(ranks[k] for k in range(len(pairs)) if pairs[k][1])
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def within_target_auc(rows, score_fn):
    """AUC computed per target, averaged with weight n_pos*n_neg (pair count)."""
    num_, den = 0.0, 0.0
    for target in sorted({r["target"] for r in rows}):
        sub = [(score_fn(r), r["binder_final"] == "True") for r in rows if r["target"] == target]
        sub = [(s, y) for s, y in sub if s is not None]
        a = auc(sub)
        if a is None:
            continue
        n_pos = sum(1 for _, y in sub if y)
        weight = n_pos * (len(sub) - n_pos)
        num_ += a * weight
        den += weight
    return num_ / den if den else None


def main():
    rows = load_rows()
    predictors = [c.replace("ipsae_min_", "") for c in rows[0] if c.startswith("ipsae_min_")]

    n_pos = sum(1 for r in rows if r["binder_final"] == "True")
    print(f"{len(rows)} designs with a wet-lab call, {n_pos} binders "
          f"({n_pos / len(rows):.1%}), {len({r['target'] for r in rows})} targets, "
          f"{len(predictors)} predictors\n")

    def ipsae_vector(r):
        return [v for v in (num(r, "ipsae_min_" + p) for p in predictors) if v is not None]

    def dockq_vector(r):
        return [v for v in (num(r, "sc_dockq_" + p) for p in predictors) if v is not None]

    print("Per-predictor ipSAE")
    print(f"  {'predictor':<10} {'pooled':>8} {'within-target':>14}")
    for p in predictors:
        col = "ipsae_min_" + p
        pooled = auc([(num(r, col), r["binder_final"] == "True") for r in rows])
        within = within_target_auc(rows, lambda r, c=col: num(r, c))
        print(f"  {p:<10} {pooled:8.3f} {within:14.3f}")

    combos = [
        ("mean ipSAE", lambda r: (lambda v: sum(v) / len(v) if v else None)(ipsae_vector(r))),
        ("min ipSAE", lambda r: (lambda v: min(v) if v else None)(ipsae_vector(r))),
        (f"frac > {IPSAE_THRESHOLD}",
         lambda r: (lambda v: sum(1 for x in v if x > IPSAE_THRESHOLD) / len(v) if v else None)(ipsae_vector(r))),
        ("neg stdev (agreement)",
         lambda r: (lambda v: -statistics.pstdev(v) if len(v) > 1 else None)(ipsae_vector(r))),
        ("mean sc_DockQ", lambda r: (lambda v: sum(v) / len(v) if v else None)(dockq_vector(r))),
    ]

    print("\nCombined scorers (within-target AUC)")
    for name, fn in combos:
        print(f"  {name:<22} {within_target_auc(rows, fn):.3f}")

    best_single = max(within_target_auc(rows, lambda r, c="ipsae_min_" + p: num(r, c))
                      for p in predictors)
    mean_auc = within_target_auc(rows, combos[0][1])
    print(f"\nbest single predictor {best_single:.3f} vs mean ipSAE {mean_auc:.3f} "
          f"({mean_auc - best_single:+.3f})")
    print("Consensus across predictors beats any single predictor. "
          "This is the baseline the learned ranker must clear.")


if __name__ == "__main__":
    main()
