#!/usr/bin/env python3
"""Leave-one-target-out ranker: can a learned model beat consensus ipSAE?

THE PROTOCOL, AND WHY IT IS THIS AND NOT CROSS-VALIDATION
---------------------------------------------------------
On Sep 28 we get a target we have never seen, generate a pool of our own
candidates for it, and must choose ~20. So:

  * Evaluation is LEAVE-ONE-TARGET-OUT. Train on 15 targets, test on the
    16th, rotate. Pooled random CV would let the model learn "TREM2 designs
    tend to bind" (72/90) and "MBP designs never do" (0/90), which is real
    signal in the dataset and completely useless to us.

  * Features are Z-SCORED WITHIN TARGET. This is not leakage. At competition
    time we hold a pool of candidates for one target and can standardize
    across that pool, exactly as Anthropic's own rank_zscore does. Raw ipSAE
    is not comparable across targets or chain counts, so per-target
    standardization is the only honest way to pool 15 targets for training.

  * The headline metric is PRECISION@20 on the held-out target, not AUC.
    AUC scores the whole ranking; we only ever act on the top of it.

BASELINES TO BEAT (established in scripts/baseline_ranker.py):
    mean ipSAE                      within-target AUC 0.767
    mean ipSAE + 0.5 z(rounds)      within-target AUC 0.782

Any model that does not clear those under this protocol gets discarded.

Usage:  .venv/bin/python lab/ranker.py
"""

import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filters import (max_hydrophobic_frac, max_homopolymer_run,  # noqa: E402
                     min_local_composition_perplexity, net_charge)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_PATH = os.path.join(ROOT, "data", "raw", "design_summary.csv")
TOP_K = 20
SEED = 0


def load():
    with open(CSV_PATH, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["binder_final"] in ("True", "False")]
    predictors = [c.replace("ipsae_min_", "") for c in rows[0] if c.startswith("ipsae_min_")]
    return rows, predictors


def num(row, col):
    try:
        return float(row[col])
    except (ValueError, KeyError):
        return None


def featurize(rows, predictors):
    """Build the design matrix. Returns X, y, targets, names."""
    generators = sorted({r["generator"] for r in rows})
    names = []
    names += [f"ipsae_{p}" for p in predictors]
    names += [f"dockq_{p}" for p in predictors]
    names += ["ipsae_mean", "ipsae_min", "ipsae_max", "ipsae_spread", "ipsae_std",
              "n_agree_50", "dockq_mean", "rounds", "length", "n_cys",
              "hydrophobic_w7", "net_charge", "lcp", "homopolymer"]
    names += [f"gen_{g}" for g in generators]

    X, y, tg = [], [], []
    for r in rows:
        ips = [num(r, "ipsae_min_" + p) for p in predictors]
        dqs = [num(r, "sc_dockq_" + p) for p in predictors]
        ok = [v for v in ips if v is not None]
        if not ok:
            continue
        dq_ok = [v for v in dqs if v is not None]
        seq = r["sequence"]
        try:
            rounds = float(r["n_optimization_rounds"])
        except ValueError:
            rounds = np.nan

        row = []
        row += [v if v is not None else np.nan for v in ips]
        row += [v if v is not None else np.nan for v in dqs]
        row += [
            float(np.mean(ok)), float(min(ok)), float(max(ok)),
            float(max(ok) - min(ok)), float(np.std(ok)),
            float(sum(1 for v in ok if v > 0.5)),
            float(np.mean(dq_ok)) if dq_ok else np.nan,
            rounds,
            float(len(seq)),
            float(seq.count("C")),
            float(max_hydrophobic_frac(seq)),
            float(net_charge(seq)),
            float(min_local_composition_perplexity(seq)),
            float(max_homopolymer_run(seq)),
        ]
        row += [1.0 if r["generator"] == g else 0.0 for g in generators]

        X.append(row)
        y.append(1 if r["binder_final"] == "True" else 0)
        tg.append(r["target"])
    return np.array(X, float), np.array(y), np.array(tg), names


def zscore_within_target(X, targets, n_continuous):
    """Standardize continuous features within each target. Mirrors deployment."""
    Z = X.copy()
    for t in np.unique(targets):
        m = targets == t
        block = Z[m, :n_continuous]
        # A feature can be entirely absent for one target (afm3 was never run
        # on the Cas9 RNP). nanmean over an all-NaN column warns and returns
        # NaN, so compute per-column and leave those columns as NaN for the
        # model's own missing-value handling.
        with np.errstate(invalid="ignore"):
            allnan = np.all(np.isnan(block), axis=0)
            mu = np.zeros(block.shape[1])
            sd = np.ones(block.shape[1])
            if (~allnan).any():
                mu[~allnan] = np.nanmean(block[:, ~allnan], axis=0)
                sd[~allnan] = np.nanstd(block[:, ~allnan], axis=0)
        sd[sd < 1e-9] = 1.0
        Z[m, :n_continuous] = (block - mu) / sd
    return Z


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    keep = ~np.isnan(scores)
    scores, labels = scores[keep], labels[keep]
    n_pos, n_neg = labels.sum(), (1 - labels).sum()
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_at_k(scores, labels, k=TOP_K):
    scores = np.asarray(scores, float)
    order = np.argsort(-np.nan_to_num(scores, nan=-1e9))
    top = order[:min(k, len(order))]
    return float(np.mean(labels[top]))


def evaluate(score_fn, X, Xz, y, targets, name, results):
    """Run leave-one-target-out and record weighted AUC + mean precision@20."""
    per_target = {}
    num_, den = 0.0, 0.0
    precs, lifts = [], []
    for t in np.unique(targets):
        test = targets == t
        train = ~test
        if y[test].sum() == 0 or y[test].sum() == test.sum():
            continue  # no positives or no negatives -> AUC undefined
        s = score_fn(X[train], Xz[train], y[train], X[test], Xz[test])
        a = auc(s, y[test])
        if a is None:
            continue
        p = precision_at_k(s, y[test])
        base = y[test].mean()
        per_target[t] = (a, p, base, int(test.sum()))
        w = y[test].sum() * (test.sum() - y[test].sum())
        num_ += a * w
        den += w
        precs.append(p)
        lifts.append(p - base)
    results[name] = {
        "auc": num_ / den,
        "prec": float(np.mean(precs)),
        "lift": float(np.mean(lifts)),
        "per_target": per_target,
    }
    return results[name]


def main():
    rows, predictors = load()
    X, y, targets, names = featurize(rows, predictors)
    n_cont = len(names) - sum(1 for n in names if n.startswith("gen_"))
    Xz = zscore_within_target(X, targets, n_cont)

    print(f"{len(y)} designs · {y.sum()} binders ({y.mean():.1%}) · "
          f"{len(np.unique(targets))} targets · {X.shape[1]} features")
    print(f"protocol: leave-one-target-out, features z-scored within target, "
          f"precision@{TOP_K}\n")

    i_mean = names.index("ipsae_mean")
    i_rounds = names.index("rounds")
    results = {}

    # --- baselines (no training) -------------------------------------------
    evaluate(lambda Xtr, Ztr, ytr, Xte, Zte: Xte[:, i_mean],
             X, Xz, y, targets, "mean ipSAE", results)

    def consensus_plus_rounds(Xtr, Ztr, ytr, Xte, Zte):
        r = np.nan_to_num(Zte[:, i_rounds], nan=0.0)
        return Zte[:, i_mean] + 0.5 * r
    evaluate(consensus_plus_rounds, X, Xz, y, targets,
             "mean ipSAE + 0.5·rounds", results)

    # --- learned models ----------------------------------------------------
    def logreg(Xtr, Ztr, ytr, Xte, Zte):
        Ztr = np.nan_to_num(Ztr, nan=0.0)
        Zte = np.nan_to_num(Zte, nan=0.0)
        m = LogisticRegression(max_iter=2000, C=0.1, random_state=SEED)
        m.fit(Ztr, ytr)
        return m.predict_proba(Zte)[:, 1]
    evaluate(logreg, X, Xz, y, targets, "logistic regression", results)

    def gbm(Xtr, Ztr, ytr, Xte, Zte):
        m = HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.06,
            min_samples_leaf=25, l2_regularization=1.0, random_state=SEED)
        m.fit(Ztr, ytr)
        return m.predict_proba(Zte)[:, 1]
    evaluate(gbm, X, Xz, y, targets, "gradient boosting", results)

    # structure-only GBM: no generator, no sequence features
    struct_idx = [k for k, n in enumerate(names)
                  if n.startswith(("ipsae", "dockq", "n_agree"))]

    def gbm_struct(Xtr, Ztr, ytr, Xte, Zte):
        m = HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.06,
            min_samples_leaf=25, l2_regularization=1.0, random_state=SEED)
        m.fit(Ztr[:, struct_idx], ytr)
        return m.predict_proba(Zte[:, struct_idx])[:, 1]
    evaluate(gbm_struct, X, Xz, y, targets, "GBM (structure only)", results)

    # --- report ------------------------------------------------------------
    print(f"{'scorer':28}{'AUC':>8}{'P@20':>8}{'lift':>8}")
    print("-" * 52)
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["auc"]):
        print(f"{k:28}{v['auc']:8.3f}{v['prec']:8.1%}{v['lift']:+8.1%}")
    print("\nlift = precision@20 minus that target's own base rate")

    best = max(results.items(), key=lambda kv: kv[1]["auc"])
    base = results["mean ipSAE"]["auc"]
    print(f"\nbest: {best[0]} (AUC {best[1]['auc']:.3f}, "
          f"{best[1]['auc'] - base:+.3f} vs mean ipSAE)")

    print(f"\nper-target precision@{TOP_K}, best model ({best[0]}):")
    print(f"  {'target':16}{'n':>5}{'base':>8}{'P@20':>8}{'AUC':>8}")
    for t, (a, p, b, n) in sorted(best[1]["per_target"].items(),
                                  key=lambda kv: -kv[1][1]):
        print(f"  {t:16}{n:5d}{b:8.1%}{p:8.1%}{a:8.3f}")


if __name__ == "__main__":
    main()
