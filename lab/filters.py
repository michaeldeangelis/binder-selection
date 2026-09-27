#!/usr/bin/env python3
"""Sequence-level liability filters for candidate binders.

These are the checks Anthropic's own campaign prompt requires before a design
sheet is finalized for order. Quoting `prompts/single_target/PD-L1.md`:

    "re-run liability checks (e.g. homopolymer runs, unpaired cysteines,
     hydrophobic patches), a monomer-foldability check at the threshold you
     froze for that target, and a diversity re-check on the final selected
     designs"

Plus Local Composition Perplexity (LCP), the sequence-design restraint from
Figure 1 of the same prompt (window 30, target ~2.32 nats).

DESIGN PRINCIPLE: these are GATES, not ranking features. Every one of them is
validated below against the 1,320 labeled designs, and most have within-target
AUC near 0.5 -- i.e. they carry almost no ranking signal. That is expected and
fine. Their job is to remove designs that will fail for avoidable physical
reasons (aggregation, disulfide scrambling, synthesis problems), not to pick
winners. Ranking is consensus ipSAE's job.

Run directly to validate every filter against the released wet-lab outcomes:
    python3 lab/filters.py

Stdlib only.
"""

import csv
import math
import os
from collections import Counter

HYDROPHOBIC = set("AVILMFWYC")
CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")

# Defaults. Tune against validation output rather than trusting these blindly.
MAX_HOMOPOLYMER = 4
HYDROPHOBIC_WINDOW = 7

# Calibrated against the 1,320 labeled designs, not guessed.
#   >0.857 (6/7) flags 497 designs (38%) and separates only 23.5% vs 28.8% --
#          far too aggressive for the signal it buys.
#   >0.9   flags 72 designs that bind at 18.1% vs 27.3% for the rest. Better
#          trade: catches near-fully-hydrophobic windows and little else.
MAX_HYDROPHOBIC_FRAC = 0.9

LCP_WINDOW = 30
# Anthropic's prompt cites a target of ~2.32 nats for Local Composition
# Perplexity. Applied as a floor on min-entropy-over-windows that is plainly
# the wrong quantity: the median design here sits at 1.95 nats, so a 2.32 floor
# rejects 98.7% of Anthropic's own designs. Their 2.32 must refer to a
# different normalization (a mean, or a per-position perplexity), which we
# could not pin down from the released figure.
#
# Empirically LCP carries no binding signal at all (within-target AUC 0.497,
# and the low-LCP tail binds slightly MORE, not less). So it is NOT a ranking
# feature and NOT a quality gate. We keep it only as an extreme-tail guard
# against sequences so repetitive that DNA synthesis or codon optimization
# would struggle. p1 of the observed distribution is 1.45 nats.
MIN_LCP_NATS = 1.45

MAX_LEN = 250  # Adaptyv precedent; per-challenge rules may be stricter


def max_homopolymer_run(seq):
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best if seq else 0


def max_hydrophobic_frac(seq, window=HYDROPHOBIC_WINDOW):
    """Worst hydrophobic fraction over any sliding window. Aggregation proxy."""
    if len(seq) < window:
        return sum(c in HYDROPHOBIC for c in seq) / max(len(seq), 1)
    counts = sum(c in HYDROPHOBIC for c in seq[:window])
    best = counts
    for i in range(window, len(seq)):
        counts += (seq[i] in HYDROPHOBIC) - (seq[i - window] in HYDROPHOBIC)
        best = max(best, counts)
    return best / window


def min_local_composition_perplexity(seq, window=LCP_WINDOW):
    """Lowest Shannon entropy (nats) of residue composition over any window.

    Low LCP means a stretch of low-complexity, compositionally biased sequence.
    Those regions are associated with aggregation and with codon-optimization
    and synthesis trouble. Returned in nats to match Anthropic's 2.32 target.
    """
    if len(seq) < window:
        window = len(seq)
    if window == 0:
        return 0.0
    best = None
    for i in range(len(seq) - window + 1):
        counts = Counter(seq[i:i + window])
        ent = -sum((n / window) * math.log(n / window) for n in counts.values())
        best = ent if best is None else min(best, ent)
    return best


def net_charge(seq, ph=7.4):
    """Crude net charge. His treated as partially protonated at 7.4."""
    return (seq.count("K") + seq.count("R") + 0.1 * seq.count("H")
            - seq.count("D") - seq.count("E"))


def evaluate(seq):
    """Return (passes, dict_of_failures, dict_of_metrics)."""
    seq = seq.strip().upper()
    n_cys = seq.count("C")
    m = {
        "length": len(seq),
        "n_cys": n_cys,
        "max_homopolymer": max_homopolymer_run(seq),
        "max_hydrophobic_frac_w7": round(max_hydrophobic_frac(seq), 3),
        "min_lcp_nats": round(min_local_composition_perplexity(seq), 3),
        "net_charge": round(net_charge(seq), 1),
    }
    fails = {}
    bad = set(seq) - CANONICAL
    if bad:
        fails["non_canonical"] = f"contains {sorted(bad)}"
    if not seq:
        fails["empty"] = "empty sequence"
    if len(seq) > MAX_LEN:
        fails["too_long"] = f"{len(seq)} > {MAX_LEN}"
    # The strongest empirical rule we found: 1 Cys binds at 9.1% vs 28.0% for 0.
    # An odd count guarantees a free thiol -> scrambling and aggregation.
    if n_cys % 2 == 1:
        fails["odd_cysteine"] = f"{n_cys} cysteines (unpaired thiol)"
    if m["max_homopolymer"] > MAX_HOMOPOLYMER:
        fails["homopolymer"] = f"run of {m['max_homopolymer']}"
    if m["max_hydrophobic_frac_w7"] > MAX_HYDROPHOBIC_FRAC:
        fails["hydrophobic_patch"] = f"{m['max_hydrophobic_frac_w7']:.2f} in w7"
    if m["min_lcp_nats"] < MIN_LCP_NATS:
        fails["low_complexity"] = f"LCP {m['min_lcp_nats']:.2f} < {MIN_LCP_NATS}"
    return (not fails), fails, m


# ---------------------------------------------------------------- validation

def _auc(pairs):
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
    rs = sum(ranks[k] for k in range(len(pairs)) if pairs[k][1])
    return (rs - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _within_target_auc(rows, fn):
    num, den = 0.0, 0.0
    for t in {r["target"] for r in rows}:
        sub = [(fn(r), r["binder_final"] == "True") for r in rows if r["target"] == t]
        sub = [(s, y) for s, y in sub if s is not None]
        a = _auc(sub)
        if a is None:
            continue
        n_pos = sum(1 for _, y in sub if y)
        w = n_pos * (len(sub) - n_pos)
        num += a * w
        den += w
    return num / den if den else None


def _validate():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "design_summary.csv")
    if not os.path.exists(path):
        print("design_summary.csv not found; run scripts/baseline_ranker.py first")
        return
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["binder_final"] in ("True", "False")]

    base = sum(1 for r in rows if r["binder_final"] == "True") / len(rows)
    print(f"Validating against {len(rows)} labeled designs (base rate {base:.1%} bind)\n")

    print("Ranking power of each metric (within-target AUC; 0.5 = no signal)")
    metrics = [
        ("n_cys", lambda r: -r["sequence"].count("C")),
        ("max_homopolymer", lambda r: -max_homopolymer_run(r["sequence"])),
        ("max_hydrophobic_frac_w7", lambda r: -max_hydrophobic_frac(r["sequence"])),
        ("min_lcp_nats", lambda r: min_local_composition_perplexity(r["sequence"])),
        ("net_charge", lambda r: net_charge(r["sequence"])),
    ]
    for name, fn in metrics:
        print(f"  {name:26} {_within_target_auc(rows, fn):.3f}")

    print("\nGate behaviour: what each rule would remove, and its binding rate")
    print(f"  {'rule':22}{'n flagged':>10}{'bind if flagged':>18}{'bind if clean':>16}")
    gates = [
        ("odd_cysteine", lambda s: s.count("C") % 2 == 1),
        ("homopolymer>4", lambda s: max_homopolymer_run(s) > MAX_HOMOPOLYMER),
        (f"hydrophobic w7>{MAX_HYDROPHOBIC_FRAC}", lambda s: max_hydrophobic_frac(s) > MAX_HYDROPHOBIC_FRAC),
        (f"LCP<{MIN_LCP_NATS}", lambda s: min_local_composition_perplexity(s) < MIN_LCP_NATS),
    ]
    for name, fn in gates:
        flag = [r for r in rows if fn(r["sequence"])]
        clean = [r for r in rows if not fn(r["sequence"])]
        fr = (sum(1 for r in flag if r["binder_final"] == "True") / len(flag)) if flag else None
        cr = sum(1 for r in clean if r["binder_final"] == "True") / len(clean)
        print(f"  {name:22}{len(flag):>10}"
              f"{(f'{fr:.1%}' if fr is not None else '-'):>18}{cr:>15.1%}")

    passed = [r for r in rows if evaluate(r["sequence"])[0]]
    removed = len(rows) - len(passed)
    pr = sum(1 for r in passed if r["binder_final"] == "True") / len(passed)
    print(f"\nAll gates combined: removes {removed}/{len(rows)} ({removed / len(rows):.1%}); "
          f"survivors bind {pr:.1%} vs {base:.1%} base")
    print("Gates are for removing avoidable physical failure, not for ranking.")


if __name__ == "__main__":
    _validate()
