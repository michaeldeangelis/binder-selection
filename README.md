# Binder selection

A learned ranker did not beat the mean of the interface scores. I kept the mean, plus a small term for how many optimization rounds a design had already survived.

The table is the public Anthropic binder-design release. The test is leave-one-target-out. Train on the other targets, score the held-out one, and judge the ranking inside that target. That is the decision I actually face: one new target, my own candidates, about twenty slots.

Mean ipSAE across predictors scores 0.767 AUC within target. Mean ipSAE plus half a within-target z-score of optimization rounds scores 0.782. Logistic regression and gradient boosting did not clear that line, so they are not the score.

Recorded in `lab/ranker.py`. Rerun it before treating the third digit as yours.

## Reproduce

`scripts/baseline_ranker.py` is the standard library. It downloads the table on first run.

```bash
python3 scripts/baseline_ranker.py
```

The learned comparison needs NumPy and scikit-learn. It reads the same table.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/baseline_ranker.py
python3 lab/ranker.py
```

## What else is here

`lab/score.py` is the score I kept, applied inside one target pool: z(mean ipSAE) + 0.5 z(rounds).

`lab/filters.py` is a set of sequence gates. I checked them on the same table. They are for throwing out broken sequences, not for ranking. Most of them sit near 0.5 AUC within a target.

`lab/select.py` applies diversity caps after the score. A short sheet stays short.

`scripts/reverse_engineer_selection.py` asks whether an observed selection rank is anything more than a sort on consensus ipSAE. Run it after the baseline script has fetched the table.

## Cite

Use the Cite this repository button. It reads `CITATION.cff`.

## License

[MIT](LICENSE). Copyright 2026 Michael DeAngelis.
