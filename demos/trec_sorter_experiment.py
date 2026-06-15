"""
TREC (6-class) curriculum-split experiment using text sorters + a linear SVM.

For each text sorter we sort the questions, take the first `train_size` of each
class as train (easy -> train) and the rest as test (hard -> test), then train a
TF-IDF + LinearSVC classifier and measure test accuracy. We also run the reverse
(hard -> train) and a stratified-random baseline. A sorter that captures a real
difficulty axis should make the "easy -> train / hard -> test" setting score
*below* the random baseline.

Run from the repo root:
    python demos/trec_sorter_experiment.py
    python demos/trec_sorter_experiment.py --train-size 0.6 --seeds 5
"""

from __future__ import annotations

import argparse

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from splytters import sorted_stratified_split
from splytters.sorters import (
    character_length,
    lexical_diversity,
    readability_score,
    sentence_count,
    tokens_length,
    vocabulary_rarity,
)

# Text sorters used as difficulty axes. (perplexity_score is skipped here — it
# needs a GPT-2 download and is slow over thousands of texts.) Note: readability
# formulas need a minimum text length, so on TREC's short single-sentence
# questions many samples fall back to "inf" and tie at the hard end.
SORTERS = {
    "character_length": character_length,
    "tokens_length": tokens_length,
    "sentence_count": sentence_count,
    "lexical_diversity": lexical_diversity,
    "vocabulary_rarity": vocabulary_rarity,
    "readability(FK grade)": readability_score,
}


def load_trec() -> tuple[list[str], np.ndarray]:
    """Return all TREC questions and their 6-class coarse labels.

    Loads the HF auto-converted parquet branch of ``CogComp/trec`` — recent
    ``datasets`` releases dropped support for legacy loading scripts, which
    both ``trec`` and ``CogComp/trec`` still ship.
    """
    ds = load_dataset("CogComp/trec", revision="refs/convert/parquet")
    label_col = "coarse_label" if "coarse_label" in ds["train"].features else "label-coarse"
    texts, labels = [], []
    for split in ("train", "test"):
        texts.extend(ds[split]["text"])
        labels.extend(ds[split][label_col])
    return texts, np.asarray(labels)


def evaluate(train_idx, test_idx, texts, y) -> tuple[float, float]:
    """TF-IDF + LinearSVC; fit on train only, score on test."""
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform([texts[i] for i in train_idx])
    X_test = vec.transform([texts[i] for i in test_idx])
    clf = LinearSVC().fit(X_train, y[train_idx])
    pred = clf.predict(X_test)
    return accuracy_score(y[test_idx], pred), balanced_accuracy_score(y[test_idx], pred)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--seeds", type=int, default=5, help="seeds for the random baseline")
    args = parser.parse_args()

    texts, y = load_trec()
    print(f"TREC: {len(texts)} questions, {len(np.unique(y))} classes "
          f"(sizes: {np.bincount(y).tolist()})\n")

    # Stratified-random baseline (averaged over seeds) for reference.
    accs, baccs = [], []
    for seed in range(args.seeds):
        tr, te = train_test_split(
            np.arange(len(texts)), train_size=args.train_size, stratify=y, random_state=seed
        )
        a, b = evaluate(tr, te, texts, y)
        accs.append(a)
        baccs.append(b)
    rand_a, rand_b = float(np.mean(accs)), float(np.mean(baccs))

    header = f"{'sorter':20} {'easy->tr acc':>14} {'hard->tr acc':>14} {'easy->tr bacc':>14}"
    print(header)
    print("-" * len(header))
    print(f"{'random baseline':20} {rand_a:>14.3f} {'-':>14} {rand_b:>14.3f}")

    for name, sorter in SORTERS.items():
        ranking = sorter(texts)  # [(index, score), ...] ascending
        # easy -> train, hard -> test
        tr_e, te_e = sorted_stratified_split(ranking, y, train_size=args.train_size)
        easy_a, easy_b = evaluate(tr_e, te_e, texts, y)
        # hard -> train, easy -> test
        tr_h, te_h = sorted_stratified_split(
            ranking, y, train_size=args.train_size, largest_first=True
        )
        hard_a, _ = evaluate(tr_h, te_h, texts, y)
        print(f"{name:20} {easy_a:>14.3f} {hard_a:>14.3f} {easy_b:>14.3f}")

    print(
        "\nReading: when 'easy->train acc' falls below the random baseline, the "
        "sorter's ordering\nconcentrates harder questions in the test set -- i.e. "
        "it captures a real difficulty axis."
    )


if __name__ == "__main__":
    main()
