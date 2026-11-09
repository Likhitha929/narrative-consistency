"""
baseline.py — claim-only baselines on the same leakage-free split as main.py.

If a model that never sees the book can classify a claim, the dataset leaks
label information through wording alone. The from-scratch NLI model is only
useful to the extent it beats these numbers.

    python baseline.py
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

import config
import data_loader
from splits import claim_groups, grouped_split


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"  {name:<44} acc={acc*100:6.2f}%   macro-F1={f1:.4f}")
    return acc, f1


def claim_only(texts, labels, tr, ev):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    texts = np.asarray(texts, dtype=object)
    Xtr = vec.fit_transform(texts[tr])          # vocabulary from train only
    clf = LogisticRegression(max_iter=3000, C=4.0).fit(Xtr, labels[tr])
    return clf.predict(vec.transform(texts[ev]))


def main():
    df = data_loader.load_train_csv(config.TRAIN_CSV)
    texts = df["content"].astype(str).tolist()
    chars = df["char"].fillna("").astype(str).tolist()
    with_char = [f"{c}: {t}" if c else t for c, t in zip(chars, texts)]
    labels = df["label"].str.strip().str.lower().map(config.LABEL2ID).values
    groups = claim_groups(texts, df["book_name"].tolist())

    print(f"\nrows: {len(df)} | distinct claims after removing paraphrased copies: {len(set(groups))}")

    # old (leaky) split — reproduces the number the README used to compare against
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.SEED)
    tr_old, va_old = next(sss.split(np.zeros(len(labels)), labels))
    leak = np.isin(groups[va_old], groups[tr_old]).mean()
    print(f"\n── OLD random 80/20 split ({leak*100:.1f}% of val rows have a near-copy in train)")
    evaluate("claim-only TF-IDF + LogReg (val)", labels[va_old], claim_only(texts, labels, tr_old, va_old))

    # new grouped split — identical to the one main.py trains on
    tr, va, te = grouped_split(labels, groups, config.VAL_FRAC, config.TEST_FRAC, config.SEED)
    print(f"\n── GROUPED split  train={len(tr)}  val={len(va)}  test={len(te)}  (no claim shared across parts)")
    majority = np.bincount(labels[tr]).argmax()
    evaluate("majority class (test)", labels[te], np.full(len(te), majority))
    evaluate("claim-only TF-IDF + LogReg (val)", labels[va], claim_only(texts, labels, tr, va))
    evaluate("claim-only TF-IDF + LogReg (test)", labels[te], claim_only(texts, labels, tr, te))
    evaluate("claim+character TF-IDF + LogReg (test)", labels[te], claim_only(with_char, labels, tr, te))
    print()


if __name__ == "__main__":
    main()
