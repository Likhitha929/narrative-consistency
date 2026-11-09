"""
splits.py — leakage-free train / val / test splitting.

The training CSV was augmented: most original claims appear several times with
discourse fillers bolted on ("Moreover, ...", "To add context, ...",
"... which is critical to note..", "... according to the text.."). A plain
random split puts near-copies of the same claim on both sides of the split, so
validation accuracy measures memorisation, not generalisation.

This module clusters each claim with its paraphrased copies and splits by
cluster (StratifiedGroupKFold), so every copy of a claim lands in exactly one
of train / val / test.
"""

import re
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# Fillers added by the augmentation script (observed in combined_train_v2.csv).
_PREFIXES = [
    "to add context", "as it turns out", "as a matter of fact", "in particular",
    "notably", "significantly", "interestingly", "furthermore", "moreover",
    "importantly", "additionally", "in fact", "indeed", "crucially",
    "it is worth noting that", "it should be noted that", "remarkably",
]
_SUFFIXES = [
    "throughout the course of events", "which is critical to note",
    "as detailed in the story", "according to the text", "as the events unfold",
    "to be precise", "in this context", "in the narrative", "as described",
]
_PREFIX_RE = re.compile(r"^(?:" + "|".join(map(re.escape, _PREFIXES)) + r")\s*,?\s*")
_SUFFIX_RE = re.compile(r"\s*(?:" + "|".join(map(re.escape, _SUFFIXES)) + r")\s*$")
_TOKEN_RE  = re.compile(r"[a-z0-9]+")


def normalise_claim(text: str) -> str:
    """Strip augmentation fillers, lowercase, drop trailing dots."""
    s = str(text).strip().lower()
    s = re.sub(r"[.\s]+$", "", s)
    for _ in range(2):                       # fillers occasionally stack
        s = _PREFIX_RE.sub("", s)
        s = _SUFFIX_RE.sub("", s)
        s = re.sub(r"[.\s]+$", "", s)
    return s


def _tokens(text: str) -> frozenset:
    return frozenset(_TOKEN_RE.findall(normalise_claim(text)))


def claim_groups(claims, books, jaccard=0.6):
    """
    Assign a group id to every claim. Two claims from the same book share a
    group if their normalised token sets have Jaccard similarity >= `jaccard`
    (union-find, so groups are transitive).
    """
    n = len(claims)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    toks = [_tokens(c) for c in claims]
    by_book = defaultdict(list)
    for i, b in enumerate(books):
        by_book[str(b).strip().lower()].append(i)

    for idx in by_book.values():
        # inverted index: token -> rows, to avoid comparing every pair
        inv = defaultdict(list)
        for i in idx:
            for t in toks[i]:
                inv[t].append(i)
        for i in idx:
            if not toks[i]:
                continue
            cand = set()
            for t in toks[i]:
                if len(inv[t]) < 200:        # skip very common words
                    cand.update(inv[t])
            for j in cand:
                if j <= i:
                    continue
                a, b = toks[i], toks[j]
                if len(a & b) / len(a | b) >= jaccard:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri

    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots])


def grouped_split(labels, groups, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Stratified, grouped three-way split. Returns (train_idx, val_idx, test_idx).
    No group appears in more than one part.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    idx = np.arange(len(labels))

    # carve off test
    n_test = max(2, round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    rest, test = next(sgkf.split(idx, labels, groups))

    # carve val out of the remainder
    n_val = max(2, round((1 - test_frac) / val_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed + 1)
    tr_rel, va_rel = next(sgkf.split(rest, labels[rest], groups[rest]))
    train, val = rest[tr_rel], rest[va_rel]

    assert not (set(groups[train]) & set(groups[val]))
    assert not (set(groups[train]) & set(groups[test]))
    assert not (set(groups[val]) & set(groups[test]))
    return train, val, test
