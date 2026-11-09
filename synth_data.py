"""
synth_data.py — self-supervised consistent / contradict pairs mined from the novels.

The labelled claims can be classified from wording alone (a claim-only
TF-IDF model scores ~72% on unseen claims), so a classifier trained only on
them has little reason to read the retrieved passages. These synthetic
examples can only be solved by comparing claim and passage:

  consistent  — a sentence from the book, lightly perturbed in a
                meaning-preserving way (filler added, a few words dropped)
  contradict  — the SAME perturbed sentence with one fact changed:
                two characters' roles swapped ("A betrayed B" -> "B betrayed A"),
                a name swapped for another from the same book, a number
                changed, or a negation flipped

Both versions of a sentence share the same wording artifacts, so the only
signal is the fact that disagrees with the book. Passages are retrieved by
the same hybrid retriever used for the real claims.

    python synth_data.py --n 4800      # -> cache/synth_samples.pkl
"""

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

import config
import data_loader
import embedding
import retrieval
from dataset_builder import ClassificationSample
from evidence import select_evidence

FILLERS_PRE = ["", "", "Moreover, ", "In fact, ", "Notably, ", "To add context, "]
FILLERS_SUF = ["", "", " according to the text", " in the narrative", " as the events unfold"]
NEG_ADD = {"was": "was not", "were": "were not", "had": "had not", "did": "did not",
           "could": "could not", "would": "would not", "is": "is not", "has": "has not"}
NEG_DEL = {"not ": "", "never ": "always "}
NUMBERS = ["two", "three", "four", "five", "six", "seven", "eight", "ten", "twelve", "twenty", "hundred"]
STOP_CAPS = {"I", "The", "He", "She", "It", "They", "We", "You", "But", "And", "A", "In", "On", "At",
             "Mr", "Mrs", "Monsieur", "Madame", "Mademoiselle", "Sir", "Lord", "Lady", "Captain",
             "God", "Oh", "Yes", "No", "What", "This", "That", "There", "His", "Her", "My", "If",
             "When", "Then", "As", "So", "Chapter", "M", "Well", "Ah", "Here", "Now", "Why"}


def entities(text, min_count=15):
    """Capitalised words that appear mid-sentence often enough to be names."""
    mid = re.findall(r"(?<=[a-z,;] )([A-Z][a-zé]+)", text)
    c = Counter(w for w in mid if w not in STOP_CAPS)
    return [w for w, k in c.items() if k >= min_count]


def sentences(text):
    for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)):
        w = s.split()
        if 10 <= len(w) <= 35 and '"' not in s and "“" not in s and "”" not in s:
            yield s.strip()


def soften(sent, rng):
    """Meaning-preserving noise so 'consistent' never means 'verbatim'."""
    w = sent.rstrip(".!?").split()
    for _ in range(int(rng.integers(0, 3))):
        cand = [i for i, t in enumerate(w) if t.lower() in {"very", "quite", "then", "now", "also", "still", "indeed", "already", "just"}]
        if not cand:
            break
        w.pop(int(rng.choice(cand)))
    return " ".join(w)


def corrupt(sent, ents, rng):
    """Change exactly one fact. Returns None if nothing can be changed."""
    ops = []
    present = [e for e in ents if re.search(rf"\b{e}\b", sent)]
    if present:
        ops.append("ent")
    if len(present) >= 2:
        ops += ["swap", "swap"]           # role swap: "A betrayed B" -> "B betrayed A"
    if re.search(r"\b(\d+|" + "|".join(NUMBERS) + r")\b", sent):
        ops.append("num")
    if re.search(r"\b(" + "|".join(NEG_ADD) + r")\b", sent) or re.search(r"\b(not|never)\b", sent):
        ops.append("neg")
    if not ops:
        return None
    op = rng.choice(ops)
    if op == "swap":
        a, b = rng.choice(present, size=2, replace=False)
        out = re.sub(rf"\b{a}\b", "\x00", sent, count=1)
        out = re.sub(rf"\b{b}\b", a, out, count=1)
        return out.replace("\x00", b)
    if op == "ent":
        e = rng.choice(present)
        new = rng.choice([x for x in ents if x != e])
        return re.sub(rf"\b{e}\b", new, sent, count=1)
    if op == "num":
        def swap(m):
            t = m.group(0)
            if t.isdigit():
                return str(int(t) + int(rng.integers(1, 9)) * (1 if rng.random() < .5 or int(t) < 10 else -1))
            return rng.choice([n for n in NUMBERS if n != t])
        return re.sub(r"\b(\d+|" + "|".join(NUMBERS) + r")\b", swap, sent, count=1)
    m = re.search(r"\b(not|never) ", sent)
    if m:
        return sent[:m.start()] + NEG_DEL[m.group(0)] + sent[m.end():]
    m = re.search(r"\b(" + "|".join(NEG_ADD) + r")\b", sent)
    if m is None:
        return None
    return sent[:m.start()] + NEG_ADD[m.group(1)] + sent[m.end():]


def build(n, seed=0):
    rng = np.random.default_rng(seed)
    books = data_loader.load_books(config.BOOKS_DIR)
    labelled = set(data_loader.load_train_csv(config.TRAIN_CSV)["book_name"].str.strip())
    retriever = retrieval.Retriever(embedding.load_index(Path("cache/embedding_index")))
    evidence_tok = None
    if getattr(config, "SENTENCE_EVIDENCE", False):
        from book_tokenizer import load_tokenizer
        evidence_tok = load_tokenizer()
    out, per_book = [], n // (2 * len(labelled))
    for title, text in books.items():
        book = next((b for b in labelled if b.lower() in title.lower() or title.lower() in b.lower()), None)
        if book is None:
            continue
        ents = entities(text)
        sents = list(sentences(text))
        rng.shuffle(sents)
        made = 0
        for s in sents:
            if made >= per_book:
                break
            bad = corrupt(s, ents, rng)
            if bad is None or bad == s:
                continue
            pre, suf = rng.choice(FILLERS_PRE), rng.choice(FILLERS_SUF)
            subj = next((e for e in ents if re.search(rf"\b{e}\b", s)), rng.choice(ents))
            for lab, sent in ((1, s), (0, bad)):
                claim = f"{subj}: {pre}{soften(sent, rng)}{suf}."
                if evidence_tok is not None:       # same sentence-level evidence as real claims
                    budget = config.MAX_SEQ_LEN - 3 - len(evidence_tok(claim, add_special_tokens=False)["input_ids"])
                    text_in = select_evidence(retriever, claim, book, subj, evidence_tok, budget) + " [SEP] " + claim
                else:
                    ret = retriever.retrieve(query=claim, top_k=config.TOP_K, book_name=book)
                    text_in = " [SEP] ".join([r.passage.text for r in ret] + [claim])
                out.append(ClassificationSample(sample_id=-1, input_text=text_in, label=lab,
                                                book_name=book, claim=claim))
            made += 1
        print(f"{book}: {made} sentence pairs, {len(ents)} entities")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="total synthetic samples (half contradict)")
    ap.add_argument("--out", default="cache/synth_samples.pkl")
    a = ap.parse_args()
    S = build(a.n)
    pickle.dump(S, open(a.out, "wb"))
    for s in S[:4]:
        print(s.label, "|", s.claim)
    print(f"{len(S)} samples -> {a.out}")
