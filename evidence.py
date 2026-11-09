"""
evidence.py — sentence-level evidence selection.

The retriever returns 200-word passages. Three of them are ~840 tokens,
but the encoder reads 256, so with plain truncation the model only ever saw
the first quarter of the top passage. Here the top passages are split into
sentences, every sentence is scored against the claim, and the best ones
are packed into the token budget (kept in reading order):

    score(sentence) = cos(MiniLM(sentence), MiniLM(claim))
                      + NAME_BONUS   if the sentence names the claim's character

so the model sees the ~10 most relevant sentences from 8 passages instead of
the opening lines of one.
"""

import re

import numpy as np

CANDIDATE_PASSAGES = 8
NAME_BONUS = 0.10
_SPLIT = re.compile(r"(?<=[.!?;])\s+")
_STOP = {"the", "of", "de", "la", "le", "mr", "mrs", "lord", "lady", "sir", "captain", "and", "m"}


def _sentences(passage_text):
    for s in _SPLIT.split(passage_text):
        s = s.strip()
        if len(s.split()) >= 4:
            yield s


def _name_tokens(char):
    return {w for w in re.findall(r"[a-z]+", char.lower()) if len(w) > 2 and w not in _STOP}


_PASSAGE_CACHE = {}       # (book, chunk_id) -> (sentences, unit vectors); passages never change


def _passage_sentences(retriever, passage):
    key = (passage.book_name, passage.chunk_id)
    if key not in _PASSAGE_CACHE:
        sents = list(_sentences(passage.text))
        vecs = (retriever.encoder.encode(sents, convert_to_numpy=True, normalize_embeddings=True,
                                         show_progress_bar=False, batch_size=64)
                if sents else np.zeros((0, retriever.embeddings.shape[1]), dtype=np.float32))
        _PASSAGE_CACHE[key] = (sents, vecs)
    return _PASSAGE_CACHE[key]


def select_evidence(retriever, query, book_name, char, tokenizer, budget):
    """Return the evidence string (sentences joined by spaces) fitting `budget` tokens."""
    ret = retriever.retrieve(query=query, top_k=CANDIDATE_PASSAGES, book_name=book_name)
    seen, sents, vecs, order = set(), [], [], []
    for r in ret:
        ps, pv = _passage_sentences(retriever, r.passage)
        for j, s in enumerate(ps):
            key = s.lower()
            if key in seen:                      # passages overlap -> duplicate sentences
                continue
            seen.add(key)
            sents.append(s); vecs.append(pv[j])
            order.append((r.passage.chunk_id, j))
    if not sents:
        return ""
    q = retriever.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)[0]
    score = np.stack(vecs) @ q
    names = _name_tokens(char or "")
    if names:
        score = score + NAME_BONUS * np.array(
            [bool(names & set(re.findall(r"[a-z]+", s.lower()))) for s in sents])
    lens = [len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in sents]
    picked, used = [], 0
    for i in np.argsort(-score):
        if used + lens[i] <= budget:
            picked.append(i)
            used += lens[i]
        if used >= budget - 4:
            break
    picked.sort(key=lambda i: order[i])          # reading order
    return " ".join(sents[i] for i in picked)
