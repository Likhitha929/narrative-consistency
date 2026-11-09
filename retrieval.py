"""
retrieval.py
============
Hybrid retriever: BM25 (keyword) + Semantic (MiniLM cosine) combined.

Why hybrid?
  Semantic search alone fails when the claim uses DIFFERENT vocabulary
  than the actual passage text (paraphrase / summarisation gap).
  BM25 catches exact keyword matches (e.g. "fifteen", "Cook", "telescope")
  that semantic search misses.

Score fusion:
  hybrid_score = alpha * semantic_score + (1 - alpha) * bm25_score
  alpha = 0.6  →  60% semantic, 40% keyword
"""

import logging
import re
from dataclasses import dataclass

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# BM25 weight: 0.0 = pure semantic, 1.0 = pure BM25
BM25_ALPHA = 0.6       # 60% semantic + 40% BM25


@dataclass
class RetrievedPassage:
    passage: object    # Passage dataclass from chunking.py
    score: float


def _tokenise(text: str):
    """Simple whitespace + lowercase tokeniser for BM25."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


class Retriever:
    def __init__(self, index, model_name=None):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # reuse the same model that built the index
        name        = model_name or index.model_name
        self.encoder    = SentenceTransformer(name, device=device)
        self.index      = index
        self.embeddings = index.embeddings   # (N, D) — already normalised

        # ── Build BM25 index (per book for speed) ─────────────────────────
        logger.info("building BM25 keyword index...")
        self._tokenised = [_tokenise(p.text) for p in index.passages]
        self._bm25      = BM25Okapi(self._tokenised)
        logger.info(f"BM25 index built over {len(self._tokenised)} passages")

    def retrieve(self, query: str, top_k: int = 5, book_name: str = None):
        """
        Hybrid retrieval: BM25 + Semantic → fused score.

        Parameters
        ----------
        query     : claim text to retrieve passages for
        top_k     : number of passages to return
        book_name : if given, restrict to passages from this book only
        """
        passages = self.index.passages
        N        = len(passages)

        # ── 1. Semantic scores ─────────────────────────────────────────────
        q_vec = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        sem_scores = self.embeddings @ q_vec         # (N,) cosine sim

        # ── 2. BM25 keyword scores ─────────────────────────────────────────
        query_tokens  = _tokenise(query)
        bm25_raw      = self._bm25.get_scores(query_tokens)   # (N,)

        # normalise BM25 to [0, 1] range
        bm25_max = bm25_raw.max()
        bm25_scores = bm25_raw / (bm25_max + 1e-9)

        # ── 3. Book filter ─────────────────────────────────────────────────
        mask = np.ones(N, dtype=bool)
        if book_name is not None:
            book_lower = book_name.lower()
            for i, p in enumerate(passages):
                if book_lower not in p.book_name.lower():
                    mask[i] = False

        # ── 4. Hybrid fusion ───────────────────────────────────────────────
        hybrid = BM25_ALPHA * sem_scores + (1.0 - BM25_ALPHA) * bm25_scores
        hybrid[~mask] = -np.inf

        # ── 5. Top-K selection ─────────────────────────────────────────────
        k        = min(top_k, int(mask.sum()))
        top_idxs = np.argpartition(hybrid, -k)[-k:]
        top_idxs = top_idxs[np.argsort(hybrid[top_idxs])[::-1]]

        results = [
            RetrievedPassage(passage=passages[i], score=float(hybrid[i]))
            for i in top_idxs
            if hybrid[i] > -np.inf
        ]
        return results
