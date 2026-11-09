"""
consistency_scorer.py
=====================
Proper multi-signal consistency scorer for the crime investigation checker.

The binary P(consistent) from the NLI model alone is insufficient — it biases
toward "consistent" when uncertain. The real consistency metric combines:

  Score = α × P(consistent | passages, claim)      ← NLI model signal
        + β × semantic_similarity(claim, passages)  ← retrieval alignment signal

Decision logic:
  1. If semantic_similarity < SIM_THRESHOLD → retrieval failed → insufficient_evidence
  2. If P(contradict) > CONTRADICT_THRESHOLD → contradict
  3. Otherwise → consistent

Usage:
    from consistency_scorer import ConsistencyScorer
    scorer = ConsistencyScorer()
    result = scorer.score(claim, book_name)
    print(result)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer

import config
from custom_nli_model import NarrativeNLIModel

logger = logging.getLogger(__name__)

# Read thresholds from config so all files share the same values
CONTRADICT_THRESHOLD = config.CONTRADICT_THRESHOLD   # e.g. 0.20
SIM_THRESHOLD        = config.SIM_THRESHOLD           # e.g. 0.45


@dataclass
class ConsistencyResult:
    label: str                    # "consistent" | "contradict"
    p_consistent: float           # P(consistent) from NLI model
    p_contradict: float           # P(contradict) from NLI model
    semantic_sim: float           # cosine sim between claim and best passage
    nli_confidence: float         # max(p_consistent, p_contradict)
    combined_score: float         # weighted fusion score
    retrieval_quality: str        # "good" | "poor"
    top_passage: str              # best retrieved passage snippet

    def __str__(self):
        return (
            f"Label           : {self.label.upper()}\n"
            f"P(consistent)   : {self.p_consistent*100:.1f}%\n"
            f"P(contradict)   : {self.p_contradict*100:.1f}%\n"
            f"Semantic Sim    : {self.semantic_sim:.4f}  ({self.retrieval_quality} retrieval)\n"
            f"NLI Confidence  : {self.nli_confidence*100:.1f}%\n"
            f"Combined Score  : {self.combined_score:.4f}\n"
            f"Best Passage    : {self.top_passage[:120]}..."
        )


class ConsistencyScorer:
    """
    Full pipeline scorer: retrieval + NLI + threshold fusion.
    """

    def __init__(self, model_dir: Optional[Path] = None, retriever=None):
        model_dir = model_dir or config.MODEL_SAVE_DIR
        if not model_dir.exists():
            raise FileNotFoundError(f"No model at {model_dir} — run main.py first")

        device_str = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
        self.device    = torch.device(device_str)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model     = NarrativeNLIModel.load(str(model_dir)).to(self.device)
        self.model.eval()
        self.retriever = retriever
        logger.info(f"ConsistencyScorer ready | device={device_str}")

    def score(self, claim: str, book_name: str,
              top_k: int = None, verbose: bool = True) -> ConsistencyResult:
        """
        Score a single claim against the book.

        Parameters
        ----------
        claim     : the claim text to verify
        book_name : book to retrieve passages from
        top_k     : number of passages (default: config.TOP_K)
        verbose   : print result to console

        Returns
        -------
        ConsistencyResult with all signals
        """
        top_k = top_k or config.TOP_K

        if self.retriever is None:
            raise RuntimeError("Pass a retriever instance to ConsistencyScorer(retriever=...)")

        # ── 1. Retrieve passages ──────────────────────────────────────────────
        results    = self.retriever.retrieve(claim, top_k=top_k, book_name=book_name)
        sem_sim    = max(r.score for r in results) if results else 0.0
        top_text   = results[0].passage.text if results else ""
        passage_text = " [SEP] ".join(r.passage.text for r in results)

        # ── 2. NLI inference ──────────────────────────────────────────────────
        enc = self.tokenizer(
            passage_text, claim,
            max_length=config.MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            ids  = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            ttid = enc.get("token_type_ids",
                           torch.zeros_like(enc["input_ids"])).to(self.device)
            out   = self.model(ids, mask, ttid)
            probs = torch.softmax(out["logits"], dim=-1)[0].cpu().numpy()

        p_contradict = float(probs[0])
        p_consistent = float(probs[1])

        # ── 3. Combined consistency score ─────────────────────────────────────
        # Weight: 70% NLI signal, 30% retrieval alignment
        combined = 0.70 * p_consistent + 0.30 * sem_sim

        # ── 4. Decision logic ─────────────────────────────────────────────────
        retrieval_quality = "good" if sem_sim >= SIM_THRESHOLD else "poor"

        if retrieval_quality == "poor":
            # can't verify = default consistent (book doesn't say it's wrong)
            label = "consistent"
        elif p_contradict >= CONTRADICT_THRESHOLD:
            label = "contradict"
        else:
            label = "consistent"

        result = ConsistencyResult(
            label=label,
            p_consistent=p_consistent,
            p_contradict=p_contradict,
            semantic_sim=sem_sim,
            nli_confidence=max(p_consistent, p_contradict),
            combined_score=combined,
            retrieval_quality=retrieval_quality,
            top_passage=top_text,
        )

        if verbose:
            print(f"\n{'='*55}")
            print(f"  Claim: {claim[:80]}{'...' if len(claim)>80 else ''}")
            print(f"  Book:  {book_name}")
            print(f"{'─'*55}")
            print(result)

        return result

    @staticmethod
    def explain_score(result: ConsistencyResult) -> str:
        """Human-readable explanation of the consistency decision."""
        lines = []

        # Retrieval signal
        if result.retrieval_quality == "poor":
            lines.append(f"⚠️  Retrieval: LOW similarity ({result.semantic_sim:.2f}) — "
                         f"the specific details may not appear verbatim in the book.")
        else:
            lines.append(f"✅ Retrieval: GOOD similarity ({result.semantic_sim:.2f}) — "
                         f"relevant passage was found.")

        # NLI signal
        if result.p_contradict >= CONTRADICT_THRESHOLD:
            lines.append(f"🔴 NLI: Model sees contradiction signal "
                         f"(P(contradict)={result.p_contradict*100:.1f}% ≥ threshold {CONTRADICT_THRESHOLD*100:.0f}%)")
        elif result.p_contradict >= 0.10:
            lines.append(f"🟡 NLI: Weak contradiction signal "
                         f"(P(contradict)={result.p_contradict*100:.1f}%) — borderline case")
        else:
            lines.append(f"🟢 NLI: Strong consistency signal "
                         f"(P(consistent)={result.p_consistent*100:.1f}%)")

        # Combined score
        lines.append(f"📊 Combined Score: {result.combined_score:.3f} "
                     f"(0=contradict, 1=consistent)")
        lines.append(f"🏷️  Final label: {result.label.upper()}")

        return "\n".join(lines)


if __name__ == "__main__":
    # ── Quick demo ────────────────────────────────────────────────────────────
    import data_loader, chunking, embedding, retrieval

    books     = data_loader.load_books(config.BOOKS_DIR)
    passages  = chunking.chunk_all_books(books, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    emb_index = embedding.load_index(Path("cache/embedding_index"))
    retr      = retrieval.Retriever(emb_index)

    scorer = ConsistencyScorer(retriever=retr)

    test_cases = [
        # (claim, book, expected)
        ("Jacques Paganel is the secretary of the Geographical Society of Paris.",
         "In Search of the Castaways", "consistent"),
        ("Lord Glenarvan never found Harry Grant and the rescue was completely unsuccessful.",
         "In Search of the Castaways", "contradict"),
        ("Edmond Dantes was unjustly imprisoned in the Chateau d'If.",
         "The Count of Monte Cristo", "consistent"),
        ("Edmond Dantes was never imprisoned and lived a happy free life.",
         "The Count of Monte Cristo", "contradict"),
        ("At fifteen, Paganel read Cook's journal and papered bedroom walls with maps.",
         "In Search of the Castaways", "contradict"),  # NEI — not in book
    ]

    print("\n" + "="*55)
    print("  CONSISTENCY SCORER — DEMO")
    print("="*55)

    for claim, book, expected in test_cases:
        result = scorer.score(claim, book, verbose=True)
        print(ConsistencyScorer.explain_score(result))
        correct = result.label == expected
        print(f"  Expected: {expected} | {'✅ CORRECT' if correct else '❌ WRONG'}")

    print("\n" + "="*55)
    print("THRESHOLD GUIDE:")
    print(f"  CONTRADICT_THRESHOLD = {CONTRADICT_THRESHOLD}")
    print("  → Increase (0.30-0.50) for fewer false alarms")
    print("  → Decrease (0.10-0.15) to catch more subtle contradictions")
    print(f"  SIM_THRESHOLD = {SIM_THRESHOLD}")
    print("  → Increase if retrieval quality is high")
    print("  → Decrease if many claims can't be verified")
